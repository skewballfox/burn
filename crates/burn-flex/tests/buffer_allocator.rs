//! With an allocator registered, the tensor memory of inference operations comes from it, not
//! from the global allocator.
//!
//! A global allocator that records allocations larger than tensor metadata catches any buffer
//! that bypasses the registered allocator. The test runs in a binary of its own, because the
//! registered allocator is global state, and in one test function, so that no other test
//! allocates while the recorder is armed. Run it without `rayon`:
//!
//! ```sh
//! cargo test -p burn-flex --no-default-features --features std,buffer-allocator --test buffer_allocator
//! ```
//!
//! Set `BURN_FLEX_TRACE_BYPASS=1` to print the backtrace of each buffer that bypasses it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use burn_backend::TensorData;
use burn_backend::ops::{ActivationOps, FloatTensorOps, ModuleOps};
#[cfg(feature = "buffer-allocator-gemm")]
use burn_backend::ops::{ConvOptions, QTensorOps};
#[cfg(feature = "buffer-allocator-gemm")]
use burn_backend::quantization::QuantScheme;
#[cfg(feature = "buffer-allocator-gemm")]
use burn_backend::{FloatDType, TensorPrimitive};
use burn_flex::{Flex, FlexTensor};
use burn_std::{AllocationError, BoolDType, BufferAllocator, FromGlobalAlloc};

/// Allocations up to this size are metadata (reference counts, allocation controllers, shapes)
/// or arguments of the operations (`float_cat` takes a `Vec` of two tensors). Every tensor in
/// the test is larger.
const METADATA_BYTES: usize = 2 * size_of::<FlexTensor>() + 64;

/// The number of outputs of [`infer`].
const OUTPUTS: usize = if cfg!(feature = "buffer-allocator-gemm") {
    9
} else {
    5
};

/// Passes every allocation to [`System`], and records those above [`METADATA_BYTES`] while armed.
///
/// A global allocator must not unwind, so it records instead of panicking.
struct Recorder {
    armed: AtomicBool,
    trace: AtomicBool,
    count: AtomicUsize,
    largest: AtomicUsize,
}

impl Recorder {
    fn record(&self, size: usize) {
        if size > METADATA_BYTES && self.armed.load(Ordering::Relaxed) {
            self.count.fetch_add(1, Ordering::Relaxed);
            self.largest.fetch_max(size, Ordering::Relaxed);
            if self.trace.load(Ordering::Relaxed) {
                // Disarmed while the backtrace allocates.
                self.armed.store(false, Ordering::Relaxed);
                let backtrace = std::backtrace::Backtrace::force_capture();
                eprintln!("bypassed the allocator: {size} bytes\n{backtrace}");
                self.armed.store(true, Ordering::Relaxed);
            }
        }
    }
}

// SAFETY: Every call goes to `System` unchanged.
unsafe impl GlobalAlloc for Recorder {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Recorder = Recorder {
    armed: AtomicBool::new(false),
    trace: AtomicBool::new(false),
    count: AtomicUsize::new(0),
    largest: AtomicUsize::new(0),
};

/// The arena: memory from [`System`] directly, so it bypasses [`GLOBAL`], with a count of the
/// live bytes.
struct Arena {
    inner: FromGlobalAlloc<System>,
    live: AtomicUsize,
    allocations: AtomicUsize,
}

// SAFETY: Every call goes to `inner`, which upholds the contract.
unsafe impl BufferAllocator for Arena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocationError> {
        let ptr = self.inner.allocate(layout)?;
        self.live.fetch_add(layout.size(), Ordering::Relaxed);
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.live.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { self.inner.deallocate(ptr, layout) }
    }
}

static ARENA: Arena = Arena {
    inner: FromGlobalAlloc(System),
    live: AtomicUsize::new(0),
    allocations: AtomicUsize::new(0),
};

fn ramp(shape: &[usize], scale: f32) -> TensorData {
    let n = shape.iter().product::<usize>();
    let values = (0..n).map(|i| (i as f32 * scale).sin()).collect::<Vec<_>>();
    TensorData::new(values, shape.to_vec())
}

/// The inputs, built before the recorder is armed: the application owns this memory.
struct Inputs {
    x: TensorData,
    w1: TensorData,
    w2: TensorData,
    gamma: TensorData,
    indices: TensorData,
    #[cfg(feature = "buffer-allocator-gemm")]
    image: TensorData,
    #[cfg(feature = "buffer-allocator-gemm")]
    kernel: TensorData,
}

impl Inputs {
    fn new() -> Self {
        Self {
            x: ramp(&[8, 64], 0.11),
            w1: ramp(&[8, 64], 0.37),
            w2: ramp(&[8, 64], 0.53),
            gamma: ramp(&[64], 0.07),
            indices: TensorData::new((0..8 * 64).map(|i| (i * 7 % 64) as i64).collect(), [8, 64]),
            #[cfg(feature = "buffer-allocator-gemm")]
            image: ramp(&[1, 4, 16, 16], 0.19),
            #[cfg(feature = "buffer-allocator-gemm")]
            kernel: ramp(&[8, 4, 3, 3], 0.29),
        }
    }
}

/// An inference pass over the operation families that the allocator covers.
///
/// The inputs and `outputs`, with room for every output, are made before the call, so the test
/// itself does not allocate.
fn infer(inputs: Inputs, outputs: &mut Vec<FlexTensor>) {
    let x = FlexTensor::from_data(inputs.x);
    let w1 = FlexTensor::from_data(inputs.w1);
    let w2 = FlexTensor::from_data(inputs.w2);
    let gamma = FlexTensor::from_data(inputs.gamma);
    let indices = FlexTensor::from_data(inputs.indices);

    let hidden = Flex::relu(Flex::float_add(x.clone(), w1.clone()));
    let scaled = Flex::float_mul_scalar(Flex::float_mul(hidden, w2.clone()), 0.5.into());
    let normed = Flex::layer_norm(scaled, gamma, None, 1e-5);
    let probs = Flex::softmax(normed.clone(), 1);
    let summed = Flex::float_sum_dim(probs.clone(), 1);
    let joined = Flex::float_cat(vec![probs.clone(), normed.clone()], 0);
    let gathered = Flex::float_gather(1, normed.clone(), indices);
    let mask = Flex::float_greater_elem(normed.clone(), 0.0.into(), BoolDType::Native);
    let masked = Flex::float_mask_fill(gathered.clone(), mask, 0.0.into());
    let transposed = Flex::float_swap_dims(x, 0, 1);
    let contiguous = Flex::float_exp(transposed);

    outputs.extend([probs, summed, joined, masked, contiguous]);

    #[cfg(feature = "buffer-allocator-gemm")]
    {
        let image = FlexTensor::from_data(inputs.image);
        let kernel = FlexTensor::from_data(inputs.kernel);
        let conv = Flex::conv2d(
            image,
            kernel,
            None,
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        );
        let pooled = Flex::max_pool2d(conv.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false);

        let product = Flex::float_matmul(w1.clone(), Flex::float_transpose(w2.clone()));
        let quantized = Flex::quantize_dynamic(w2, &QuantScheme::default());
        let q_product = match Flex::q_matmul(
            TensorPrimitive::Float(w1),
            TensorPrimitive::QFloat(Flex::q_transpose(quantized)),
        ) {
            TensorPrimitive::Float(tensor) => tensor,
            TensorPrimitive::QFloat(tensor) => Flex::dequantize(tensor, FloatDType::F32),
        };
        outputs.extend([conv, pooled, product, q_product]);
    }
}

fn run(inputs: Inputs) -> Vec<FlexTensor> {
    let mut outputs = Vec::with_capacity(OUTPUTS);
    infer(inputs, &mut outputs);
    outputs
}

fn values(outputs: Vec<FlexTensor>) -> Vec<Vec<f32>> {
    outputs
        .into_iter()
        .map(|tensor| tensor.into_data().convert::<f32>().into_vec().unwrap())
        .collect()
}

#[test]
fn inference_memory_comes_from_the_registered_allocator() {
    let trace = std::env::var_os("BURN_FLEX_TRACE_BYPASS").is_some_and(|value| value == "1");
    GLOBAL.trace.store(trace, Ordering::Relaxed);

    // On the global heap: the reference values, and the warm-up of lazy statics.
    let expected = values(run(Inputs::new()));

    burn_flex::set_allocator(&ARENA);
    let inputs = Inputs::new();
    let mut outputs = Vec::with_capacity(OUTPUTS);
    GLOBAL.armed.store(true, Ordering::Relaxed);
    infer(inputs, &mut outputs);
    GLOBAL.armed.store(false, Ordering::Relaxed);

    let bypassed = GLOBAL.count.load(Ordering::Relaxed);
    let largest = GLOBAL.largest.load(Ordering::Relaxed);
    assert_eq!(
        bypassed, 0,
        "{bypassed} buffers came from the global allocator (largest: {largest} bytes)"
    );
    assert!(ARENA.allocations.load(Ordering::Relaxed) > 0);

    assert_eq!(values(outputs), expected);
    assert_eq!(ARENA.live.load(Ordering::Relaxed), 0, "arena memory leaked");
}
