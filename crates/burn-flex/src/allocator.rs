//! The allocator that Flex tensor memory comes from.
//!
//! By default, every tensor buffer comes from the global allocator. With the `buffer-allocator`
//! feature, an application can register one [`BufferAllocator`] at start-up, for example an
//! arena in a fixed memory region of a microcontroller. Then the storage of every tensor that
//! Flex creates, and the temporary buffers of its kernels, come from that allocator.
//!
//! Tensors made from existing data ([`FlexTensor::from_data`](crate::FlexTensor::from_data))
//! keep shared, static and file memory as it is, so weights in flash stay zero-copy. Their
//! heap memory is copied into the allocator.
//!
//! Not yet covered: the backward-only operations (the `*_backward` operations of pooling,
//! convolution, deformable convolution and interpolation), the FFT, and the per-tensor metadata
//! (the reference count and the allocation controller), which stay on the global heap, as do the
//! packing buffers that `gemm` allocates for large matrix multiplications.

use burn_std::BufferAllocator;

static ALLOCATOR: spin::Once<&'static dyn BufferAllocator> = spin::Once::new();

/// Registers the allocator for the memory of Flex tensors and of the temporary buffers of Flex
/// kernels.
///
/// Call it once, before the first tensor is created. Tensors created before the call keep their
/// memory in the global allocator.
///
/// When the allocator has no memory for a tensor, the operation that creates the tensor panics.
///
/// ```ignore
/// static ARENA: FromGlobalAlloc<embedded_alloc::LlffHeap> =
///     FromGlobalAlloc(embedded_alloc::LlffHeap::empty());
///
/// burn_flex::set_allocator(&ARENA);
/// ```
///
/// # Panics
///
/// When an allocator is already registered.
pub fn set_allocator(allocator: &'static dyn BufferAllocator) {
    let mut registered = false;
    ALLOCATOR.call_once(|| {
        registered = true;
        allocator
    });
    assert!(registered, "burn-flex: an allocator is already registered");
}

/// The registered allocator, if any.
pub fn allocator() -> Option<&'static dyn BufferAllocator> {
    ALLOCATOR.get().copied()
}
