use burn_tensor::Element;
use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

#[ctor(unsafe)]
fn init_device_settings() {
    let mut device = burn_tensor::Device::default();
    device
        .configure(
            burn_tensor::DeviceConfig::default()
                .float_dtype(<FloatElem as Element>::dtype())
                .int_dtype(<IntElem as Element>::dtype()),
        )
        .unwrap();

    // Every Flex tensor buffer comes from the registered allocator, so the suite checks that path.
    #[cfg(feature = "flex-buffer-allocator")]
    {
        static SYSTEM: burn_flex::FromGlobalAlloc<std::alloc::System> =
            burn_flex::FromGlobalAlloc(std::alloc::System);
        burn_flex::set_allocator(&SYSTEM);
    }
}

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use burn_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<D>;
    pub type TestTensorInt<const D: usize> = Tensor<D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<D, burn_tensor::Bool>;
}

#[allow(unused)]
pub use prelude::*;
