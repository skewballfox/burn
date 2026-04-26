#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_tensor::{Complex, ComplexElement, Tensor, TensorData, backend::Backend};
        type TestBackend = burn_flex::Flex;
        pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        pub type TestTensorComplex<const D: usize> = burn_tensor::Tensor<TestBackend, D, Complex>;

        burn_complex::tests::ops::testgen_ops!();
    };
}
