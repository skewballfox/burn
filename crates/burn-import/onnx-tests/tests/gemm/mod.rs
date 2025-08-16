use crate::include_models;
include_models!(gemm, gemm_no_c, gemm_non_unit_alpha_beta);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn gemm_test() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm::Model::<Backend>::new(&device);

        // Create input matrices
        let a =
            Tensor::<Backend, 2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b =
            Tensor::<Backend, 2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);
        let c = 1.0;

        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0 + 1.0, 22.0 + 1.0] = [20.0, 23.0]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0 + 1.0, 50.0 + 1.0] = [44.0, 51.0]
        let expected = Tensor::<Backend, 2>::from_data(
            TensorData::from([[20.0, 23.0], [44.0, 51.0]]),
            &device,
        );

        // Run the model
        let output = model.forward(a, b, c);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn gemm_test_non_unit_alpha_beta() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm_non_unit_alpha_beta::Model::<Backend>::new(&device);

        // Create input matrices
        let a =
            Tensor::<Backend, 2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b =
            Tensor::<Backend, 2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);
        let c = 1.0;

        // Alpha = Beta = 0.5
        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0 * .5 + 1.0 * .5, 22.0 * .5 + 1.0 * .5] = [10.0, 11.5]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0 * .5 + 1.0 * .5, 50.0 * .5 + 1.0 * .5] = [22.0, 25.5]
        let expected = Tensor::<Backend, 2>::from_data(
            TensorData::from([[10.0, 11.5], [22.0, 25.5]]),
            &device,
        );

        // Run the model
        let output = model.forward(a, b, c);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn gemm_test_no_c() {
        // Test for GEMM
        let device = Default::default();
        let model = gemm_no_c::Model::<Backend>::new(&device);

        // Create input matrices
        let a =
            Tensor::<Backend, 2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let b =
            Tensor::<Backend, 2>::from_data(TensorData::from([[5.0, 6.0], [7.0, 8.0]]), &device);

        // Alpha = Beta = 0.5
        // Expected result of matrix multiplication
        // [1.0, 2.0] × [5.0, 6.0] = [1×5 + 2×7, 1×6 + 2×8] = [19.0, 22.0]
        // [3.0, 4.0] × [7.0, 8.0] = [3×5 + 4×7, 3×6 + 4×8] = [43.0, 50.0]
        let expected = Tensor::<Backend, 2>::from_data(
            TensorData::from([[19.0, 22.0], [43.0, 50.0]]),
            &device,
        );

        // Run the model
        let output = model.forward(a, b);

        // Verify the output
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
