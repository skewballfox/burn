use burn_backend::{
    AutodiffBackend, Backend, ComplexTensorBackend,
    ops::ComplexTensorOps,
    tensor::{BoolTensor, ComplexTensor, Device, IntTensor},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::ComplexAutodiffTensor};

impl<B: ComplexTensorBackend + Backend, C: CheckpointStrategy> ComplexTensorBackend
    for Autodiff<B, C>
{
    type InnerBackend = Self;

    fn complex_from_real_data(
        data: burn_std::TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_from_real_data(data, device))
    }

    fn complex_from_imag_data(
        data: burn_std::TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_from_imag_data(data, device))
    }

    fn complex_from_interleaved_data(
        data: burn_std::TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_from_interleaved_data(data, device))
    }

    fn complex_from_parts_data(
        real_data: burn_std::TensorData,
        imag_data: burn_std::TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_from_parts_data(real_data, imag_data, device))
    }
}

impl<B: ComplexTensorBackend + Backend, C: CheckpointStrategy> ComplexTensorOps<Self>
    for Autodiff<B, C>
{
    fn complex_device(tensor: &ComplexTensor<Self>) -> Device<Self> {
        B::complex_device(&tensor.primitive)
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        B::complex_into_interleaved_data(tensor.primitive).await
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<(burn_std::TensorData, burn_std::TensorData), burn_std::ExecutionError> {
        B::complex_into_split_data(tensor.primitive).await
    }

    fn complex_squared_norm(
        tensor: ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_std::Distribution,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_zeros(
        shape: burn_std::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_ones(
        shape: burn_std::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_full(
        shape: burn_std::Shape,
        fill_value: burn_std::Scalar,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_to_device(
        tensor: ComplexTensor<Self>,
        device: &Device<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    async fn complex_into_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        todo!()
    }

    fn complex_reshape(tensor: ComplexTensor<Self>, shape: burn_std::Shape) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_transpose(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_add(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sub(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_div(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_neg(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_conj(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_recip(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_finv(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_real(tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_imag(tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_into_float(
        tensor: ComplexTensor<Self>,
        dtype: burn_std::FloatDType,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_into_int(
        tensor: ComplexTensor<Self>,
        dtype: burn_std::IntDType,
    ) -> burn_backend::tensor::IntTensor<Self> {
        todo!()
    }

    fn complex_abs(tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_arg(tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_from_parts(
        real: burn_std::TensorData,
        imag: burn_std::TensorData,
        device: &burn_backend::tensor::Device<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_polar(
        magnitude: burn_backend::tensor::FloatTensor<Self>,
        phase: burn_backend::tensor::FloatTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_exp(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_log(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sqrt(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_tan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_acos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_acosh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cast(
        tensor: ComplexTensor<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_asin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_asinh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atanh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atan2(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_select(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice(
        tensor: ComplexTensor<Self>,
        slices: &[burn_std::Slice],
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<Self>,
        ranges: &[burn_std::Slice],
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_nd(
        tensor: ComplexTensor<Self>,
        indices: IntTensor<Self>,
        value: ComplexTensor<Self>,
        reduction: burn_std::IndexingUpdateOp,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_not_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_cat(
        tensors: alloc::vec::Vec<ComplexTensor<Self>>,
        dim: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_any(
        tensor: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_any_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_all(
        tensor: ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_all_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_permute(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_expand(tensor: ComplexTensor<Self>, shape: burn_std::Shape) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_flip(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_unfold(
        tensor: ComplexTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_select_add(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<Self>,
        rhs: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_mask_where(
        tensor: ComplexTensor<Self>,
        mask: BoolTensor<Self>,
        source: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<Self>,
        mask: BoolTensor<Self>,
        value: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: IntTensor<Self>,
        values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sign(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc_scalar(lhs: ComplexTensor<Self>, rhs: burn_std::Scalar) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf(
        lhs: ComplexTensor<Self>,
        rhs: burn_backend::tensor::FloatTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powi(
        lhs: ComplexTensor<Self>,
        rhs: burn_backend::tensor::IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf_scalar(lhs: ComplexTensor<Self>, rhs: burn_std::Scalar) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_matmul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumsum(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumprod(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        todo!()
    }
}
