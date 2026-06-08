use burn_backend::{
    ComplexScalar, ComplexTensorBackend, Distribution, InterleavedLayout, Shape, TensorData,
    TensorMetadata, ops::ComplexTensorOps, tensor::{Device, IntTensor},
};

use crate::{IntoKind, LibTorch, TchShape, TchTensor, ops::TchOps};

impl ComplexTensorBackend for LibTorch {
    type InnerBackend = Self;

    type Layout = InterleavedLayout;

    fn complex_from_real_data(
        data: burn_backend::TensorData,
        device: &Self::Device,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_imag_data(
        data: burn_backend::TensorData,
        device: &Self::Device,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_interleaved_data(
        data: burn_backend::TensorData,
        device: &Self::Device,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_parts_data(
        real_data: burn_backend::TensorData,
        imag_data: burn_backend::TensorData,
        device: &Self::Device,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }
}

impl ComplexTensorOps<Self> for LibTorch {
    fn complex_device(tensor: &burn_backend::ComplexTensor<Self>) -> Device<Self> {
        tensor.tensor.device().into()
    }

    async fn complex_into_real_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_backend::TensorData, burn_backend::ExecutionError> {
        // Ok(burn_std::complex_utils::interleaved_data_to_real_data(
        //     tensor.into_data(),
        // ))
        todo!()
    }

    async fn complex_into_imag_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_backend::TensorData, burn_backend::ExecutionError> {
        // Ok(burn_std::complex_utils::interleaved_data_to_imag_data(
        //     tensor.into_data(),
        // ))
        todo!()
    }

    async fn complex_into_interleaved_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_backend::TensorData, burn_backend::ExecutionError> {
        let shape = tensor.shape();
        let tensor = Self::complex_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        Ok(match tensor.tensor.kind() {
            // tch::Kind::ComplexHalf => {
            //     let values = Vec::<f16>::try_from(&tensor).unwrap();
            //     TensorData::new(values, shape)
            // }
            tch::Kind::ComplexFloat => {
                let values = Vec::<ComplexScalar<f32>>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            tch::Kind::ComplexDouble => {
                let values = Vec::<ComplexScalar<f64>>::try_from(&tensor).unwrap();
                TensorData::new(values, shape)
            }
            _ => panic!("Not a valid float kind"),
        })
    }

    async fn complex_into_split_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<(burn_backend::TensorData, burn_backend::TensorData), burn_backend::ExecutionError>
    {
        todo!()
    }

    fn complex_squared_norm(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_random(
        shape: burn_backend::Shape,
        distribution: burn_backend::Distribution,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_backend::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::empty(shape, *device, dtype.into());
                tensor
                    .mut_ops(|tensor| tensor.rand_like_out(tensor))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::empty(shape, *device, dtype.into());
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::empty(shape, *device, dtype.into());
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::empty(shape, *device, dtype.into());
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn complex_zeros(
        shape: burn_backend::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_backend::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (dtype.into_kind(), device)))
    }

    fn complex_ones(
        shape: burn_backend::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_backend::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (dtype.into_kind(), device)))
    }

    fn complex_full(
        shape: burn_backend::Shape,
        fill_value: burn_backend::Scalar,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_backend::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::full(
            shape.dims,
            fill_value.elem::<ComplexScalar<f64>>(),
            (dtype.into_kind(), device),
        ))
    }

    fn complex_to_device(
        tensor: burn_backend::ComplexTensor<Self>,
        device: &Device<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        TchOps::to_device(tensor, device)
    }

    async fn complex_into_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_backend::TensorData, burn_backend::ExecutionError> {
        todo!()
    }

    fn complex_reshape(
        tensor: burn_backend::ComplexTensor<Self>,
        shape: burn_backend::Shape,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_transpose(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_add(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sub(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_mul(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_div(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_neg(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_conj(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_real(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_imag(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_abs(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_arg(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_from_parts(
        real: burn_backend::TensorData,
        imag: burn_backend::TensorData,
        device: &burn_backend::tensor::Device<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_polar(
        magnitude: burn_backend::tensor::FloatTensor<Self>,
        phase: burn_backend::tensor::FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_exp(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_log(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sqrt(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sin(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_cos(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_tan(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_acos(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_acosh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_asin(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_asinh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_atan(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_atanh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_select(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice(
        tensor: burn_backend::ComplexTensor<Self>,
        slices: &[burn_backend::Slice],
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice_assign(
        tensor: burn_backend::ComplexTensor<Self>,
        ranges: &[burn_backend::Slice],
        value: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_nd(
        tensor: burn_backend::ComplexTensor<Self>,
        indices: IntTensor<Self>,
        value: burn_backend::ComplexTensor<Self>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_swap_dims(
        tensor: burn_backend::ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_repeat_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        times: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_not_equal(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_cat(
        tensors: alloc::vec::Vec<burn_backend::ComplexTensor<Self>>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_any(
        tensor: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_any_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_all(
        tensor: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_all_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_permute(
        tensor: burn_backend::ComplexTensor<Self>,
        axes: &[usize],
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_expand(
        tensor: burn_backend::ComplexTensor<Self>,
        shape: burn_backend::Shape,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_flip(
        tensor: burn_backend::ComplexTensor<Self>,
        axes: &[usize],
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_unfold(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_select_add(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        values: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal_elem(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_not_equal_elem(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> <Self>::BoolTensorPrimitive {
        todo!()
    }

    fn complex_mask_where(
        tensor: burn_backend::ComplexTensor<Self>,
        mask: <Self>::BoolTensorPrimitive,
        source: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_mask_fill(
        tensor: burn_backend::ComplexTensor<Self>,
        mask: <Self>::BoolTensorPrimitive,
        value: burn_backend::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_gather(
        dim: usize,
        tensor: burn_backend::ComplexTensor<Self>,
        indices: IntTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: burn_backend::ComplexTensor<Self>,
        indices: IntTensor<Self>,
        values: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_sign(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::tensor::FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_matmul(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumsum(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumprod(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        todo!()
    }
}
