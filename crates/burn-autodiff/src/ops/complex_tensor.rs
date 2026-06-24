use std::marker::PhantomData;

use burn_backend::{
    Backend, ComplexTensorBackend, TensorMetadata, ops::ComplexTensorOps, tensor::{BoolTensor, ComplexTensor, Device, IntTensor}
};

use crate::{Autodiff, NodeId, checkpoint::{base::Checkpointer, retro_forward::RetroForward, state::BackwardStates, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind, unary}, tensor::ComplexAutodiffTensor};

impl<B: ComplexTensorBackend, C: CheckpointStrategy> ComplexTensorBackend
    for Autodiff<B, C>
{
    type InnerBackend = B::InnerBackend;

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

impl<B: ComplexTensorBackend, C: CheckpointStrategy> ComplexTensorOps<Self>
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
        _tensor: ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_std::Distribution,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_random(shape, distribution, device, dtype))
    }

    fn complex_zeros(
        shape: burn_std::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_zeros(shape, device, dtype))
    }

    fn complex_ones(
        shape: burn_std::Shape,
        device: &burn_backend::tensor::Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        ComplexAutodiffTensor::new(B::complex_ones(shape, device, dtype))
    }

    fn complex_full(
        _shape: burn_std::Shape,
        _fill_value: burn_std::Scalar,
        _device: &burn_backend::tensor::Device<Self>,
        _dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_to_device(
        _tensor: ComplexTensor<Self>,
        _device: &Device<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(
        level="trace",
        skip(tensor),
        fields(
            from = ?tensor.node,
            shape = ?tensor.shape(),
            dtype = ?tensor.dtype(),
        )
    ))]
    async fn complex_into_data(
        tensor: ComplexTensor<Self>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        B::complex_into_data(tensor.primitive).await
    }

    fn complex_reshape(
        _tensor: ComplexTensor<Self>,
        _shape: burn_std::Shape,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_transpose(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_add(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sub(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mul(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_div(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_neg(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_conj(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_recip(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_finv(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_real(_tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_imag(_tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_into_float(
        _tensor: ComplexTensor<Self>,
        _dtype: burn_std::FloatDType,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_into_int(
        _tensor: ComplexTensor<Self>,
        _dtype: burn_std::IntDType,
    ) -> burn_backend::tensor::IntTensor<Self> {
        todo!()
    }

    fn complex_abs(_tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_arg(_tensor: ComplexTensor<Self>) -> burn_backend::tensor::FloatTensor<Self> {
        todo!()
    }

    fn complex_from_parts(
        _real: burn_std::TensorData,
        _imag: burn_std::TensorData,
        _device: &burn_backend::tensor::Device<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_from_polar(
        _magnitude: burn_backend::tensor::FloatTensor<Self>,
        _phase: burn_backend::tensor::FloatTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_exp(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_log(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sqrt(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sin(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cos(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_tan(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_acos(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_acosh(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cast(
        _tensor: ComplexTensor<Self>,
        _dtype: burn_std::ComplexDType,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_asin(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_asinh(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atan(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atanh(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_atan2(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_select(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice(
        _tensor: ComplexTensor<Self>,
        _slices: &[burn_std::Slice],
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_slice_assign(
        _tensor: ComplexTensor<Self>,
        _ranges: &[burn_std::Slice],
        _value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_nd(
        _tensor: ComplexTensor<Self>,
        _indices: IntTensor<Self>,
        _value: ComplexTensor<Self>,
        _reduction: burn_std::IndexingUpdateOp,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<Self> {
        #[derive(Debug)]
        struct SwapDim<B: ComplexTensorBackend>(pub(crate) std::marker::PhantomData<B>);

        #[derive(new, Debug)]
        struct RetroSwapDims<B: ComplexTensorBackend> {
            input_id: NodeId,
            dim1: usize,
            dim2: usize,
            _backend: PhantomData<B>,
        }

        impl<B: ComplexTensorBackend + std::marker::Send> RetroForward for RetroSwapDims<B> {
            fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
                let input = states.get_state::<B::ComplexTensorPrimitive>(&self.input_id);
                let out = B::complex_swap_dims(input, self.dim1, self.dim2);
                states.save(out_node, out)
            }
        }

        impl<B: ComplexTensorBackend> Backward<B, 1> for SwapDim<B> {
            type State = (usize, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (dim1, dim2) = ops.state;

                unary::<ComplexAutodiffTensor<B>, _>(ops.parents, ops.node, grads, |grad| {
                    B::complex_swap_dims(grad, dim2, dim1)
                });
            }
        }

        match SwapDim::<B>(PhantomData)
            .prepare::<C, ComplexAutodiffTensor<B>>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSwapDims::<B>::new(tensor.node.id, dim1, dim2))
            .parents::<B,ComplexAutodiffTensor<B>,_>([&tensor])
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (dim1, dim2),
                B::complex_swap_dims(tensor.primitive, dim1, dim2),
            ),
            OpsKind::UnTracked(prep) => {
                prep.finish(B::complex_swap_dims(tensor.primitive, dim1, dim2))
            }
        }
    }

    fn complex_repeat_dim(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _times: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal(
        _lhs: ComplexTensor<Self>,
        _rhs: ComplexTensor<Self>,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_not_equal(
        _lhs: ComplexTensor<Self>,
        _rhs: ComplexTensor<Self>,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_cat(
        _tensors: alloc::vec::Vec<ComplexTensor<Self>>,
        _dim: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_any(
        _tensor: ComplexTensor<Self>,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_any_dim(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_all(
        _tensor: ComplexTensor<Self>,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_all_dim(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_permute(_tensor: ComplexTensor<Self>, _axes: &[usize]) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_expand(
        _tensor: ComplexTensor<Self>,
        _shape: burn_std::Shape,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_flip(_tensor: ComplexTensor<Self>, _axes: &[usize]) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_unfold(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_select_add(
        _tensor: ComplexTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
        _values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sum_dim(_tensor: ComplexTensor<Self>, _dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_prod_dim(_tensor: ComplexTensor<Self>, _dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mean_dim(_tensor: ComplexTensor<Self>, _dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder(
        _lhs: ComplexTensor<Self>,
        _rhs: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_remainder_scalar(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_equal_elem(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_std::Scalar,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_not_equal_elem(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_std::Scalar,
        _out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Self> {
        todo!()
    }

    fn complex_mask_where(
        _tensor: ComplexTensor<Self>,
        _mask: BoolTensor<Self>,
        _source: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_mask_fill(
        _tensor: ComplexTensor<Self>,
        _mask: BoolTensor<Self>,
        _value: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_gather(
        _dim: usize,
        _tensor: ComplexTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_scatter_add(
        _dim: usize,
        _tensor: ComplexTensor<Self>,
        _indices: IntTensor<Self>,
        _values: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_sign(_tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powc_scalar(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_backend::tensor::FloatTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powi(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_backend::tensor::IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_powf_scalar(
        _lhs: ComplexTensor<Self>,
        _rhs: burn_std::Scalar,
    ) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_matmul(_lhs: ComplexTensor<Self>, _rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumsum(_tensor: ComplexTensor<Self>, _dim: usize) -> ComplexTensor<Self> {
        todo!()
    }

    fn complex_cumprod(_tensor: ComplexTensor<Self>, _dim: usize) -> ComplexTensor<Self> {
        todo!()
    }
}
