use std::{any::TypeId, marker::PhantomData};

use burn_backend::{Backend, BackendTypes, ops::UntypedTensorOps};

use crate::{
    Autodiff, NodeId,
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
    tensor::AutodiffTensor,
};

impl<B: Backend, C: CheckpointStrategy> UntypedTensorOps<Self> for Autodiff<B, C> {
    fn bitcast<T1: burn_backend::TensorMetadata, T2: burn_backend::TensorMetadata>(
        tensor: T1,
    ) -> T2 {
        // This is more of an intermediary no-op for bitwise operations or operations that require temporary erasure
        // at the backend level, so this should never be tracked.
        B::bitcast::<T1,T2>(tensor)
    }

    fn bitwise_xor<T: burn_backend::TensorMetadata>(
        lhs: T,
        rhs: <Self as burn_backend::BackendTypes>::IntTensorPrimitive,
    ) -> T {
        //TODO: work out what this would even look like
        B::bitwise_xor::<T>(tensor)
        
    }

    fn swap_dims<T: burn_backend::TensorMetadata + 'static>(
        tensor: T,
        dim1: usize,
        dim2: usize,
    ) -> T {
        if should_track::<Self, T>() {
            as_autodiff::<B, C, _, _>(tensor, |tensor: AutodiffTensor<B>| {
                #[derive(Debug)]
                struct SwapDim;

                #[derive(new, Debug)]
                struct RetroSwapDims<B: Backend> {
                    input_id: NodeId,
                    dim1: usize,
                    dim2: usize,
                    _backend: PhantomData<B>,
                }

                impl<B: Backend> RetroForward for RetroSwapDims<B> {
                    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
                        let input = states.get_state::<B::FloatTensorPrimitive>(&self.input_id);
                        let out =
                            B::swap_dims::<B::FloatTensorPrimitive>(input, self.dim1, self.dim2);
                        states.save(out_node, out)
                    }
                }

                impl<B: Backend> Backward<B, 1> for SwapDim {
                    type State = (usize, usize);

                    fn backward(
                        self,
                        ops: Ops<Self::State, 1>,
                        grads: &mut Gradients,
                        _checkpointer: &mut Checkpointer,
                    ) {
                        let (dim1, dim2) = ops.state;

                        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                            B::swap_dims::<B::FloatTensorPrimitive>(grad, dim2, dim1)
                        });
                    }
                }

                match SwapDim
                    .prepare::<C>([tensor.node.clone()])
                    .memory_bound()
                    .retro_forward(RetroSwapDims::<B>::new(tensor.node.id, dim1, dim2))
                    .parents([&tensor])
                    .stateful()
                {
                    OpsKind::Tracked(prep) => prep.finish(
                        (dim1, dim2),
                        B::swap_dims::<B::FloatTensorPrimitive>(tensor.primitive, dim1, dim2),
                    ),
                    OpsKind::UnTracked(prep) => prep.finish(
                        B::swap_dims::<B::FloatTensorPrimitive>(tensor.primitive, dim1, dim2),
                    ),
                }
            })
        } else {
            B::swap_dims::<T>(tensor, dim1, dim2)
        }
    }
}

fn should_track<B: BackendTypes, T: burn_backend::TensorMetadata + 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<B::FloatTensorPrimitive>()
}

fn as_autodiff<B, C, T, F>(value: T, f: F) -> T
where
    B: Backend,
    C: CheckpointStrategy,
    F: FnOnce(AutodiffTensor<B>) -> AutodiffTensor<B>,
{
    let autodiff: AutodiffTensor<B> = transmute_same_type(value);
    let result = f(autodiff);
    transmute_same_type(result)
}

#[inline]
fn transmute_same_type<A, B>(value: A) -> B {
    // Assert sizes match at compile time to catch mistakes early
    debug_assert_eq!(core::mem::size_of::<A>(), core::mem::size_of::<B>());

    let ptr = &value as *const A as *const B;
    let result = unsafe { ptr.read() };
    core::mem::forget(value);
    result
}
