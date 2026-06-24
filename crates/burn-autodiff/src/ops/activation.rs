use core::marker::PhantomData;

use crate::{
    Autodiff,
    checkpoint::{
        base::Checkpointer, retro_forward::RetroForward, state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::NodeId,
    ops::{Backward, Ops, OpsKind, unary},
    retro_unary,
    tensor::{AutodiffTensor},
};
use burn_backend::{Backend, ops::ActivationOps, tensor::FloatTensor};

impl<B: Backend, C: CheckpointStrategy> ActivationOps<Autodiff<B, C>> for Autodiff<B, C> {
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Gelu<B: Backend>(pub(crate) std::marker::PhantomData<B>);

        retro_unary!(RetroGelu, B::gelu);

        impl<B: Backend> Backward<B, 1> for Gelu<B> {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<AutodiffTensor<B>, _>(ops.parents, ops.node, grads, |grad| {
                    B::gelu_backward(input, grad)
                });
            }
        }

        match Gelu::<B>(PhantomData)
            .prepare::<C,AutodiffTensor<B>>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroGelu::<B>::new(tensor.node.id))
            .parents::<B,AutodiffTensor<B>,_>([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::gelu(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::gelu(tensor.primitive)),
        }
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Relu<B: Backend>(pub(crate) std::marker::PhantomData<B>);

        retro_unary!(RetroRelu, B::relu);

        impl<B: Backend> Backward<B, 1> for Relu<B> {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let state = checkpointer.retrieve_node_output(ops.state);
                unary::<AutodiffTensor<B>, _>(ops.parents, ops.node, grads, |grad| {
                    B::relu_backward(state, grad)
                });
            }
        }

        match Relu::<B>(PhantomData)
            .prepare::<C,AutodiffTensor<B>>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroRelu::<B>::new(tensor.node.id))
            .parents::<B,AutodiffTensor<B>,_>([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::relu(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::relu(tensor.primitive)),
        }
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Sigmoid<B: Backend>(pub(crate) std::marker::PhantomData<B>);

        retro_unary!(RetroSigmoid, B::sigmoid);

        impl<B: Backend> Backward<B, 1> for Sigmoid<B> {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);
                let output = B::sigmoid(input);
                unary::<AutodiffTensor<B>, _>(ops.parents, ops.node, grads, |grad| {
                    B::sigmoid_backward(output, grad)
                });
            }
        }

        match Sigmoid::<B>(PhantomData)
            .prepare::<C, AutodiffTensor<B>>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroSigmoid::<B>::new(tensor.node.id))
            .parents::<B, AutodiffTensor<B>, _>([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::sigmoid(tensor.primitive))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::sigmoid(tensor.primitive)),
        }
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LogSigmoid<B: Backend>(pub(crate) std::marker::PhantomData<B>);

        retro_unary!(RetroLogSigmoid, B::log_sigmoid);

        impl<B: Backend> Backward<B, 1> for LogSigmoid<B> {
            type State = NodeId;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let input = checkpointer.retrieve_node_output(ops.state);

                unary::<AutodiffTensor<B>, _>(ops.parents, ops.node, grads, |grad| {
                    B::log_sigmoid_backward(input, grad)
                });
            }
        }

        match LogSigmoid::<B>(PhantomData)
            .prepare::<C,AutodiffTensor<B>>([tensor.node.clone()])
            .memory_bound()
            .retro_forward(RetroLogSigmoid::<B>::new(tensor.node.id))
            .parents::<B, AutodiffTensor<B>, _>([&tensor])
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let state = prep.checkpoint(&tensor);
                prep.finish(state, B::log_sigmoid(tensor.primitive.clone()))
            }
            OpsKind::UnTracked(prep) => prep.finish(B::log_sigmoid(tensor.primitive)),
        }
    }
}
