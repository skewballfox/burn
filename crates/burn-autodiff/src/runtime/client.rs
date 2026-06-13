use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::{BackwardMode, Gradients},
    graph::StepBoxed,
    tensor::{AutodiffTensorTrait, NodeRefCount},
};

/// Client used to communicate with the autodiff server.
pub trait AutodiffClient: Send + Clone {
    /// Register a new step.
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder);
    /// Call backpropagation from the given tensor.
    fn backward<T: AutodiffTensorTrait>(&self, tensor: T, mode: BackwardMode) -> Gradients;
}

/// Client implementation in used.
pub type AutodiffClientImpl = super::graph::GraphMutexClient;
