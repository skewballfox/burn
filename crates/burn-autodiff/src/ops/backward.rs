use super::{Ops, OpsPrep};
use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::{ComputingProperty, NodeRef, Requirement},
    tensor::AutodiffTensorTrait,
    utils::duplicate,
};
use burn_backend::BackendTypes;

/// Trait for all operations.
///
/// # Notes
///
/// Concrete types implementing this trait should not have any state.
/// If a state is necessary during the backward pass,
/// they should be declared with the associated type 'State'.
pub trait Backward<B, const N: usize>: Send + core::fmt::Debug
where
    Self: Sized + 'static,
{
    /// Associated type to compute the backward pass.
    type State: Clone + Send + core::fmt::Debug + 'static;

    /// The backward pass.
    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    );

    /// Prepare the backward ops.
    fn prepare<C: CheckpointStrategy, T: AutodiffTensorTrait>(
        self,
        nodes: [NodeRef; N],
    ) -> OpsPrep<Self, B, Self::State, C, N> {
        let requirement = Requirement::from_nodes(&nodes);
        OpsPrep::new(
            nodes,
            requirement,
            self,
            ComputingProperty::Ambiguous, // If not specified we start with ambiguous
            CheckpointerBuilder::default(),
        )
    }
}

/// Execute a binary operation during the backward step.
pub fn binary<T: AutodiffTensorTrait, FLhs, FRhs>(
    parents: [Option<NodeRef>; 2],
    node: NodeRef,
    grads: &mut Gradients,
    func_lhs: FLhs,
    func_rhs: FRhs,
) where
    FLhs: FnOnce(T::Primitive) -> T::Primitive,
    FRhs: FnOnce(T::Primitive) -> T::Primitive,
{
    let [grad_4lhs, grad_4rhs] = duplicate(&parents, Some(grads.consume_typed::<T>(&node)));
    let [node_lhs, node_rhs] = parents;

    if let Some(node) = node_lhs {
        let grad = func_lhs(grad_4lhs.unwrap());
        grads.register_typed::<T>(node.id, grad)
    }

    if let Some(node) = node_rhs {
        let grad = func_rhs(grad_4rhs.unwrap());
        grads.register_typed::<T>(node.id, grad)
    }
}

/// Execute a unary operation during the backward step.
pub fn unary<T, F>(parents: [Option<NodeRef>; 1], node: NodeRef, grads: &mut Gradients, func: F)
where
    T: AutodiffTensorTrait,
    F: FnOnce(T::Primitive) -> T::Primitive,
{
    let [parent_node] = parents;
    let grad = grads.consume_typed::<T>(&node);

    if let Some(node) = parent_node {
        let grad = func(grad);
        grads.register_typed::<T>(node.id, grad)
    }
}
