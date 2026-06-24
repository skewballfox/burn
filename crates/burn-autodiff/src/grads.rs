use alloc::boxed::Box;
use burn_backend::{Backend, ComplexTensorBackend};
use burn_std::tensor::container::TensorContainer;

use crate::{
    NodeId,
    graph::{NodeRef, Requirement},
    tensor::{AutodiffTensor, AutodiffTensorTrait, ComplexAutodiffTensor},
};

#[cfg(feature = "std")]
use crate::collections::HashMap;
#[cfg(feature = "std")]
use burn_backend::distributed::DistributedParams;

/// Gradient identifier.
pub type GradID = u64;

#[cfg(feature = "std")]
#[derive(Clone)]
pub(crate) struct GradSyncContext {
    pub n_required_map: HashMap<NodeId, usize>,
    pub distributed_params: HashMap<NodeId, DistributedParams>,
}

/// Hook type executed when a gradient is registered.
type OnRegisterHook = Box<dyn FnMut(&NodeId, &mut TensorContainer<GradID>) + Send + Sync>;

/// Trait for registering distributed gradients.
pub trait DistributedRegistration: Send + Sync {
    /// Performs distributed registration operations on the tensor with the corresponding [`NodeId`].
    fn on_register(&mut self, node_id: &NodeId, container: &mut TensorContainer<GradID>);
}

#[derive(Default)]
pub(crate) enum BackwardMode {
    #[default]
    Standard,
    // Distributed registration hook.
    #[cfg(feature = "std")]
    Distributed(Box<dyn FnOnce(GradSyncContext) -> Box<dyn DistributedRegistration>>),
}

/// Gradients container used during the backward pass.
pub struct Gradients {
    container: TensorContainer<GradID>,
    /// Optional hook called after each gradient is registered, used to trigger
    /// distributed gradient synchronization operations.
    on_register: Option<OnRegisterHook>,
}

impl Gradients {
    /// Creates a new gradients container.
    pub fn new<B: Backend>(root_node: NodeRef, root_tensor: B::FloatTensorPrimitive) -> Self {
        Self::new_with_hook::<AutodiffTensor<B>>(root_node, root_tensor, None)
    }

    /// Creates a new gradients container.
    pub fn new_complex<B: ComplexTensorBackend>(
        root_node: NodeRef,
        root_tensor: B::ComplexTensorPrimitive,
    ) -> Self {
        Self::new_with_hook::<ComplexAutodiffTensor<B>>(root_node, root_tensor, None)
    }

    /// Creates a new gradients container.
    pub(crate) fn new_with_hook<T: AutodiffTensorTrait>(
        root_node: NodeRef,
        root_tensor: T::Primitive,
        on_register: Option<OnRegisterHook>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
            on_register,
        };
        gradients.register_typed::<T>(root_node.id, T::ones_like(&root_tensor));
        gradients
    }

    /// Consumes the gradients for a given float tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<B: Backend>(&mut self, node: &NodeRef) -> B::FloatTensorPrimitive {
        self.consume_typed::<AutodiffTensor<B>>(node)
    }
    /// Consumes the gradients for a given complex tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume_complex<B: ComplexTensorBackend>(
        &mut self,
        node: &NodeRef,
    ) -> B::ComplexTensorPrimitive {
        self.consume_typed::<ComplexAutodiffTensor<B>>(node)
    }
    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub(crate) fn consume_typed<T: AutodiffTensorTrait>(&mut self, node: &NodeRef) -> T::Primitive {
        match node.requirement {
            Requirement::Grad => self
                .container
                .get::<T::PrimitivePlaceholder>(&node.id.value)
                .map(|tensor| T::placeholder_primitive(tensor))
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::GradInBackward => self
                .container
                .remove::<T::PrimitivePlaceholder>(&node.id.value)
                .map(|tensor| T::placeholder_primitive(tensor))
                .expect("Can't consume the gradients before they are registered at least once."),
            Requirement::None => panic!("Trying to consume the gradients for an untracked tensor"),
        }
    }
    /// Removes a grad tensor from the container.
    pub fn remove<B: Backend>(
        &mut self,
        tensor: &AutodiffTensor<B>,
    ) -> Option<B::FloatTensorPrimitive> {
        self.remove_inner::<AutodiffTensor<B>>(tensor)
    }
    /// Removes a grad tensor from the container.
    pub fn remove_complex<B: ComplexTensorBackend>(
        &mut self,
        tensor: &ComplexAutodiffTensor<B>,
    ) -> Option<B::ComplexTensorPrimitive> {
        self.remove_inner::<ComplexAutodiffTensor<B>>(tensor)
    }
    /// Removes a grad tensor from the container.
    pub(crate) fn remove_inner<T: AutodiffTensorTrait>(
        &mut self,
        tensor: &T,
    ) -> Option<T::Primitive> {
        self.container
            .remove::<T::PrimitivePlaceholder>(&tensor.node().id.value)
            .map(|tensor| T::placeholder_primitive(tensor))
    }

    /// Gets a grad tensor from the container.
    pub fn get<T: AutodiffTensorTrait>(&self, tensor: &T) -> Option<T::Primitive> {
        self.container
            .get::<T::PrimitivePlaceholder>(&tensor.node().id.value)
            .map(|tensor| T::placeholder_primitive(tensor))
    }

    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is distributed, launches a syncing operation on the gradients.
    pub fn register<B: Backend>(&mut self, node_id: NodeId, value: B::FloatTensorPrimitive) {
        self.register_typed::<AutodiffTensor<B>>(node_id, value)
    }
    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is distributed, launches a syncing operation on the gradients.
    pub fn register_complex<B: ComplexTensorBackend>(
        &mut self,
        node_id: NodeId,
        value: B::ComplexTensorPrimitive,
    ) {
        self.register_typed::<ComplexAutodiffTensor<B>>(node_id, value)
    }
    /// Register a grad tensor in the container.
    ///
    /// If the tensor already exists, add both tensors together before saving the result.
    ///
    /// If the registered tensor is distributed, launches a syncing operation on the gradients.
    pub(crate) fn register_typed<T: AutodiffTensorTrait>(
        &mut self,
        node_id: NodeId,
        value: T::Primitive,
    ) {
        let out = if let Some(tensor_old) = self
            .container
            .remove::<T::PrimitivePlaceholder>(&node_id.value)
        {
            T::add(value, T::placeholder_primitive(tensor_old))
        } else {
            value
        };

        self.container
            .register::<T::PrimitivePlaceholder>(node_id.value, T::primitive_to_placeholder(out));

        if let Some(hook) = &mut self.on_register {
            hook(&node_id, &mut self.container);
        }
    }
}

#[cfg(feature = "std")]
impl Gradients {
    /// Creates a new gradients container with a registration hook for distributed gradients.
    pub fn new_distributed<B: Backend>(
        root_node: NodeRef,
        root_tensor: B::FloatTensorPrimitive,
        reg: Box<dyn DistributedRegistration>,
    ) -> Self {
        Self::new_distributed_typed::<AutodiffTensor<B>>(root_node, root_tensor, reg)
    }
    /// Creates a new gradients container with a registration hook for distributed gradients.
    pub fn new_distributed_complex<B: ComplexTensorBackend>(
        root_node: NodeRef,
        root_tensor: B::ComplexTensorPrimitive,
        reg: Box<dyn DistributedRegistration>,
    ) -> Self {
        Self::new_distributed_typed::<ComplexAutodiffTensor<B>>(root_node, root_tensor, reg)
    }

    /// Creates a new gradients container with a registration hook for distributed gradients.
    pub(crate) fn new_distributed_typed<T: AutodiffTensorTrait>(
        root_node: NodeRef,
        root_tensor: T::Primitive,
        mut reg: Box<dyn DistributedRegistration>,
    ) -> Self {
        let on_register: Option<OnRegisterHook> = Some(Box::new(move |id, container| {
            reg.on_register(id, container);
        }));
        Self::new_with_hook::<T>(root_node, root_tensor, on_register)
    }
}
