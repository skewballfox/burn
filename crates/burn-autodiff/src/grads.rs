use alloc::boxed::Box;
use burn_backend::{Backend, TensorMetadata, TensorPrimitive, tensor::FloatTensor};
use burn_std::tensor::container::TensorContainer;

use crate::{
    NodeId,
    graph::{NodeRef, Requirement},
    tensor::{AutodiffTensor, AutodiffTensorTrait},
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
    pub fn new<T: AutodiffTensorTrait>(root_node: NodeRef, root_tensor: T::Primitive) -> Self {
        Self::new_with_hook::<T>(root_node, root_tensor, None)
    }

    /// Creates a new gradients container.
    fn new_with_hook<T: AutodiffTensorTrait>(
        root_node: NodeRef,
        root_tensor: T::Primitive,
        on_register: Option<OnRegisterHook>,
    ) -> Self {
        let mut gradients = Self {
            container: TensorContainer::new(),
            on_register,
        };
        gradients.register::<T>(root_node.id, T::ones_like(&root_tensor));
        gradients
    }

    /// Creates a new gradients container with a registration hook for distributed gradients.
    #[cfg(feature = "std")]
    pub fn new_distributed<T: AutodiffTensorTrait>(
        root_node: NodeRef,
        root_tensor: T::Primitive,
        mut reg: Box<dyn DistributedRegistration>,
    ) -> Self {
        let on_register: Option<OnRegisterHook> = Some(Box::new(move |id, container| {
            reg.on_register(id, container);
        }));
        Self::new_with_hook::<T>(root_node, root_tensor, on_register)
    }

    /// Consumes the gradients for a given tensor.
    ///
    /// Each tensor should be consumed exactly 1 time if its gradients are only required during the
    /// backward pass, otherwise, it may be consume multiple times.
    pub fn consume<T: AutodiffTensorTrait>(&mut self, node: &NodeRef) -> T::Primitive {
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
    pub fn remove<T: AutodiffTensorTrait>(&mut self, tensor: &T) -> Option<T::Primitive> {
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
    pub fn register<T: AutodiffTensorTrait>(&mut self, node_id: NodeId, value: T::Primitive) {
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
