use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::{BackwardMode, Gradients},
    graph::{ComputingProperty, Node, NodeId, NodeRef, Parent, Requirement, Step},
    runtime::{AutodiffClient, AutodiffClientImpl},
};
#[cfg(feature = "std")]
use crate::{distributed::DistributedGradientRegistration, grads::GradSyncContext};
use alloc::{boxed::Box, vec};
use burn_backend::{
    AutodiffTensor as BackendAutodiffTensor, Backend, BackendTypes, ComplexTensorBackend,
    TensorMetadata, TensorPrimitive, distributed::DistributedOps,
};

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use burn_backend::distributed::{DistributedParamId, DistributedParams};

#[derive(Debug, Clone)]
pub struct AutodiffTensor<B: BackendTypes> {
    pub primitive: B::FloatTensorPrimitive,
    pub node: NodeRef,
    pub rc: NodeRefCount,
}

#[derive(Debug, Clone)]
pub struct ComplexAutodiffTensor<B: BackendTypes> {
    pub primitive: B::ComplexTensorPrimitive,
    pub node: NodeRef,
    pub rc: NodeRefCount,
}

/// Trait implemented by all autodiff tensors, providing the necessary interface for the backward pass and gradient management.
pub trait AutodiffTensorTrait: BackendAutodiffTensor {
    /// wraps the enum used for gets until I can figure out whether it should be used
    /// as the primitive associated type for float autodiff
    type PrimitivePlaceholder: Clone + Send + Sync + 'static;
    /// Maps to the underlying tensor primitive's "ones_like" function, used during the backward pass to create gradient tensors with the same shape and device as the original tensor.
    fn ones_like(tensor: &Self::Primitive) -> Self::Primitive;
    /// Maps to the underlying tensor primitive's "sum_dim" function, used for broadcasting
    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Create a tensor from parent infos.
    fn new_with_node(primitive: Self::Primitive, node: NodeRef) -> Self;
    /// Get a mutable reference to the tensor's node.
    fn node_mut(&mut self) -> &mut NodeRef;
    /// Get a reference to the tensor's node.
    fn node(&self) -> &NodeRef;
    /// Get a reference to the tensor's reference count.
    fn ref_count(&self) -> &NodeRefCount;
    /// Get the primitive value of the tensor.
    fn primitive(&self) -> &Self::Primitive;
    /// Consume the tensor and return its primitive value.
    fn into_primitive(self) -> Self::Primitive;
    /// Break the tensor into its primitive, node, and reference count components.
    fn destructure(self) -> (Self::Primitive, NodeRef, NodeRefCount);
    /// Add two primitives together, used during the backward pass to accumulate gradients.
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
    /// Register a step into a graph for that tensor.
    ///
    /// # Warning
    ///
    /// This should be called only once per tensor.
    fn register_step<S: Step + 'static>(
        self,
        step_that_created_the_tensor: S,
        actions: CheckpointerBuilder,
    ) -> Self {
        self.node().client.register(
            self.ref_count().clone(),
            Box::new(step_that_created_the_tensor),
            actions,
        );
        self
    }
    /// Check if the tensor is tracked, meaning it requires gradients and will be part of the backward pass.
    fn is_tracked(&self) -> bool {
        !self.node().requirement.is_none()
    }
    /// TODO: figure out what to name placeholder
    fn placeholder_primitive(placeholder: Self::PrimitivePlaceholder) -> Self::Primitive;

    /// TODO: figure out what to name placeholder
    fn primitive_to_placeholder(primitive: Self::Primitive) -> Self::PrimitivePlaceholder;
    /// Get the gradients for this tensor from the gradients container, if they exist.
    fn grad(&self, grads: &Gradients) -> Option<Self::Primitive> {
        grads.get::<Self>(self)
    }
    /// Remove the gradients for this tensor from the gradients container and return them, if they exist.
    fn grad_remove(&self, grads: &mut Gradients) -> Option<Self::Primitive> {
        grads.remove_inner::<Self>(self)
    }
    /// Replace the gradients for this tensor in the gradients container with the provided gradients.
    fn grad_replace(&self, grads: &mut Gradients, grad: Self::Primitive) {
        grads.remove_inner::<Self>(self);
        grads.register_typed::<Self>(self.node().id, grad);
    }

    /// Mark the tensor as distributed across multiple devices.
    /// Its gradients will be automatically aggregated from those devices after the backward pass.
    ///
    /// # Arguments
    ///
    /// * `param_id` - The module tensor's [`DistributedParamId`].
    fn grad_distributed(mut self, param_id: DistributedParamId) -> Self {
        let node = self.node_mut();
        *node = Node::new(
            vec![],
            0,
            node.id,
            node.requirement,
            node.properties.clone(),
            node.client.clone(),
            Some(DistributedParams { param_id }),
        )
        .into();
        let step = RootStep::new(node.clone());

        self.register_step(step, CheckpointerBuilder::default())
    }
    /// Mark the tensor as requiring gradients.
    ///
    /// # Panics
    ///
    /// It panics if the tensor is not a leaf.
    fn require_grad(mut self) -> Self {
        let node = self.node_mut();
        match node.requirement {
            Requirement::Grad => self,
            Requirement::GradInBackward => {
                panic!("Can't convert a non leaf tensor into a tracked tensor")
            }
            Requirement::None => {
                *node = Node::new(
                    vec![],
                    0,
                    node.id,
                    Requirement::Grad,
                    node.properties.clone(),
                    node.client.clone(),
                    node.distributed_params.clone(),
                )
                .into();
                let step = RootStep::new(node.clone());

                self.register_step(step, CheckpointerBuilder::default())
            }
        }
    }

    /// Create a tensor from parent infos.
    fn from_parents(
        primitive: Self::Primitive,
        parent_nodes: &[NodeRef],
        requirement: Requirement,
        computing_properties: ComputingProperty,
    ) -> Self {
        let order = parent_nodes
            .iter()
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let client = parent_nodes
            .first()
            .map(|node| node.client.clone())
            .unwrap_or_else(AutodiffClientImpl::new);

        let node: NodeRef = Node::new(
            parent_nodes
                .iter()
                .filter_map(|node| node.clone_if_require_grad())
                .map(|node| Parent::new(node.id))
                .collect(),
            order,
            NodeId::new(),
            requirement,
            computing_properties,
            client,
            None,
        )
        .into();

        Self::new_with_node(primitive, node)
    }
}

impl<B: Backend> BackendAutodiffTensor for AutodiffTensor<B> {
    type Primitive = B::FloatTensorPrimitive;
    type Gradients = Gradients;

    fn backward(self) -> Self::Gradients {
        self.backward()
    }
    fn grad(&self, grads: &Self::Gradients) -> Option<Self::Primitive> {
        <Self as AutodiffTensorTrait>::grad(self, grads)
    }

    fn inner(self) -> Self::Primitive {
        self.primitive
    }

    fn from_inner(tensor: Self::Primitive) -> Self {
        AutodiffTensor::new(tensor)
    }

    fn grad_remove(&self, grads: &mut Self::Gradients) -> Option<Self::Primitive> {
        <Self as AutodiffTensorTrait>::grad_remove(self, grads)
    }

    fn grad_replace(&self, grads: &mut Self::Gradients, grad: Self::Primitive) {
        <Self as AutodiffTensorTrait>::grad_replace(self, grads, grad)
    }
}

impl<B: ComplexTensorBackend> BackendAutodiffTensor for ComplexAutodiffTensor<B> {
    type Primitive = B::ComplexTensorPrimitive;
    type Gradients = Gradients;

    fn backward(self) -> Self::Gradients {
        ComplexAutodiffTensor::<B>::backward(self)
    }
    fn grad(&self, grads: &Self::Gradients) -> Option<Self::Primitive> {
        <Self as AutodiffTensorTrait>::grad(self, grads)
    }

    fn inner(self) -> Self::Primitive {
        self.primitive
    }

    fn from_inner(tensor: Self::Primitive) -> Self {
        ComplexAutodiffTensor::new(tensor)
    }

    fn grad_remove(&self, grads: &mut Self::Gradients) -> Option<Self::Primitive> {
        <Self as AutodiffTensorTrait>::grad_remove(self, grads)
    }

    fn grad_replace(&self, grads: &mut Self::Gradients, grad: Self::Primitive) {
        <Self as AutodiffTensorTrait>::grad_replace(self, grads, grad)
    }
}

impl<B: ComplexTensorBackend> AutodiffTensorTrait for ComplexAutodiffTensor<B> {
    /// wraps the enum used for gets until I can figure out whether it should be used
    /// as the primitive associated type for float autodiff
    type PrimitivePlaceholder = Self::Primitive;
    fn node_mut(&mut self) -> &mut NodeRef {
        &mut self.node
    }

    fn node(&self) -> &NodeRef {
        &self.node
    }

    fn ref_count(&self) -> &NodeRefCount {
        &self.rc
    }

    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::complex_add(lhs, rhs)
    }

    fn ones_like(tensor: &Self::Primitive) -> Self::Primitive {
        B::complex_ones(
            tensor.shape(),
            &B::complex_device(tensor),
            tensor.dtype().into(),
        )
    }
    fn primitive(&self) -> &Self::Primitive {
        &self.primitive
    }
    fn into_primitive(self) -> Self::Primitive {
        self.primitive
    }

    fn placeholder_primitive(placeholder: Self::PrimitivePlaceholder) -> Self::Primitive {
        placeholder
    }

    fn primitive_to_placeholder(primitive: Self::Primitive) -> Self::PrimitivePlaceholder {
        primitive
    }

    fn destructure(self) -> (Self::Primitive, NodeRef, NodeRefCount) {
        (self.primitive, self.node, self.rc)
    }

    fn new_with_node(primitive: Self::Primitive, node: NodeRef) -> Self {
        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }
    
    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::complex_sum_dim(tensor, dim)
    }
}
impl<B: Backend> AutodiffTensorTrait for AutodiffTensor<B> {
    /// wraps the enum used for gets until I can figure out whether it should be used
    /// as the primitive associated type for float autodiff
    type PrimitivePlaceholder = TensorPrimitive<B>;
    fn node_mut(&mut self) -> &mut NodeRef {
        &mut self.node
    }

    fn node(&self) -> &NodeRef {
        &self.node
    }

    fn ref_count(&self) -> &NodeRefCount {
        &self.rc
    }

    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::float_add(lhs, rhs)
    }

    fn ones_like(tensor: &Self::Primitive) -> Self::Primitive {
        B::float_ones(
            tensor.shape(),
            &B::float_device(tensor),
            tensor.dtype().into(),
        )
    }
    fn primitive(&self) -> &Self::Primitive {
        &self.primitive
    }
    fn into_primitive(self) -> Self::Primitive {
        self.primitive
    }

    fn placeholder_primitive(placeholder: Self::PrimitivePlaceholder) -> Self::Primitive {
        placeholder.tensor()
    }

    fn primitive_to_placeholder(primitive: Self::Primitive) -> Self::PrimitivePlaceholder {
        TensorPrimitive::Float(primitive)
    }

    fn destructure(self) -> (Self::Primitive, NodeRef, NodeRefCount) {
        (self.primitive, self.node, self.rc)
    }

    fn new_with_node(primitive: Self::Primitive, node: NodeRef) -> Self {
        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }
    
    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::float_sum_dim(tensor, dim)
    }
}
impl<B: BackendTypes> TensorMetadata for AutodiffTensor<B> {
    fn dtype(&self) -> burn_std::DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> burn_std::Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }
}

impl<B: BackendTypes> TensorMetadata for ComplexAutodiffTensor<B> {
    fn dtype(&self) -> burn_std::DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> burn_std::Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }
}

pub type NodeRefCount = Arc<NodeId>;

#[derive(new, Debug)]
pub(crate) struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeId {
        self.node.id
    }

    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }

    fn depth(&self) -> usize {
        self.node.order
    }

    fn distributed_params(&self) -> Option<DistributedParams> {
        self.node.distributed_params.clone()
    }
}

impl<B: Backend> AutodiffTensor<B> {
    /// Create a new leaf tensor.
    pub fn new(primitive: B::FloatTensorPrimitive) -> Self {
        let id = NodeId::new();
        let node: NodeRef = Node::new(
            vec![],
            0,
            id,
            Requirement::None,
            ComputingProperty::Ambiguous,
            AutodiffClientImpl::new(),
            None,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node: node.clone(),
        }
    }

    #[cfg(not(feature = "std"))]
    pub fn backward(self) -> Gradients {
        let client = self.node.client.clone();

        AutodiffClient::backward::<Self>(&client, self, BackwardMode::default())
    }
    #[cfg(feature = "std")]
    pub fn backward(self) -> Gradients {
        let device = B::float_device(&self.primitive);
        let device_cloned = device.clone();
        let client = self.node.client.clone();

        let mode = BackwardMode::Distributed(Box::new(|ctx: GradSyncContext| {
            let registration = DistributedGradientRegistration::<B>::new(
                ctx.n_required_map,
                ctx.distributed_params,
                device_cloned,
            );
            Box::new(registration)
        }));

        let grads = AutodiffClient::backward::<Self>(&client, self, mode);
        B::submit_sync_collective(&device);
        grads
    }
}

impl<B: BackendTypes> ComplexAutodiffTensor<B> {
    /// Create a new leaf tensor.
    pub fn new(primitive: B::ComplexTensorPrimitive) -> Self {
        let id = NodeId::new();
        let node: NodeRef = Node::new(
            vec![],
            0,
            id,
            Requirement::None,
            ComputingProperty::Ambiguous,
            AutodiffClientImpl::new(),
            None,
        )
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node: node.clone(),
        }
    }
}

impl<B: ComplexTensorBackend> ComplexAutodiffTensor<B> {
    #[cfg(not(feature = "std"))]
    pub fn backward(self) -> Gradients {
        let client = self.node.client.clone();

        AutodiffClient::backward::<Self>(&client, self, BackwardMode::default())
    }
    #[cfg(feature = "std")]
    pub fn backward(self) -> Gradients {
        let device = B::complex_device(&self.primitive);
        let device_cloned = device.clone();
        let client = self.node.client.clone();

        let mode = BackwardMode::Distributed(Box::new(|ctx: GradSyncContext| {
            let registration = DistributedGradientRegistration::<B::InnerBackend>::new(
                ctx.n_required_map,
                ctx.distributed_params,
                device_cloned,
            );
            Box::new(registration)
        }));

        let grads = AutodiffClient::backward::<Self>(&client, self, mode);
        B::InnerBackend::submit_sync_collective(&device);
        grads
    }
}
