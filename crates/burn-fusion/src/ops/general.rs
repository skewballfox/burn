use std::marker::PhantomData;

use burn_backend::{StreamId, TensorMetadata, ops::UntypedTensorOps};
use burn_ir::{HandleContainer, SwapDimsOpIr};

use crate::{Fusion, FusionBackend, stream::Operation};

impl<B: FusionBackend> UntypedTensorOps<Self> for Fusion<B> {
    fn bitcast<T1: TensorMetadata, T2: TensorMetadata>(tensor: T1) -> T2 {
        todo!()
    }

    fn swap_dims<T>(tensor: T, dim1: usize, dim2: usize) -> T {
        #[derive(new, Debug)]
        struct SwapDimsOps<B: FusionBackend, T: TensorMetadata+'static> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
            _t: PhantomData<T>,
        }

        impl<B: FusionBackend, T: TensorMetadata+'static> Operation<B::FusionRuntime> for SwapDimsOps<B,T> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.desc.input);
                let output = B::swap_dims::<T>(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }
        let streams = StreamId::current();

        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::SwapDims(desc.clone())),
                SwapDimsOps::<B>::new(desc),
            )
            .output()
    }
}

u