use burn_backend::{TensorMetadata, ops::{UntypedTensorOps, safety_check_coercive, safety_check_exact}};

use crate::{Flex, FlexTensor};

impl UntypedTensorOps<Flex> for Flex {
    fn bitcast<T1: burn_backend::TensorMetadata, T2: burn_backend::TensorMetadata>(tensor: T1) -> T2 {
        safety_check_exact(lhs, rhs)
    }

    fn swap_dims<T: TensorMetadata + 'static>(tensor: T, dim1: usize, dim2: usize) -> T {
    
        let result = flex.transpose(dim1, dim2);
        unsafe { core::mem::transmute(result) }
    }
}



