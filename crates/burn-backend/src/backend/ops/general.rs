use std::any::TypeId;

use burn_std::DType;

use crate::{BackendTypes, TensorMetadata};




pub trait UntypedTensorOps<B: BackendTypes> {

    fn bitcast<T1: TensorMetadata, T2: TensorMetadata>(tensor: T1, target: DType) -> T2;
    //fn bitwise_xor<T: TensorMetadata>(lhs: T, rhs:B::IntTensorPrimitive) -> T;
    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn swap_dims<T: TensorMetadata+'static>(tensor: T, dim1: usize, dim2: usize) -> T;

}
#[inline]
pub fn safety_check_coercive<T1: TensorMetadata, T2: TensorMetadata>(lhs: &T1, rhs:&T2) -> bool {
    let size1 = lhs.dtype().size();
    let size2 = rhs.dtype().size();

    // Sub-byte types are unsupported
    if size1 == 0 || size2 == 0 {
        return false;
    }
    let elems1 = lhs.shape().num_elements();
    let elems2 = rhs.shape().num_elements();

    // Total byte footprint must be identical
    let bytes1 = elems1.checked_mul(size1);
    let bytes2 = elems2.checked_mul(size2);
    if bytes1 != bytes2 || bytes1.is_none() {
        return false;
    }

    // Alignment safety: the larger alignment must be a multiple of the smaller,
    // and the reinterpreted type must not require stricter alignment than the source.
    // Since we're working with primitive numeric types, alignment == size for all
    // supported dtypes. The destination element must not be more strictly aligned
    // than the source element.
    //
    // e.g. f16 (2 bytes) -> i32 (4 bytes): illegal — i32 needs 4-byte alignment
    //      but adjacent f16 elements are only 2-byte aligned.
    // e.g. i32 (4 bytes) -> f16 (2 bytes): legal — f16 only needs 2-byte alignment.
    // e.g. f32 (4 bytes) -> i32 (4 bytes): legal — same alignment.
    if size2 > size1 {
        return false;
    }

    true
}



#[inline]
pub fn safety_check_exact<T1: TensorMetadata>(lhs: &T1, target: &DType) -> bool {
    let size1 = lhs.dtype().size();
    let size2 = target.size();

    // Sub-byte types are unsupported
    if size1 == 0 || size2 == 0 {
        return false;
    }

    
    size1 == size2
}