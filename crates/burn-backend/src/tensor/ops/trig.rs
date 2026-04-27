use crate::{Backend, tensor::Numeric};

/// Trait that lists all trigonometric operations that can be applied on floating-point tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct.
pub trait Trigonometric<B: Backend>: Numeric<B> {
    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cos"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cos`")]
    /// function, which is more high-level and designed for public use.
    fn cos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sin`")]
    /// function, which is more high-level and designed for public use.
    fn sin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("tan"))]
    #[cfg_attr(not(doc), doc = "`Tensor::tan`")]
    /// function, which is more high-level and designed for public use.
    fn tan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cosh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cosh`")]
    /// function, which is more high-level and designed for public use.
    fn cosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sinh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sinh`")]
    /// function, which is more high-level and designed for public use.
    fn sinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("tanh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::tanh`")]
    /// function, which is more high-level and designed for public use.
    fn tanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("acos"))]
    #[cfg_attr(not(doc), doc = "`Tensor::acos`")]
    /// function, which is more high-level and designed for public use.
    fn acos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic cosine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("acosh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::acosh`")]
    /// function, which is more high-level and designed for public use.
    fn acosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("asin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::asin`")]
    /// function, which is more high-level and designed for public use.
    fn asin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic sine of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("asinh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::asinh`")]
    /// function, which is more high-level and designed for public use.
    fn asinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atan"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atan`")]
    /// function, which is more high-level and designed for public use.
    fn atan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic tangent of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atanh"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atanh`")]
    /// function, which is more high-level and designed for public use.
    fn atanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a tensor with the four-quadrant inverse tangent values of `y` and `x`.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The tensor with y coordinates.
    /// * `rhs` - The tensor with x coordinates.
    ///
    /// # Returns
    ///
    /// A tensor with the four-quadrant inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the four-quadrant inverse tangent of two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("atan2"))]
    #[cfg_attr(not(doc), doc = "`Tensor::atan2`")]
    /// function, which is more high-level and designed for public use.
    fn atan2(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
}
