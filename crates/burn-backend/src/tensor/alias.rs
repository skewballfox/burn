use crate::backend::BackendCore;

// We provide some type aliases to improve the readability of using associated types without
// having to use the disambiguation syntax.

/// Device type used by the backend.
pub type Device<B> = <B as BackendCore>::Device;

/// Float element type used by backend.
pub type FloatElem<B> = <B as BackendCore>::FloatElem;
/// Integer element type used by backend.
pub type IntElem<B> = <B as BackendCore>::IntElem;
/// Boolean element type used by backend.
pub type BoolElem<B> = <B as BackendCore>::BoolElem;

/// Float tensor primitive type used by the backend.
pub type FloatTensor<B> = <B as BackendCore>::FloatTensorPrimitive;
/// Integer tensor primitive type used by the backend.
pub type IntTensor<B> = <B as BackendCore>::IntTensorPrimitive;
/// Boolean tensor primitive type used by the backend.
pub type BoolTensor<B> = <B as BackendCore>::BoolTensorPrimitive;
/// Quantized tensor primitive type used by the backend.
pub type QuantizedTensor<B> = <B as BackendCore>::QuantizedTensorPrimitive;
