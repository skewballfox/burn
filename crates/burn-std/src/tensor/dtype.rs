//! Tensor data type.

use serde::{Deserialize, Serialize};

use crate::tensor::quantization::{QuantScheme, QuantStore, QuantValue};
use crate::{bf16, dtype, f16};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompoundLayout {
    InterLeaved,
    Split,
}

/// Describes a compound data type, which is made up of multiple primitive data types.
/// The data type may be Contiguous (Interleaved) or non-contiguous (Split).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CompoundDtypeScheme {
    pub name: &'static str,
    pub inner_dtypes: &'static [PrimitiveDType],
    pub layout: CompoundLayout,
}

impl CompoundDtypeScheme {
    /// Creates a new compound data type. 
    /// Will panic if: 
    /// - inner_dtypes is empty
    /// - layout is Interleaved and the inner dtypes all the same primitive dtype  
    pub const fn new(
        name: &'static str,
        inner_dtypes: &'static [PrimitiveDType],
        layout: CompoundLayout,
    ) -> Self {
        
        Self {
            name,
            inner_dtypes,
            layout,
        }
    }

    /// Returns the data type name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    
    /// The size of the compound dtype in bytes. Comprized of the sum of the sizes of its inner dtypes.
    /// will PANIC if any of the inner dtypes are sub-byte types.
    pub const fn size(&self) -> usize {
        let mut total_size = 0;
        let mut dtype = 0;
        let length = self.inner_dtypes.len();
        loop {
            if self.inner_dtypes[dtype].size() == 0 {
                panic!("CompoundDtypeScheme contains sub-byte dtype, size is undefined.");
            }
            total_size += self.inner_dtypes[dtype].size();
            dtype += 1;
            if dtype >= length {
                return total_size;
            } 
        }
    }

}


pub enum DType {
    Primitive(PrimitiveDType),
    Compound(CompoundDtypeScheme),
}

impl DType {
    
    /// Returns the layout of the data type if it is compound.
    pub fn layout(&self) -> Option<CompoundLayout> {
        match self {
            DType::Primitive(_) => None,
            DType::Compound(compound_dtype_scheme) => Some(compound_dtype_scheme.layout),
        }
    }

    /// Returns the size of a type in bytes.
    pub fn size(&self) -> usize {
        match self {
            DType::Primitive(p) => p.size(),
            DType::Compound(compound_dtype_scheme) => compound_dtype_scheme.size(),
        }
    }
    /// Returns true if the data type is a floating point type.
    pub fn is_float(&self) -> bool {
        match self {
            DType::Primitive(p) => p.is_float(),
            DType::Compound(_) => false,
        }
    }
    
    /// Returns true if the data type is a signed integer type.
    pub fn is_int(&self) -> bool {
        match self {
            DType::Primitive(primitive_dtype) => primitive_dtype.is_int(),
            DType::Compound(_) => false,
        }
    }
    /// Returns true if the data type is an unsigned integer type.
    pub fn is_uint(&self) -> bool {
        match self {
            DType::Primitive(primitive_dtype) => primitive_dtype.is_uint(),
            DType::Compound(_) => false,
        }
    }

    /// Returns true if the data type is a boolean type
    pub fn is_bool(&self) -> bool {
        match self {
            DType::Primitive(primitive_dtype) => primitive_dtype.is_bool(),
            DType::Compound(_) => false,
        }
    }

    /// Returns the data type name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Primitive(primitive_dtype) => primitive_dtype.name(),
            DType::Compound(compound_dtype_scheme) => compound_dtype_scheme.name(),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitiveDType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    QFloat(QuantScheme),
}

#[cfg(feature = "cubecl")]
impl From<cubecl::ir::ElemType> for PrimitiveDType {
    fn from(value: cubecl::ir::ElemType) -> Self {
        match value {
            cubecl::ir::ElemType::Float(float_kind) => match float_kind {
                cubecl::ir::FloatKind::F16 =>PrimitiveDType::F16,
                cubecl::ir::FloatKind::BF16 =>PrimitiveDType::BF16,
                cubecl::ir::FloatKind::Flex32 =>PrimitiveDType::Flex32,
                cubecl::ir::FloatKind::F32 =>PrimitiveDType::F32,
                cubecl::ir::FloatKind::F64 =>PrimitiveDType::F64,
                cubecl::ir::FloatKind::TF32 => panic!("Not a validPrimitiveDType for tensors."),
                cubecl::ir::FloatKind::E2M1
                | cubecl::ir::FloatKind::E2M3
                | cubecl::ir::FloatKind::E3M2
                | cubecl::ir::FloatKind::E4M3
                | cubecl::ir::FloatKind::E5M2
                | cubecl::ir::FloatKind::UE8M0 => {
                    unimplemented!("Not yet supported, will be used for quantization")
                }
            },
            cubecl::ir::ElemType::Int(int_kind) => match int_kind {
                cubecl::ir::IntKind::I8 =>PrimitiveDType::I8,
                cubecl::ir::IntKind::I16 =>PrimitiveDType::I16,
                cubecl::ir::IntKind::I32 =>PrimitiveDType::I32,
                cubecl::ir::IntKind::I64 =>PrimitiveDType::I64,
            },
            cubecl::ir::ElemType::UInt(uint_kind) => match uint_kind {
                cubecl::ir::UIntKind::U8 =>PrimitiveDType::U8,
                cubecl::ir::UIntKind::U16 =>PrimitiveDType::U16,
                cubecl::ir::UIntKind::U32 =>PrimitiveDType::U32,
                cubecl::ir::UIntKind::U64 =>PrimitiveDType::U64,
            },
            _ => panic!("Not a validPrimitiveDType for tensors."),
        }
    }
}

impl PrimitiveDType {
    /// Returns the size of a type in bytes.
    pub const fn size(&self) -> usize {
        match self {
           PrimitiveDType::F64 => core::mem::size_of::<f64>(),
           PrimitiveDType::F32 => core::mem::size_of::<f32>(),
           PrimitiveDType::Flex32 => core::mem::size_of::<f32>(),
           PrimitiveDType::F16 => core::mem::size_of::<f16>(),
           PrimitiveDType::BF16 => core::mem::size_of::<bf16>(),
           PrimitiveDType::I64 => core::mem::size_of::<i64>(),
           PrimitiveDType::I32 => core::mem::size_of::<i32>(),
           PrimitiveDType::I16 => core::mem::size_of::<i16>(),
           PrimitiveDType::I8 => core::mem::size_of::<i8>(),
           PrimitiveDType::U64 => core::mem::size_of::<u64>(),
           PrimitiveDType::U32 => core::mem::size_of::<u32>(),
           PrimitiveDType::U16 => core::mem::size_of::<u16>(),
           PrimitiveDType::U8 => core::mem::size_of::<u8>(),
           PrimitiveDType::Bool => core::mem::size_of::<bool>(),
           PrimitiveDType::QFloat(scheme) => match scheme.store {
                QuantStore::Native => match scheme.value {
                    QuantValue::Q8F | QuantValue::Q8S => core::mem::size_of::<i8>(),
                    // e2m1 native is automatically packed by the kernels, so the actual storage is
                    // 8 bits wide.
                    QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                        core::mem::size_of::<u8>()
                    }
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                        // Sub-byte values have fractional size
                        0
                    }
                },
                QuantStore::U32 => core::mem::size_of::<u32>(),
            },
        }
    }
    /// Returns true if the data type is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
           PrimitiveDType::F64 |PrimitiveDType::F32 |PrimitiveDType::Flex32 |PrimitiveDType::F16 |PrimitiveDType::BF16
        )
    }
    /// Returns true if the data type is a signed integer type.
    pub fn is_int(&self) -> bool {
        matches!(self,PrimitiveDType::I64 |PrimitiveDType::I32 |PrimitiveDType::I16 |PrimitiveDType::I8)
    }
    /// Returns true if the data type is an unsigned integer type.
    pub fn is_uint(&self) -> bool {
        matches!(self,PrimitiveDType::U64 |PrimitiveDType::U32 |PrimitiveDType::U16 |PrimitiveDType::U8)
    }

    /// Returns true if the data type is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self,PrimitiveDType::Bool)
    }

    /// Returns the data type name.
    pub fn name(&self) -> &'static str {
        match self {
           PrimitiveDType::F64 => "f64",
           PrimitiveDType::F32 => "f32",
           PrimitiveDType::Flex32 => "flex32",
           PrimitiveDType::F16 => "f16",
           PrimitiveDType::BF16 => "bf16",
           PrimitiveDType::I64 => "i64",
           PrimitiveDType::I32 => "i32",
           PrimitiveDType::I16 => "i16",
           PrimitiveDType::I8 => "i8",
           PrimitiveDType::U64 => "u64",
           PrimitiveDType::U32 => "u32",
           PrimitiveDType::U16 => "u16",
           PrimitiveDType::U8 => "u8",
           PrimitiveDType::Bool => "bool",
           PrimitiveDType::QFloat(_) => "qfloat",
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatDType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
}

impl From<PrimitiveDType> for FloatDType {
    fn from(value:PrimitiveDType) -> Self {
        match value {
           PrimitiveDType::F64 => FloatDType::F64,
           PrimitiveDType::F32 => FloatDType::F32,
           PrimitiveDType::Flex32 => FloatDType::Flex32,
           PrimitiveDType::F16 => FloatDType::F16,
           PrimitiveDType::BF16 => FloatDType::BF16,
            _ => panic!("Expected float data type, got {value:?}"),
        }
    }
}

impl From<FloatDType> for PrimitiveDType {
    fn from(value: FloatDType) -> Self {
        match value {
            FloatDType::F64 =>PrimitiveDType::F64,
            FloatDType::F32 =>PrimitiveDType::F32,
            FloatDType::Flex32 =>PrimitiveDType::Flex32,
            FloatDType::F16 =>PrimitiveDType::F16,
            FloatDType::BF16 =>PrimitiveDType::BF16,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntDType {
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl From<PrimitiveDType> for IntDType {
    fn from(value:PrimitiveDType) -> Self {
        match value {
           PrimitiveDType::I64 => IntDType::I64,
           PrimitiveDType::I32 => IntDType::I32,
           PrimitiveDType::I16 => IntDType::I16,
           PrimitiveDType::I8 => IntDType::I8,
           PrimitiveDType::U64 => IntDType::U64,
           PrimitiveDType::U32 => IntDType::U32,
           PrimitiveDType::U16 => IntDType::U16,
           PrimitiveDType::U8 => IntDType::U8,
            _ => panic!("Expected int data type, got {value:?}"),
        }
    }
}

impl From<IntDType> for PrimitiveDType {
    fn from(value: IntDType) -> Self {
        match value {
            IntDType::I64 =>PrimitiveDType::I64,
            IntDType::I32 =>PrimitiveDType::I32,
            IntDType::I16 =>PrimitiveDType::I16,
            IntDType::I8 =>PrimitiveDType::I8,
            IntDType::U64 =>PrimitiveDType::U64,
            IntDType::U32 =>PrimitiveDType::U32,
            IntDType::U16 =>PrimitiveDType::U16,
            IntDType::U8 =>PrimitiveDType::U8,
        }
    }
}
