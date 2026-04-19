use burn_backend::TensorData;
use burn_complex::{
    base::{
        ComplexDevice, ComplexTensor, ComplexTensorBackend, ComplexTensorOps, InterleavedLayout,
        element::Complex,
    },
    utils::{
        interleave_from_split_data, interleaved_data_from_imag_data,
        interleaved_data_from_real_data, interleaved_data_to_real_data,
        interleaved_data_to_split_data,
    },
};
use burn_std::{BoolDType, DType};

use crate::ops::comparison::compare_typed;
use crate::ops::comparison::{CompareOp, compare_elem_typed};

use crate::{Flex, FlexDevice, FlexTensor, ops::binary::scalar_op_typed, simd::CmpOp};

impl ComplexTensorBackend for Flex {
    type InnerBackend = Flex;

    type ComplexScalar = Complex<f32>;

    type Layout = InterleavedLayout<FlexTensor>;

    fn complex_from_real_data(
        data: TensorData,
        _device: &ComplexDevice<Self>,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_real_data(data);

        FlexTensor::from_data(interleaved_data).into()
    }

    fn complex_from_imag_data(
        data: TensorData,
        _device: &ComplexDevice<Self>,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_imag_data(data);

        FlexTensor::from_data(interleaved_data).into()
    }

    fn complex_from_interleaved_data(
        data: TensorData,
        _device: &<Self::InnerBackend as burn_backend::Backend>::Device,
    ) -> ComplexTensor<Self> {
        FlexTensor::from_data(data).into()
    }

    fn complex_from_split_data(
        real_data: TensorData,
        imag_data: TensorData,
        _device: &<Self::InnerBackend as burn_backend::Backend>::Device,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleave_from_split_data(real_data, imag_data);
        FlexTensor::from_data(interleaved_data).into()
    }
}

impl ComplexTensorOps<Flex> for Flex {
    async fn complex_into_real_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_real_data(tensor.into_data()))
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_real_data(tensor.into_data()))
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(tensor.into_data())
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<(TensorData, TensorData), burn_backend::ExecutionError> {
        Ok(interleaved_data_to_split_data(tensor.into_data()))
    }

    fn to_complex(tensor: burn_complex::base::FloatTensor<Flex>) -> ComplexTensor<Flex> {
        let interleaved_data = interleaved_data_from_real_data(tensor.into_data());
        FlexTensor::from_data(interleaved_data).into()
    }

    fn complex_squared_norm(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.norm_sqr())
    }

    fn complex_device(_tensor: &ComplexTensor<Flex>) -> ComplexDevice<Flex> {
        Default::default()
    }

    fn complex_to_device(
        tensor: ComplexTensor<Flex>,
        _device: &ComplexDevice<Flex>,
    ) -> ComplexTensor<Flex> {
        tensor
    }

    fn complex_add(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a + b)
    }

    fn complex_sub(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a - b)
    }

    fn complex_mul(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a * b)
    }

    fn complex_div(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a / b)
    }

    fn real(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.real)
    }

    fn imag(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.imag)
    }

    fn complex_abs(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.norm())
    }

    fn complex_from_parts(
        real: burn_complex::base::FloatTensor<Flex>,
        imag: burn_complex::base::FloatTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        <Flex as ComplexTensorBackend>::complex_from_split_data(
            real.into_data(),
            imag.into_data(),
            &FlexDevice,
        )
    }

    fn complex_from_polar(
        magnitude: burn_complex::base::FloatTensor<Flex>,
        phase: burn_complex::base::FloatTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        // Convert polar to cartesian: magnitude * e^(i*phase) = magnitude * (cos(phase) + i*sin(phase))
        let real_part = crate::ops::binary::binary_op(
            magnitude.clone(),
            crate::ops::unary::cos(phase.clone()),
            |m, cos_p| m * cos_p,
            |m, cos_p| m * cos_p,
            None,
        );
        let imag_part = crate::ops::binary::binary_op(
            magnitude,
            crate::ops::unary::sin(phase),
            |m, sin_p| m * sin_p,
            |m, sin_p| m * sin_p,
            None,
        );
        Self::complex_from_parts(real_part, imag_part)
    }

    fn complex_exp(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.exp())
    }

    fn complex_log(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.ln())
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
        out_dtype: BoolDType,
    ) -> burn_complex::base::BoolTensor<Flex> {
        let lhs = lhs;
        let rhs = rhs;
        let f32_cmp = |a, b| a != b;
        let f64_cmp = |a, b| a != b;
        //let _simd_hint = Some(CompareOp::Ne);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_elem_typed(lhs, rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> burn_complex::base::BoolTensor<Flex> {
        let lhs = lhs;
        let rhs = rhs;
        let f32_cmp = |a, b| a == b;
        let f64_cmp = |a, b| a == b;
        //let _simd_hint = Some(CompareOp::Eq);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_elem_typed(lhs, rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_equal(
        lhs: ComplexTensor<Flex>,
        rhs: ComplexTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> burn_complex::base::BoolTensor<Flex> {
        let lhs = lhs;
        let rhs = rhs;
        let f32_cmp = |a, b| a == b;
        let f64_cmp = |a, b| a == b;
        //let _simd_hint = Some(CompareOp::Eq);
        debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

        // Broadcast to same shape if needed
        let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_typed::<Complex<f32>, _>(lhs, &rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_typed::<Complex<f64>, _>(lhs, &rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_not_equal(
        lhs: ComplexTensor<Flex>,
        rhs: ComplexTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> burn_complex::base::BoolTensor<Flex> {
        let lhs = lhs;
        let rhs = rhs;
        let f32_cmp = |a, b| a != b;
        let f64_cmp = |a, b| a != b;
        //let _simd_hint = Some(CompareOp::Ne);
        debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

        // Broadcast to same shape if needed
        let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_typed::<Complex<f32>, _>(lhs, &rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_typed::<Complex<f64>, _>(lhs, &rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<Flex>,
        indices: burn_complex::base::IntTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => {
                crate::ops::gather_scatter::gather::<Complex<f32>>(tensor, dim, indices)
            }
            DType::Complex64 => {
                crate::ops::gather_scatter::gather::<Complex<f64>>(tensor, dim, indices)
            }
            _ => panic!("complex_gather: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<Flex>,
        indices: burn_complex::base::IntTensor<Flex>,
        values: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::gather_scatter::scatter_add::<Complex<f32>>(
                tensor, dim, indices, values,
            ),
            DType::Complex64 => crate::ops::gather_scatter::scatter_add::<Complex<f64>>(
                tensor, dim, indices, values,
            ),
            _ => panic!(
                "complex_scatter_add: unsupported dtype {:?}",
                tensor.dtype()
            ),
        }
    }
}

#[macro_export]
macro_rules! c2c_binary_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        $crate::ops::complex::c2c_binary_op(
            $lhs,
            $rhs,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2c_scalar_op {
    ($tensor:expr, $scalar:expr, $op:expr) => {
        $crate::ops::complex::c2c_scalar_op(
            $tensor,
            $scalar,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2r_binary_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        $crate::ops::complex::c2r_binary_op(
            $lhs,
            $rhs,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2r_unary_op {
    ($tensor:expr, $op:expr) => {
        $crate::ops::complex::c2r_unary_op($tensor, $op, $op)
    };
}

#[macro_export]
macro_rules! c2c_unary_op {
    ($tensor:expr, $op:expr) => {
        $crate::ops::complex::c2c_unary_op($tensor, $op, $op)
    };
}

#[macro_export]
macro_rules! c2r_scalar_op {
    ($tensor:expr, $scalar:expr, $op:expr) => {
        $crate::ops::complex::c2r_scalar_op(
            $tensor,
            $scalar,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

pub fn c2c_binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    use crate::ops::binary::binary_op_typed;

    debug_assert_eq!(
        lhs.dtype(),
        rhs.dtype(),
        "complex_binary_op: dtype mismatch"
    );

    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::Complex32 => binary_op_typed::<Complex<f32>, _>(lhs, &rhs, f32_op),
        DType::Complex64 => binary_op_typed::<Complex<f64>, _>(lhs, &rhs, f64_op),
        _ => panic!("complex_binary_op: unsupported dtype {:?}", lhs.dtype()),
    }
}

pub fn c2c_scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: Complex<f64>,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    match tensor.dtype() {
        DType::Complex32 => {
            let s = Complex {
                real: scalar.real as f32,
                imag: scalar.imag as f32,
            };
            scalar_op_typed::<Complex<f32>, _>(tensor, s, f32_op)
        }
        DType::Complex64 => scalar_op_typed::<Complex<f64>, _>(tensor, scalar, f64_op),
        _ => panic!("complex_scalar_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2r_binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    use crate::ops::binary::binary_op_typed;

    debug_assert_eq!(
        lhs.dtype(),
        rhs.dtype(),
        "complex_binary_op: dtype mismatch"
    );

    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::Complex32 => binary_op_typed::<Complex<f32>, _>(lhs, &rhs, f32_op),
        DType::Complex64 => binary_op_typed::<Complex<f64>, _>(lhs, &rhs, f64_op),
        _ => panic!("complex_binary_op: unsupported dtype {:?}", lhs.dtype()),
    }
}

pub fn c2r_scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: Complex<f64>,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    match tensor.dtype() {
        DType::Complex32 => {
            let s = Complex {
                real: scalar.real as f32,
                imag: scalar.imag as f32,
            };
            scalar_op_typed(tensor, s, f32_op)
        }
        DType::Complex64 => scalar_op_typed(tensor, scalar, f64_op),
        _ => panic!("complex_scalar_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2r_unary_op<F32Op, F64Op>(tensor: FlexTensor, f32_op: F32Op, f64_op: F64Op) -> FlexTensor
where
    F32Op: Fn(Complex<f32>) -> f32,
    F64Op: Fn(Complex<f64>) -> f64,
{
    use crate::ops::unary::unary_op_typed_convert;

    match tensor.dtype() {
        DType::Complex32 => unary_op_typed_convert(tensor, f32_op),
        DType::Complex64 => unary_op_typed_convert(tensor, f64_op),
        _ => panic!("c2r_unary_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2c_unary_op<F32Op, F64Op>(tensor: FlexTensor, f32_op: F32Op, f64_op: F64Op) -> FlexTensor
where
    F32Op: Fn(Complex<f32>) -> Complex<f32>,
    F64Op: Fn(Complex<f64>) -> Complex<f64>,
{
    use crate::ops::unary::unary_op_typed;

    match tensor.dtype() {
        DType::Complex32 => unary_op_typed(tensor, f32_op),
        DType::Complex64 => unary_op_typed(tensor, f64_op),
        _ => panic!("c2c_unary_op: unsupported dtype {:?}", tensor.dtype()),
    }
}
