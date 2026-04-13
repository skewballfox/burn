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
use burn_std::DType;

use crate::{Flex, FlexTensor, ops::binary::scalar_op_typed};

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
        crate::complex_binary_op!(lhs, rhs, |a, b| a + b)
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
        crate::complex_binary_op!(lhs, rhs, |a, b| a + b)
    }

    fn complex_sub(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::complex_binary_op!(lhs, rhs, |a, b| a - b)
    }

    fn complex_mul(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::complex_binary_op!(lhs, rhs, |a, b| a * b)
    }

    fn complex_div(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        todo!()
    }

    fn real(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        todo!()
    }

    fn imag(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        todo!()
    }

    fn complex_abs(tensor: ComplexTensor<Flex>) -> burn_complex::base::FloatTensor<Flex> {
        todo!()
    }

    fn complex_from_parts(
        real: burn_complex::base::FloatTensor<Flex>,
        imag: burn_complex::base::FloatTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        todo!()
    }

    fn complex_from_polar(
        magnitude: burn_complex::base::FloatTensor<Flex>,
        phase: burn_complex::base::FloatTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        todo!()
    }

    fn complex_exp(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        todo!()
    }

    fn complex_log(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        todo!()
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
    ) -> burn_complex::base::BoolTensor<Flex> {
        todo!()
    }
}

#[macro_export]
macro_rules! complex_binary_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        $crate::ops::complex::complex_binary_op(
            $lhs,
            $rhs,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[cfg(feature = "complex")]
#[macro_export]
macro_rules! complex_scalar_op {
    ($tensor:expr, $scalar:expr, $op:expr) => {
        $crate::ops::complex::complex_scalar_op(
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

pub fn complex_scalar_op<F32Op, F64Op>(
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
