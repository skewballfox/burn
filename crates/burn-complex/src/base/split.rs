use crate::{
    base::{
        ComplexDevice, ComplexTensor, ComplexTensorBackend, ComplexTensorOps, Layout, SplitLayout, SplitTensorData, element::Complex
    },
    utils::real_to_complex_dtype,
};
use burn_std::FloatDType;
use burn_tensor::{
    ElementComparison, Scalar, TensorData, TensorMetadata, backend::Backend, cast::ToElement,
    ops::FloatTensorOps,
};
use bytemuck::Pod;

impl<T: TensorMetadata + 'static> Layout for SplitLayout<T> {
    type ComplexTensorPrimitive = SplitComplexTensor<T>;
}

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<P: TensorMetadata> {
    pub real: P,
    pub imag: P,
}

impl<T: TensorMetadata + 'static> TensorMetadata for SplitComplexTensor<T> {
    fn shape(&self) -> burn_tensor::Shape {
        self.real.shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }

    fn dtype(&self) -> burn_std::DType {
        match self.real.dtype() {
            burn_std::DType::F32 => burn_std::DType::Complex32,
            burn_std::DType::F64 => burn_std::DType::Complex64,
            dtype => panic!("Unsupported dtype for complex tensor: {dtype:?}"),
        }
    }
}

/// A newtype that wraps a real backend B and exposes a split-layout complex backend.
pub struct SplitBackend<B: Backend>(core::marker::PhantomData<B>);

impl<B: Backend> ComplexTensorBackend for SplitBackend<B>
where
    <B as Backend>::FloatElem: ElementComparison + Pod,
    B::FloatTensorPrimitive: TensorMetadata + 'static,
{
    type InnerBackend = B;
    type ComplexScalar = Complex<B::FloatElem>;
    type Layout = SplitLayout<B::FloatTensorPrimitive>;

    fn complex_from_real_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        let real = B::float_from_data(data, device);
        // https://github.com/rust-lang/rust/issues/54628
        let imag = B::float_from_data(
            TensorData::from_bytes_vec(
                vec![0u8; real.shape().num_elements() * real.dtype().size()],
                real.shape().clone(),
                real.dtype(),
            ),
            device,
        );
        SplitComplexTensor { real, imag }
    }

    fn complex_from_imag_data(
        data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let imag = B::float_from_data(data, device);
        // https://github.com/rust-lang/rust/issues/54628
        let real = B::float_from_data(
            TensorData::from_bytes_vec(
                vec![0u8; imag.shape().num_elements() * imag.dtype().size()],
                imag.shape().clone(),
                imag.dtype(),
            ),
            device,
        );
        SplitComplexTensor { real, imag }
    }
    // Should these be a result
    fn complex_from_interleaved_data(
        data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let mut real_bytes: Vec<u8> = Vec::with_capacity(data.bytes.len() / 2);
        let mut imag_bytes: Vec<u8> = Vec::with_capacity(data.bytes.len() / 2);

        let element_size = data.dtype.size();
        let complex_pair_size = 2 * element_size;

        // Iterate through the bytes in chunks of (Real + Imag)
        for chunk in data.bytes.chunks_exact(complex_pair_size) {
            let (real_part, imag_part) = chunk.split_at(element_size);

            real_bytes.extend_from_slice(real_part);
            imag_bytes.extend_from_slice(imag_part);
        }
        SplitComplexTensor {
            real: B::float_from_data(
                TensorData::from_bytes_vec(real_bytes, data.shape.clone(), data.dtype),
                device,
            ),
            imag: B::float_from_data(
                TensorData::from_bytes_vec(imag_bytes, data.shape, data.dtype),
                device,
            ),
        }
    }

    fn complex_from_split_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let real = B::float_from_data(real_data, device);
        let imag = B::float_from_data(imag_data, device);
        assert_eq!(
            real.shape(),
            imag.shape(),
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            real.dtype(),
            imag.dtype(),
            "Real and imaginary parts must have the same dtype"
        );
        SplitComplexTensor { real, imag }
    }
}
type FlOps<B> = <SplitBackend<B> as ComplexTensorBackend>::InnerBackend;
impl<B> ComplexTensorOps<SplitBackend<B>> for SplitBackend<B>
where
    B: Backend,
    <B as Backend>::FloatElem: ElementComparison + Pod,
{
    fn to_complex(tensor: super::FloatTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            imag: B::float_zeros(
                tensor.shape().clone(),
                &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor),
                tensor.dtype().into(),
            ),
            real: tensor,
        }
    }

    fn real(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.real
    }
    fn imag(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.imag
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_not_equal_elem(
            lhs_real,
            burn_tensor::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_not_equal_elem(
            lhs_imag,
            burn_tensor::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_or(real_cmp, imag_cmp)
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_equal_elem(
            lhs_real,
            burn_tensor::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_equal_elem(
            lhs_imag,
            burn_tensor::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_equal(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let real_cmp = B::float_equal(lhs.real, rhs.real, out_dtype);
        let imag_cmp = B::float_equal(lhs.imag, rhs.imag, out_dtype);
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_not_equal(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let real_cmp = B::float_not_equal(lhs.real, rhs.real, out_dtype);
        let imag_cmp = B::float_not_equal(lhs.imag, rhs.imag, out_dtype);
        B::bool_or(real_cmp, imag_cmp)
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.real).await
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.imag).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        let element_size = real_data.dtype.size();
        let mut interleaved_bytes = Vec::with_capacity(real_data.bytes.len() * 2);
        for (real_chunk, imag_chunk) in real_data
            .bytes
            .chunks_exact(element_size)
            .zip(imag_data.bytes.chunks_exact(element_size))
        {
            interleaved_bytes.extend_from_slice(real_chunk);
            interleaved_bytes.extend_from_slice(imag_chunk);
        }
        Ok(TensorData::from_bytes_vec(
            interleaved_bytes,
            real_data.shape,
            real_to_complex_dtype(real_data.dtype),
        ))
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<SplitTensorData, burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        Ok(SplitTensorData {
            real_bytes: real_data.bytes,
            imag_bytes: imag_data.bytes,
            shape: real_data.shape,
            dtype: real_data.dtype,
        })
    }

    fn complex_device(tensor: &ComplexTensor<SplitBackend<B>>) -> ComplexDevice<SplitBackend<B>> {
        <<SplitBackend<B> as ComplexTensorBackend>::InnerBackend as FloatTensorOps<
            <SplitBackend<B> as ComplexTensorBackend>::InnerBackend,
        >>::float_device(&tensor.real)
    }

    fn complex_add(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_add(lhs.real, rhs.real),
            imag: FlOps::<B>::float_add(lhs.imag, rhs.imag),
        }
    }

    fn complex_sub(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_sub(lhs.real, rhs.real),
            imag: FlOps::<B>::float_sub(lhs.imag, rhs.imag),
        }
    }

    fn complex_mul(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_sub(
                FlOps::<B>::float_mul(lhs.real.clone(), rhs.real.clone()),
                FlOps::<B>::float_mul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            imag: FlOps::<B>::float_add(
                FlOps::<B>::float_mul(lhs.real, rhs.imag),
                FlOps::<B>::float_mul(rhs.real, lhs.imag),
            ),
        }
    }

    fn complex_div(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]

        let norm_sqr = SplitBackend::<B>::complex_squared_norm(rhs.clone());

        SplitComplexTensor {
            real: FlOps::<B>::float_div(
                FlOps::<B>::float_add(
                    FlOps::<B>::float_mul(lhs.real.clone(), rhs.real.clone()),
                    FlOps::<B>::float_mul(lhs.imag.clone(), rhs.imag.clone()),
                ),
                norm_sqr.clone(),
            ),
            imag: FlOps::<B>::float_div(
                FlOps::<B>::float_sub(
                    FlOps::<B>::float_mul(lhs.imag.clone(), rhs.real.clone()),
                    FlOps::<B>::float_mul(lhs.real.clone(), rhs.imag.clone()),
                ),
                norm_sqr.clone(),
            ),
        }
    }
    fn complex_abs(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        //todo! https://github.com/tracel-ai/burn/issues/4836
        // |z| = sqrt(real^2 + imag^2)
        let real_sq = FlOps::<B>::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = FlOps::<B>::float_mul(tensor.imag.clone(), tensor.imag.clone());
        let norm_sq = FlOps::<B>::float_add(real_sq, imag_sq);
        FlOps::<B>::float_sqrt(norm_sq)
    }

    fn complex_from_parts(
        real: super::FloatTensor<SplitBackend<B>>,
        imag: super::FloatTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor { real, imag }
    }

    fn complex_exp(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let exp_real = FlOps::<B>::float_exp(tensor.real.clone());
        let cos_imag = FlOps::<B>::float_cos(tensor.imag.clone());
        let sin_imag = FlOps::<B>::float_sin(tensor.imag);

        SplitComplexTensor {
            real: FlOps::<B>::float_mul(exp_real.clone(), cos_imag),
            imag: FlOps::<B>::float_mul(exp_real, sin_imag),
        }
    }

    fn complex_log(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let real_sq = FlOps::<B>::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = FlOps::<B>::float_mul(tensor.imag.clone(), tensor.imag.clone());
        let norm_sq = FlOps::<B>::float_add(real_sq, imag_sq);
        let norm = FlOps::<B>::float_sqrt(norm_sq);

        // Compute arg: atan2(imag, real)
        let arg = FlOps::<B>::float_atan2(tensor.imag, tensor.real);

        SplitComplexTensor {
            real: FlOps::<B>::float_log(norm),
            imag: arg,
        }
    }

    fn complex_squared_norm(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> super::FloatTensor<SplitBackend<B>> {
        let real_sq = FlOps::<B>::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = FlOps::<B>::float_mul(tensor.imag.clone(), tensor.imag.clone());
        FlOps::<B>::float_add(real_sq, imag_sq)
    }

    fn complex_from_polar(
        magnitude: super::FloatTensor<SplitBackend<B>>,
        phase: super::FloatTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_mul(magnitude.clone(), FlOps::<B>::float_cos(phase.clone())),
            imag: FlOps::<B>::float_mul(magnitude, FlOps::<B>::float_sin(phase)),
        }
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B>>,
        indices: super::IntTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_gather(dim, tensor.real, indices.clone()),
            imag: B::float_gather(dim, tensor.imag, indices),
        }
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B>>,
        indices: super::IntTensor<SplitBackend<B>>,
        values: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_scatter_add(dim, tensor.real, indices.clone(), values.real),
            imag: B::float_scatter_add(dim, tensor.imag, indices, values.imag),
        }
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_tensor::Distribution,
        device: &ComplexDevice<SplitBackend<B>>,
        dtype: FloatDType,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_random(shape.clone(), distribution, device, dtype),
            imag: B::float_random(shape, distribution, device, dtype),
        }
    }

    fn complex_to_device(
        tensor: ComplexTensor<SplitBackend<B>>,
        device: &ComplexDevice<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_to_device(tensor.real, device),
            imag: B::float_to_device(tensor.imag, device),
        }
    }

    fn complex_reshape(
        tensor: ComplexTensor<SplitBackend<B>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_reshape(tensor.real, shape.clone()),
            imag: B::float_reshape(tensor.imag, shape),
        }
    }

    fn complex_transpose(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_transpose(tensor.real),
            imag: B::float_transpose(tensor.imag),
        }
    }

    fn complex_neg(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_neg(tensor.real),
            imag: B::float_neg(tensor.imag),
        }
    }

    fn complex_conj(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // conj(a + bi) = a - bi
        SplitComplexTensor {
            real: tensor.real,
            imag: B::float_neg(tensor.imag),
        }
    }

    fn complex_arg(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        // arg(a + bi) = atan2(b, a)
        FlOps::<B>::float_atan2(tensor.imag, tensor.real)
    }

    fn complex_powc(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // z^w = exp(w * ln(z))
        let log_lhs = SplitBackend::<B>::complex_log(lhs);
        let product = SplitBackend::<B>::complex_mul(rhs, log_lhs);
        SplitBackend::<B>::complex_exp(product)
    }

    fn complex_sqrt(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // sqrt(z) = from_polar(sqrt(|z|), arg(z) / 2)
        let abs = SplitBackend::<B>::complex_abs(tensor.clone());
        let sqrt_abs = FlOps::<B>::float_sqrt(abs);
        let arg = FlOps::<B>::float_atan2(tensor.imag, tensor.real);
        let half_arg = FlOps::<B>::float_div_scalar(arg, burn_tensor::Scalar::Float(2.0));
        SplitBackend::<B>::complex_from_polar(sqrt_abs, half_arg)
    }

    fn complex_sin(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        SplitComplexTensor {
            real: FlOps::<B>::float_mul(
                FlOps::<B>::float_sin(tensor.real.clone()),
                FlOps::<B>::float_cosh(tensor.imag.clone()),
            ),
            imag: FlOps::<B>::float_mul(
                FlOps::<B>::float_cos(tensor.real),
                FlOps::<B>::float_sinh(tensor.imag),
            ),
        }
    }

    fn complex_cos(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        SplitComplexTensor {
            real: FlOps::<B>::float_mul(
                FlOps::<B>::float_cos(tensor.real.clone()),
                FlOps::<B>::float_cosh(tensor.imag.clone()),
            ),
            imag: FlOps::<B>::float_neg(FlOps::<B>::float_mul(
                FlOps::<B>::float_sin(tensor.real),
                FlOps::<B>::float_sinh(tensor.imag),
            )),
        }
    }

    fn complex_tan(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // tan(z) = sin(z) / cos(z)
        let sin = SplitBackend::<B>::complex_sin(tensor.clone());
        let cos = SplitBackend::<B>::complex_cos(tensor);
        SplitBackend::<B>::complex_div(sin, cos)
    }

    fn complex_slice(
        tensor: ComplexTensor<SplitBackend<B>>,
        slices: &[burn_tensor::Slice],
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_slice(tensor.real, slices),
            imag: B::float_slice(tensor.imag, slices),
        }
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<SplitBackend<B>>,
        ranges: &[burn_tensor::Slice],
        value: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_slice_assign(tensor.real, ranges, value.real),
            imag: B::float_slice_assign(tensor.imag, ranges, value.imag),
        }
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_swap_dims(tensor.real, dim1, dim2),
            imag: B::float_swap_dims(tensor.imag, dim1, dim2),
        }
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_repeat_dim(tensor.real, dim, times),
            imag: B::float_repeat_dim(tensor.imag, dim, times),
        }
    }

    fn complex_cat(
        tensors: Vec<ComplexTensor<SplitBackend<B>>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        let (reals, imags): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.real, t.imag)).unzip();
        SplitComplexTensor {
            real: B::float_cat(reals, dim),
            imag: B::float_cat(imags, dim),
        }
    }

    fn complex_any(
        tensor: ComplexTensor<SplitBackend<B>>,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let real_any = B::float_any(tensor.real, out_dtype);
        let imag_any = B::float_any(tensor.imag, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let real_any = B::float_any_dim(tensor.real, dim, out_dtype);
        let imag_any = B::float_any_dim(tensor.imag, dim, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_all(
        tensor: ComplexTensor<SplitBackend<B>>,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        // A complex element is nonzero if either real or imag is nonzero.
        // all(z != 0) = all(real != 0 || imag != 0)
        let real_nonzero =
            B::float_not_equal_elem(tensor.real, burn_tensor::Scalar::Float(0.0), out_dtype);
        let imag_nonzero =
            B::float_not_equal_elem(tensor.imag, burn_tensor::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all(elem_nonzero)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> super::BoolTensor<SplitBackend<B>> {
        let real_nonzero =
            B::float_not_equal_elem(tensor.real, burn_tensor::Scalar::Float(0.0), out_dtype);
        let imag_nonzero =
            B::float_not_equal_elem(tensor.imag, burn_tensor::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all_dim(elem_nonzero, dim)
    }

    fn complex_permute(
        tensor: ComplexTensor<SplitBackend<B>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_permute(tensor.real, axes),
            imag: B::float_permute(tensor.imag, axes),
        }
    }

    fn complex_expand(
        tensor: ComplexTensor<SplitBackend<B>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_expand(tensor.real, shape.clone()),
            imag: B::float_expand(tensor.imag, shape),
        }
    }

    fn complex_flip(
        tensor: ComplexTensor<SplitBackend<B>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_flip(tensor.real, axes),
            imag: B::float_flip(tensor.imag, axes),
        }
    }

    fn complex_unfold(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_unfold(tensor.real, dim, size, step),
            imag: B::float_unfold(tensor.imag, dim, size, step),
        }
    }

    fn complex_select(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        indices: super::IntTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_select(tensor.real, dim, indices.clone()),
            imag: B::float_select(tensor.imag, dim, indices),
        }
    }

    fn complex_sum(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_sum(tensor.real),
            imag: B::float_sum(tensor.imag),
        }
    }

    fn complex_sum_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_sum_dim(tensor.real, dim),
            imag: B::float_sum_dim(tensor.imag, dim),
        }
    }

    fn complex_prod(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // prod(z) = exp(sum(log(z)))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let sum_log = SplitBackend::<B>::complex_sum(log_tensor);
        SplitBackend::<B>::complex_exp(sum_log)
    }

    fn complex_prod_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        // prod_dim(z, dim) = exp(sum_dim(log(z), dim))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let sum_log = SplitBackend::<B>::complex_sum_dim(log_tensor, dim);
        SplitBackend::<B>::complex_exp(sum_log)
    }

    fn complex_mean(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_mean(tensor.real),
            imag: B::float_mean(tensor.imag),
        }
    }

    fn complex_mean_dim(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_mean_dim(tensor.real, dim),
            imag: B::float_mean_dim(tensor.imag, dim),
        }
    }

    fn complex_remainder(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // Componentwise remainder (matching Complex<E> Rem impl)
        SplitComplexTensor {
            real: B::float_remainder(lhs.real, rhs.real),
            imag: B::float_remainder(lhs.imag, rhs.imag),
        }
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B>> {
        use burn_tensor::cast::ToElement;
        SplitComplexTensor {
            real: B::float_remainder_scalar(
                lhs.real,
                burn_tensor::Scalar::Float(rhs.real().to_f64()),
            ),
            imag: B::float_remainder_scalar(
                lhs.imag,
                burn_tensor::Scalar::Float(rhs.imag().to_f64()),
            ),
        }
    }

    fn complex_mask_where(
        tensor: ComplexTensor<SplitBackend<B>>,
        mask: super::BoolTensor<SplitBackend<B>>,
        source: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_mask_where(tensor.real, mask.clone(), source.real),
            imag: B::float_mask_where(tensor.imag, mask, source.imag),
        }
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<SplitBackend<B>>,
        mask: super::BoolTensor<SplitBackend<B>>,
        value: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B>> {
        use burn_tensor::cast::ToElement;
        SplitComplexTensor {
            real: B::float_mask_fill(
                tensor.real,
                mask.clone(),
                burn_tensor::Scalar::Float(value.real().to_f64()),
            ),
            imag: B::float_mask_fill(
                tensor.imag,
                mask,
                burn_tensor::Scalar::Float(value.imag().to_f64()),
            ),
        }
    }

    fn complex_sign(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = SplitBackend::<B>::complex_abs(tensor.clone());
        SplitComplexTensor {
            real: FlOps::<B>::float_div(tensor.real, abs.clone()),
            imag: FlOps::<B>::float_div(tensor.imag, abs),
        }
    }

    fn complex_matmul(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        SplitComplexTensor {
            real: FlOps::<B>::float_sub(
                B::float_matmul(lhs.real.clone(), rhs.real.clone()),
                B::float_matmul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            imag: FlOps::<B>::float_add(
                B::float_matmul(lhs.real, rhs.imag),
                B::float_matmul(lhs.imag, rhs.real),
            ),
        }
    }

    fn complex_cumsum(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        // cumsum is linear, so it works componentwise
        SplitComplexTensor {
            real: B::float_cumsum(tensor.real, dim),
            imag: B::float_cumsum(tensor.imag, dim),
        }
    }

    fn complex_cumprod(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B>> {
        // cumprod(z, dim) = exp(cumsum(log(z), dim))
        let log_tensor = SplitBackend::<B>::complex_log(tensor);
        let cumsum_log = SplitBackend::<B>::complex_cumsum(log_tensor, dim);
        SplitBackend::<B>::complex_exp(cumsum_log)
    }

    fn complex_select_add(
        tensor: ComplexTensor<SplitBackend<B>>,
        dim: usize,
        indices: super::IntTensor<SplitBackend<B>>,
        values: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: B::float_select_add(tensor.real, dim, indices.clone(), values.real),
            imag: B::float_select_add(tensor.imag, dim, indices, values.imag),
        }
    }

    fn complex_powc_scalar(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B>> {
        // z^c = exp(c * ln(z)) where c = a + bi is a scalar
        // (a + bi) * (u + vi) = (au - bv) + (av + bu)i
        let a = rhs.real().to_f64();
        let b = rhs.imag().to_f64();
        let ln_z = SplitBackend::<B>::complex_log(lhs);
        let c_ln_z = SplitComplexTensor {
            real: FlOps::<B>::float_sub(
                FlOps::<B>::float_mul_scalar(ln_z.real.clone(), burn_tensor::Scalar::Float(a)),
                FlOps::<B>::float_mul_scalar(ln_z.imag.clone(), burn_tensor::Scalar::Float(b)),
            ),
            imag: FlOps::<B>::float_add(
                FlOps::<B>::float_mul_scalar(ln_z.real, burn_tensor::Scalar::Float(b)),
                FlOps::<B>::float_mul_scalar(ln_z.imag, burn_tensor::Scalar::Float(a)),
            ),
        };
        SplitBackend::<B>::complex_exp(c_ln_z)
    }

    fn complex_powf(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: super::FloatTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // z^w = exp(w * ln(z)) where w is a real tensor
        let log_z = SplitBackend::<B>::complex_log(lhs);
        let w_log_z = SplitComplexTensor {
            real: FlOps::<B>::float_mul(rhs.clone(), log_z.real),
            imag: FlOps::<B>::float_mul(rhs, log_z.imag),
        };
        SplitBackend::<B>::complex_exp(w_log_z)
    }

    fn complex_powf_scalar(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_powf_scalar(lhs.real, rhs),
            imag: FlOps::<B>::float_powf_scalar(lhs.imag, rhs),
        }
    }
}
