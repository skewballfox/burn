// impl<B> SplitComplexTensor<B::FloatTensorPrimitive>
// where
//     B: Backend,
// {

//     fn empty(shape: Shape, device: &B::Device, dtype: DType) -> Self {
//         // should I check then pass the dtype?
//         SplitBackend::<B>::complex_zeros(shape, device)
//     }

//     fn reshape(tensor: Self, shape: Shape) -> Self {
//         SplitBackend::<B>::complex_reshape(tensor, shape)
//     }

//     fn transpose(tensor: Self) -> Self {
//         SplitBackend::<B>::complex_transpose(tensor)
//     }

//     fn swap_dims(tensor: Self, dim1: usize, dim2: usize) -> Self {
//         SplitBackend::<B>::complex_swap_dims(tensor, dim1, dim2)
//     }

//     fn slice(tensor: Self, ranges: &[Slice]) -> Self {
//         //TensorPrimitive::Complex(B::complex_slice(tensor, ranges))
//         SplitBackend::<B>::complex_slice(tensor, ranges)
//     }

//     fn device(tensor: &Self) -> Device<B> {
//         SplitBackend::<B>::complex_device(tensor)
//     }

//     fn to_device(tensor: Self, device: &B::Device) -> Self {
//         SplitBackend::<B>::complex_to_device(tensor, device)
//     }

//     async fn into_data_async(tensor: Self) -> Result<TensorData, ExecutionError> {
//         SplitBackend::<B>::complex_into_interleaved_data(tensor).await
//     }

//     fn from_data(data: TensorData, device: &B::Device, dtype: DType) -> Self {
//         SplitBackend::<B>::complex_from_interleaved_data(data.convert::<<SplitBackend<B> as ComplexTensorBackend>::ComplexScalar>(), device)
//     }

//     fn repeat_dim(tensor: Self, dim: usize, times: usize) -> Self {
//         SplitBackend::<B>::complex_repeat_dim(tensor, dim, times)
//     }
//     fn equal(lhs: Self, rhs: Self) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&lhs)).bool_dtype;
//         SplitBackend::<B>::complex_equal(lhs, rhs, out_dtype)
//     }

//     fn not_equal(lhs: Self, rhs: Self) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&lhs)).bool_dtype;
//         SplitBackend::<B>::complex_not_equal(lhs, rhs, out_dtype)
//     }

//     fn cat(tensors: Vec<Self>, dim: usize) -> Self {
//         SplitBackend::<B>::complex_cat(tensors, dim)
//     }

//     fn any(tensor: Self) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&tensor)).bool_dtype;
//         SplitBackend::<B>::complex_any(tensor, out_dtype)
//     }

//     fn any_dim(tensor: Self, dim: usize) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&tensor)).bool_dtype;
//         SplitBackend::<B>::complex_any_dim(tensor, dim, out_dtype)
//     }

//     fn all(tensor: Self) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&tensor)).bool_dtype;
//         SplitBackend::<B>::complex_all(tensor, out_dtype)
//     }

//     fn all_dim(tensor: Self, dim: usize) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&tensor)).bool_dtype;
//         SplitBackend::<B>::complex_all_dim(tensor, dim, out_dtype)
//     }

//     fn permute(tensor: Self, axes: &[usize]) -> Self {
//         SplitBackend::<B>::complex_permute(tensor, axes)
//     }

//     fn expand(tensor: Self, shape: Shape) -> Self {
//         SplitBackend::<B>::complex_expand(tensor, shape)
//     }

//     fn flip(tensor: Self, axes: &[usize]) -> Self {
//         SplitBackend::<B>::complex_flip(tensor, axes)
//     }

//     fn unfold(tensor: Self, dim: usize, size: usize, step: usize) -> Self {
//         SplitBackend::<B>::complex_unfold(tensor, dim, size, step)
//     }

//     fn slice_assign(
//         tensor: Self,
//         ranges: &[Slice],
//         value: Self,
//     ) -> Self {
//         SplitBackend::<B>::complex_slice_assign(tensor, ranges, value)
//     }

//     fn select(
//         tensor: Self,
//         dim: usize,
//         indices: B::IntTensorPrimitive,
//     ) -> Self {
//         // Uses your existing `select` name.
//         SplitBackend::<B>::complex_select(tensor, dim, indices)
//     }

//     fn select_assign(
//         tensor: Self,
//         dim: usize,
//         indices: B::IntTensorPrimitive,
//         values: Self,
//         update: IndexingUpdateOp,
//     ) -> Self {
//         match update {
//             IndexingUpdateOp::Add => SplitBackend::<B>::complex_select_add(tensor, dim, indices, values),
//             _ => unimplemented!(),
//         }
//     }

//     fn zeros(shape: Shape, device: &B::Device, dtype: DType) -> Self {
//         match dtype {
//             DType::Complex32 | DType::Complex64 => SplitBackend::<B>::complex_zeros(shape, device),
//             _ => panic!("Unsupported complex dtype"),
//         }
//     }

//     fn ones(shape: Shape, device: &B::Device, dtype: DType) -> Self {
//         match dtype {
//             DType::Complex32 | DType::Complex64 => SplitBackend::<B>::complex_ones(shape, device),
//             _ => panic!("Unsupported complex dtype"),
//         }
//     }

//     fn mask_where(
//         tensor: Self,
//         mask: B::BoolTensorPrimitive,
//         source: Self,
//     ) -> Self {
//         SplitBackend::<B>::complex_mask_where(tensor, mask, source)
//     }

//     fn mask_fill(
//         tensor: Self,
//         mask: B::BoolTensorPrimitive,
//         value: burn_tensor::Scalar,
//     ) -> Self {
//         SplitBackend::<B>::complex_mask_fill(tensor, mask, value.elem())
//     }

//     fn gather(
//         dim: usize,
//         tensor: Self,
//         indices: B::IntTensorPrimitive,
//     ) -> Self {
//         SplitBackend::<B>::complex_gather(dim, tensor, indices)
//     }

//     fn scatter(
//         dim: usize,
//         tensor: Self,
//         indices: B::IntTensorPrimitive,
//         values: Self,
//         update: burn_tensor::IndexingUpdateOp,
//     ) -> Self {
//         match update {
//             IndexingUpdateOp::Add => SplitBackend::<B>::complex_scatter_add(dim, tensor, indices, values),
//             _ => unimplemented!(),
//         }
//     }

//     fn equal_elem(
//         lhs: Self,
//         rhs: burn_tensor::Scalar,
//     ) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<B>(&SplitBackend::<B>::complex_device(&lhs)).bool_dtype;
//         SplitBackend::<B>::complex_equal_elem(lhs, rhs.elem(), out_dtype)
//     }

//     fn not_equal_elem(
//         lhs: Self,
//         rhs: burn_tensor::Scalar,
//     ) -> B::BoolTensorPrimitive {
//         let out_dtype = get_device_settings::<SplitBackend::<B>>(&SplitBackend::<B>::complex_device(&lhs)).bool_dtype;
//         SplitBackend::<B>::complex_not_equal_elem(lhs, rhs.elem(), out_dtype)
//     }

//     fn full(
//         shape: Shape,
//         fill_value: burn_tensor::Scalar,
//         device: &SplitBackend::<B>::Device,
//         dtype: DType,
//     ) -> Self {
//         // Enforce complex dtype for clarity (mirrors from_data_dtype below).
//         if !dtype.is_complex() {
//             panic!("Expected complex dtype, got {dtype:?}");
//         }
//         // `elem()` should yield something convertible to `B::ComplexElem`.
//         SplitBackend::<B>::complex_full(shape, fill_value.elem(), device)
//     }

//     fn scatter_nd(
//         data: Self,
//         indices: B::IntTensorPrimitive,
//         values: Self,
//         reduction: IndexingUpdateOp,
//     ) -> Self {
//         SplitBackend::<B>::complex_scatter_nd(data, indices, values, reduction)
//     }

//     fn gather_nd(
//         data: Self,
//         indices: B::IntTensorPrimitive,
//     ) -> Self {
//         todo!()
//     }
// }
// impl<B, F> Numeric<B> for SplitComplexTensor<F>
// where
//     B: ComplexTensorBackend<
//         FloatTensorPrimitive = F,
//         ComplexTensorPrimitive = SplitComplexTensor<F>,
//     >,
//     F: TensorMetadata,
// {
//     type IntTensor = burn_tensor::Int;

//     fn add(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn add_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn sub(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn sub_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn div(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn div_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn remainder(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn remainder_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn mul(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn mul_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn neg(tensor: Self) -> Self {
//         todo!()
//     }

//     fn sign(tensor: Self) -> Self {
//         todo!()
//     }

//     fn sum(tensor: Self) -> Self {
//         todo!()
//     }

//     fn sum_dim(tensor: Self, dim: usize) -> Self {
//         todo!()
//     }

//     fn prod(tensor: Self) -> Self {
//         todo!()
//     }

//     fn prod_dim(tensor: Self, dim: usize) -> Self {
//         todo!()
//     }

//     fn mean(tensor: Self) -> Self {
//         todo!()
//     }

//     fn mean_dim(tensor: Self, dim: usize) -> Self {
//         todo!()
//     }

//     fn cumsum(tensor: Self, dim: usize) -> Self {
//         todo!()
//     }

//     fn cumprod(tensor: Self, dim: usize) -> Self {
//         todo!()
//     }

//     fn powi(lhs: Self, rhs: B::IntTensorPrimitive) -> Self {
//         todo!()
//     }

//     fn powi_scalar(lhs: Self, rhs: burn_tensor::Scalar) -> Self {
//         todo!()
//     }

//     fn random(
//         shape: burn_std::Shape,
//         distribution: burn_tensor::Distribution,
//         device: &B::Device,
//         dtype: burn_std::DType,
//     ) -> Self {
//         todo!()
//     }

//     fn matmul(lhs: Self, rhs: Self) -> Self {
//         todo!()
//     }
// }
