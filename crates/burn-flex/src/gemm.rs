//! The matrix multiplication behind every Flex kernel.
//!
//! This is the `gemm` crate by default. With the `microgemm` feature it is [`microgemm`] instead,
//! with the same arguments. `gemm` allocates its packing buffers from the global allocator and,
//! without `std`, up to 512 KiB for one small product. `microgemm` packs into memory that the
//! caller gives it, here a [`FlexBuf`](crate::FlexBuf) of at most [`SCRATCH`] elements.
//!
//! The `microgemm` path is for embedded targets. It has no SIMD kernels, and it runs each product
//! on one thread.

pub(crate) use ::gemm::Parallelism;

#[cfg(not(feature = "microgemm"))]
pub(crate) use direct::gemm;

#[cfg(feature = "microgemm")]
pub(crate) use portable::gemm;

#[cfg(not(feature = "microgemm"))]
mod direct {
    use super::Parallelism;

    /// `gemm::gemm` without the conjugations: Flex multiplies real elements only.
    ///
    /// # Safety
    ///
    /// As `gemm::gemm`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm<T: Copy + 'static>(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut T,
        dst_col_stride: isize,
        dst_row_stride: isize,
        read_dst: bool,
        lhs: *const T,
        lhs_col_stride: isize,
        lhs_row_stride: isize,
        rhs: *const T,
        rhs_col_stride: isize,
        rhs_row_stride: isize,
        alpha: T,
        beta: T,
        parallelism: Parallelism,
    ) {
        // SAFETY: As the caller's.
        unsafe {
            ::gemm::gemm(
                m,
                n,
                k,
                dst,
                dst_col_stride,
                dst_row_stride,
                read_dst,
                lhs,
                lhs_col_stride,
                lhs_row_stride,
                rhs,
                rhs_col_stride,
                rhs_row_stride,
                alpha,
                beta,
                false,
                false,
                false,
                parallelism,
            );
        }
    }
}

#[cfg(feature = "microgemm")]
mod portable {
    use core::{
        any::TypeId,
        ops::{Add, Mul},
    };

    use super::Parallelism;
    use crate::FlexBuf;
    use burn_std::f16;
    use microgemm::{Kernel, MatMut, MatRef, PackSizes, kernels::GenericKernel4x4};
    use num_traits::{One, Zero};

    type Tile<T> = GenericKernel4x4<T>;

    /// The side of a kernel tile.
    const TILE: usize = 4;
    /// Rows of `lhs`, depth, and columns of `rhs` that are packed at once.
    const MC: usize = 16;
    const KC: usize = 128;
    const NC: usize = 32;
    /// The most elements of scratch for one product: 24 KiB of f32.
    const SCRATCH: usize = KC * (MC + NC);

    /// An element that the kernels multiply in its own precision.
    trait Scalar:
        bytemuck::Pod + Zero + One + Add<Output = Self> + Mul<Output = Self> + Send + Sync
    {
    }
    impl Scalar for f32 {}
    impl Scalar for f64 {}

    /// A matrix in memory. The element at (`row`, `col`) is at `row * rs + col * cs`.
    #[derive(Clone, Copy)]
    struct Strided<P> {
        ptr: P,
        cs: isize,
        rs: isize,
    }

    impl<P> Strided<P> {
        fn offset(&self, row: usize, col: usize) -> isize {
            row.cast_signed() * self.rs + col.cast_signed() * self.cs
        }

        fn transposed(self) -> Self {
            Self {
                ptr: self.ptr,
                cs: self.rs,
                rs: self.cs,
            }
        }
    }

    impl<T: Copy> Strided<*const T> {
        /// # Safety
        ///
        /// The element must be in the matrix.
        unsafe fn get(&self, row: usize, col: usize) -> T {
            // SAFETY: As the caller's.
            unsafe { self.ptr.offset(self.offset(row, col)).read() }
        }
    }

    /// `dst = alpha * dst + beta * lhs * rhs`, or `beta * lhs * rhs` when `read_dst` is false. The
    /// same as `gemm::gemm`.
    ///
    /// # Safety
    ///
    /// The same as `gemm::gemm`: the pointers must be valid for the dimensions and strides, and
    /// `dst` must not overlap `lhs` or `rhs`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm<T: Copy + 'static>(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut T,
        dst_col_stride: isize,
        dst_row_stride: isize,
        read_dst: bool,
        lhs: *const T,
        lhs_col_stride: isize,
        lhs_row_stride: isize,
        rhs: *const T,
        rhs_col_stride: isize,
        rhs_row_stride: isize,
        alpha: T,
        beta: T,
        _parallelism: Parallelism,
    ) {
        macro_rules! call {
            ($ty:ty, $product:ident) => {
                // SAFETY: `T` is `$ty`, so the casts and the copies keep the type. The rest is the
                // caller's guarantee.
                unsafe {
                    $product(
                        [m, n, k],
                        Strided {
                            ptr: dst.cast::<$ty>(),
                            cs: dst_col_stride,
                            rs: dst_row_stride,
                        },
                        read_dst.then(|| core::mem::transmute_copy(&alpha)),
                        Strided {
                            ptr: lhs.cast::<$ty>(),
                            cs: lhs_col_stride,
                            rs: lhs_row_stride,
                        },
                        Strided {
                            ptr: rhs.cast::<$ty>(),
                            cs: rhs_col_stride,
                            rs: rhs_row_stride,
                        },
                        core::mem::transmute_copy(&beta),
                    )
                }
            };
        }

        let element = TypeId::of::<T>();
        if element == TypeId::of::<f32>() {
            call!(f32, product);
        } else if element == TypeId::of::<f64>() {
            call!(f64, product);
        } else if element == TypeId::of::<f16>() {
            call!(f16, product_f16);
        } else {
            panic!("burn-flex: no matrix multiplication for this element type")
        }
    }

    /// `dst = scale * dst + beta * lhs * rhs`. Without `scale`, `dst` is not read: it can be
    /// uninitialized.
    ///
    /// # Safety
    ///
    /// As [`gemm`].
    unsafe fn product<T: Scalar>(
        [m, n, k]: [usize; 3],
        dst: Strided<*mut T>,
        scale: Option<T>,
        lhs: Strided<*const T>,
        rhs: Strided<*const T>,
        beta: T,
    ) {
        // `microgemm` computes `0 * dst` for a `dst` to overwrite, which keeps a NaN.
        for row in 0..m {
            for col in 0..n {
                // SAFETY: The element is in `dst`, and is read only when it is initialized.
                unsafe {
                    let at = dst.ptr.offset(dst.offset(row, col));
                    at.write(match scale {
                        Some(scale) => scale * at.read(),
                        None => T::zero(),
                    });
                }
            }
        }
        if m == 0 || n == 0 || k == 0 {
            return;
        }

        // SAFETY: As the caller's. The transposed product has the same elements.
        unsafe {
            if n == 1 {
                matrix_vector([m, k], dst, lhs, rhs, beta);
            } else if m == 1 {
                let (dst, lhs, rhs) = (dst.transposed(), rhs.transposed(), lhs.transposed());
                matrix_vector([n, k], dst, lhs, rhs, beta);
            } else if dst.cs == 1 && dst.rs == n.cast_signed() {
                packed([m, n, k], dst.ptr, lhs, rhs, beta);
            } else {
                // Between the elements of `dst` there can be memory of another thread.
                for col in 0..n {
                    let dst = Strided {
                        ptr: dst.ptr.offset(dst.offset(0, col)),
                        ..dst
                    };
                    let rhs = Strided {
                        ptr: rhs.ptr.offset(rhs.offset(0, col)),
                        ..rhs
                    };
                    matrix_vector([m, k], dst, lhs, rhs, beta);
                }
            }
        }
    }

    /// `dst += beta * lhs * rhs`, where `dst` and `rhs` are column vectors.
    ///
    /// # Safety
    ///
    /// As [`gemm`]. `dst` must be initialized.
    unsafe fn matrix_vector<T: Scalar>(
        [m, k]: [usize; 2],
        dst: Strided<*mut T>,
        lhs: Strided<*const T>,
        rhs: Strided<*const T>,
        beta: T,
    ) {
        // SAFETY: Every element is in its matrix.
        unsafe {
            if lhs.cs.unsigned_abs() <= lhs.rs.unsigned_abs() {
                // Along the rows of `lhs`, in four sums so that they don't wait for each other.
                for row in 0..m {
                    let mut sums = [T::zero(); 4];
                    for depth in 0..k {
                        let sum = &mut sums[depth % 4];
                        *sum = *sum + lhs.get(row, depth) * rhs.get(depth, 0);
                    }
                    let sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                    let at = dst.ptr.offset(dst.offset(row, 0));
                    at.write(at.read() + beta * sum);
                }
            } else {
                for depth in 0..k {
                    let factor = beta * rhs.get(depth, 0);
                    for row in 0..m {
                        let at = dst.ptr.offset(dst.offset(row, 0));
                        at.write(at.read() + lhs.get(row, depth) * factor);
                    }
                }
            }
        }
    }

    /// `dst += beta * lhs * rhs`, where `dst` is a dense row-major matrix.
    ///
    /// # Safety
    ///
    /// As [`gemm`]. `dst` must be initialized.
    unsafe fn packed<T: Scalar>(
        [m, n, k]: [usize; 3],
        dst: *mut T,
        lhs: Strided<*const T>,
        rhs: Strided<*const T>,
        beta: T,
    ) {
        let sizes = PackSizes {
            mc: m.next_multiple_of(TILE).min(MC),
            kc: k.min(KC),
            nc: n.next_multiple_of(TILE).min(NC),
        };
        let mut scratch = FlexBuf::<T>::zeroed(sizes.buf_len());
        debug_assert!(scratch.len() <= SCRATCH);

        let (mut lhs_copy, mut rhs_copy) = (None, None);
        // SAFETY: The matrices are valid for their dimensions. `dst` is dense and initialized, and
        // nothing else refers to it.
        unsafe {
            let lhs = view([m, k], lhs, &mut lhs_copy);
            let rhs = view([k, n], rhs, &mut rhs_copy);
            let dst = core::slice::from_raw_parts_mut(dst, m * n);
            Tile::<T>::new().gemm(
                beta,
                lhs,
                rhs,
                T::one(),
                &mut MatMut::row_major(m, n, dst),
                sizes,
                &mut scratch,
            );
        }
    }

    /// `matrix` as `microgemm` takes it. One with a negative stride (a flipped tensor) or
    /// without one (an expanded tensor) is copied into `copy`.
    ///
    /// # Safety
    ///
    /// `matrix` must be valid for `rows` and `cols`, which must not be zero, and the memory between
    /// its elements must be initialized.
    unsafe fn view<T: Scalar>(
        [rows, cols]: [usize; 2],
        matrix: Strided<*const T>,
        copy: &mut Option<FlexBuf<T>>,
    ) -> MatRef<'_, T> {
        let (Ok(rs @ 1..), Ok(cs @ 1..)) = (usize::try_from(matrix.rs), usize::try_from(matrix.cs))
        else {
            // SAFETY: Every element is in the matrix.
            let dense = copy.insert(FlexBuf::from_fn(rows * cols, |i| unsafe {
                matrix.get(i / cols, i % cols)
            }));
            return MatRef::row_major(rows, cols, dense.as_slice());
        };
        let len = (rows - 1) * rs + (cols - 1) * cs + 1;
        // SAFETY: The slice goes from the first element of the matrix to the last.
        let values = unsafe { core::slice::from_raw_parts(matrix.ptr, len) };
        MatRef::from_parts(rows, cols, values, rs, cs).expect("the slice ends at the last element")
    }

    /// [`product`] in f32, so that the sums do not round to f16.
    ///
    /// # Safety
    ///
    /// As [`gemm`].
    unsafe fn product_f16(
        [m, n, k]: [usize; 3],
        dst: Strided<*mut f16>,
        scale: Option<f16>,
        lhs: Strided<*const f16>,
        rhs: Strided<*const f16>,
        beta: f16,
    ) {
        let widen = |matrix: Strided<*const f16>, rows: usize, cols: usize| {
            // SAFETY: Every element is in the matrix.
            FlexBuf::from_fn(rows * cols, |i| unsafe {
                matrix.get(i / cols, i % cols).to_f32()
            })
        };
        let (lhs, rhs) = (widen(lhs, m, k), widen(rhs, k, n));
        let mut out = FlexBuf::<f32>::zeroed(m * n);
        let dense = |ptr, cols: usize| Strided {
            ptr,
            cs: 1,
            rs: cols.cast_signed(),
        };
        let sums = Strided {
            ptr: out.as_mut_ptr(),
            cs: 1,
            rs: n.cast_signed(),
        };
        // SAFETY: The three are dense, with these dimensions.
        unsafe {
            product(
                [m, n, k],
                sums,
                None,
                dense(lhs.as_ptr(), k),
                dense(rhs.as_ptr(), n),
                beta.to_f32(),
            );
        }
        for row in 0..m {
            for col in 0..n {
                // SAFETY: The element is in `dst`, and is read only when it is initialized.
                unsafe {
                    let at = dst.ptr.offset(dst.offset(row, col));
                    let kept = scale.map_or(0.0, |scale| scale.to_f32() * at.read().to_f32());
                    at.write(f16::from_f32(kept + out[row * n + col]));
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use alloc::{vec, vec::Vec};

        const NONE: Parallelism = Parallelism::None;

        /// Small integers, so that every kernel gives the same f32 sums.
        fn values(len: usize, seed: usize) -> Vec<f32> {
            (0..len)
                .map(|i| f32::from(u8::try_from((i * 7919 + seed) % 13).unwrap()) - 6.0)
                .collect()
        }

        fn expected(
            [m, n, k]: [usize; 3],
            lhs: impl Fn(usize, usize) -> f32,
            rhs: impl Fn(usize, usize) -> f32,
        ) -> Vec<f32> {
            (0..m * n)
                .map(|i| (0..k).map(|p| lhs(i / n, p) * rhs(p, i % n)).sum())
                .collect()
        }

        const SHAPES: [[usize; 3]; 12] = [
            [1, 1, 1],
            [1, 9, 5],
            [9, 1, 5],
            [3, 2, 7],
            [4, 4, 4],
            [5, 7, 3],
            [64, 2, 384],
            [128, 4, 387],
            [1, 512, 128],
            [17, 33, 129],
            [40, 70, 260],
            [2, 3, 0],
        ];

        #[test]
        fn overwrites_an_output_of_nan_in_every_shape() {
            for [m, n, k] in SHAPES {
                let (left, right) = (values(m * k, 1), values(k * n, 2));
                let mut out = vec![f32::NAN; m * n];
                unsafe {
                    gemm(
                        m,
                        n,
                        k,
                        out.as_mut_ptr(),
                        1,
                        n.cast_signed(),
                        false,
                        left.as_ptr(),
                        1,
                        k.cast_signed(),
                        right.as_ptr(),
                        1,
                        n.cast_signed(),
                        0.0,
                        1.0,
                        NONE,
                    );
                }
                let want = expected([m, n, k], |i, p| left[i * k + p], |p, j| right[p * n + j]);
                assert_eq!(out, want, "{m}x{n}x{k}");
            }
        }

        #[test]
        fn transposed_and_flipped_operands() {
            for [m, n, k] in SHAPES {
                if k == 0 {
                    continue;
                }
                // Column-major lhs, and an rhs with its rows and columns reversed.
                let (left, right) = (values(m * k, 3), values(k * n, 4));
                let mut out = vec![f32::NAN; m * n];
                unsafe {
                    gemm(
                        m,
                        n,
                        k,
                        out.as_mut_ptr(),
                        1,
                        n.cast_signed(),
                        false,
                        left.as_ptr(),
                        m.cast_signed(),
                        1,
                        right.as_ptr().add(k * n - 1),
                        -1,
                        -n.cast_signed(),
                        0.0,
                        1.0,
                        NONE,
                    );
                }
                let want = expected(
                    [m, n, k],
                    |i, p| left[p * m + i],
                    |p, j| right[(k - 1 - p) * n + (n - 1 - j)],
                );
                assert_eq!(out, want, "{m}x{n}x{k}");
            }
        }

        #[test]
        fn broadcast_operand() {
            let [m, n, k] = [6, 5, 7];
            let (left, right) = (values(k, 5), values(k * n, 6));
            let mut out = vec![f32::NAN; m * n];
            unsafe {
                gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr(),
                    1,
                    n.cast_signed(),
                    false,
                    left.as_ptr(),
                    1,
                    0,
                    right.as_ptr(),
                    1,
                    n.cast_signed(),
                    0.0,
                    1.0,
                    NONE,
                );
            }
            assert_eq!(
                out,
                expected([m, n, k], |_, p| left[p], |p, j| right[p * n + j])
            );
        }

        #[test]
        fn accumulates_with_alpha_and_beta() {
            for [m, n, k] in SHAPES {
                let (left, right) = (values(m * k, 1), values(k * n, 2));
                let before = values(m * n, 7);
                let mut out = before.clone();
                unsafe {
                    gemm(
                        m,
                        n,
                        k,
                        out.as_mut_ptr(),
                        1,
                        n.cast_signed(),
                        true,
                        left.as_ptr(),
                        1,
                        k.cast_signed(),
                        right.as_ptr(),
                        1,
                        n.cast_signed(),
                        3.0,
                        2.0,
                        NONE,
                    );
                }
                let want: Vec<f32> =
                    expected([m, n, k], |i, p| left[i * k + p], |p, j| right[p * n + j])
                        .iter()
                        .zip(&before)
                        .map(|(x, y)| 2.0 * x + 3.0 * y)
                        .collect();
                assert_eq!(out, want, "{m}x{n}x{k}");
            }
        }

        #[test]
        fn leaves_the_gaps_of_a_strided_output() {
            let [m, n, k] = [5, 6, 7];
            let (left, right) = (values(m * k, 1), values(k * n, 2));
            let mut out = vec![-99.0f32; m * (n + 2)];
            unsafe {
                gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr(),
                    1,
                    (n + 2).cast_signed(),
                    false,
                    left.as_ptr(),
                    1,
                    k.cast_signed(),
                    right.as_ptr(),
                    1,
                    n.cast_signed(),
                    0.0,
                    1.0,
                    NONE,
                );
            }
            let want = expected([m, n, k], |i, p| left[i * k + p], |p, j| right[p * n + j]);
            for (row, want) in out.chunks(n + 2).zip(want.chunks(n)) {
                assert_eq!(&row[..n], want);
                assert_eq!(&row[n..], [-99.0; 2]);
            }
        }

        #[test]
        fn f16_sums_in_f32() {
            // 4096 ones: an f16 sum stops at 2048.
            let k = 4096;
            let left = vec![f16::ONE; 2 * k];
            let mut out = vec![f16::NAN; 4];
            unsafe {
                gemm(
                    2,
                    2,
                    k,
                    out.as_mut_ptr(),
                    1,
                    2,
                    false,
                    left.as_ptr(),
                    1,
                    k.cast_signed(),
                    left.as_ptr(),
                    1,
                    2,
                    f16::ZERO,
                    f16::ONE,
                    NONE,
                );
            }
            assert_eq!(out, vec![f16::from_f32(4096.0); 4]);
        }

        #[test]
        fn f64_matches() {
            let [m, n, k] = [17, 33, 129];
            let (left, right) = (values(m * k, 1), values(k * n, 2));
            let want = expected([m, n, k], |i, p| left[i * k + p], |p, j| right[p * n + j]);
            let wide = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
            let (left, right) = (wide(&left), wide(&right));
            let mut out = vec![f64::NAN; m * n];
            unsafe {
                gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr(),
                    1,
                    n.cast_signed(),
                    false,
                    left.as_ptr(),
                    1,
                    k.cast_signed(),
                    right.as_ptr(),
                    1,
                    n.cast_signed(),
                    0.0,
                    1.0,
                    NONE,
                );
            }
            assert_eq!(out, wide(&want));
        }
    }
}
