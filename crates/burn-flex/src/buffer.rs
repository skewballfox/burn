//! Typed buffers for tensor storage and kernel scratch.
//!
//! Every buffer that Flex creates comes from [`FlexBuf`], so the choice of allocator is made in
//! one place: the registered allocator with the `buffer-allocator` feature, the global allocator
//! otherwise.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use burn_std::Bytes;

/// Allocates `size` bytes aligned to `align`, with a length of zero.
///
/// # Panics
///
/// When the allocator has no memory for `size` bytes.
fn alloc_bytes(size: usize, align: usize) -> Bytes {
    #[cfg(feature = "buffer-allocator")]
    if let Some(allocator) = crate::allocator::allocator() {
        return Bytes::with_capacity_in(allocator, size, align).unwrap_or_else(|err| {
            panic!("burn-flex: the registered allocator has no memory for {size} bytes: {err}")
        });
    }
    Bytes::with_capacity(size, align)
        .unwrap_or_else(|err| panic!("burn-flex: cannot allocate {size} bytes: {err}"))
}

/// Moves `bytes` into the registered allocator when they are in the global heap.
///
/// Shared, static, file and allocator memory stays where it is: weights in flash stay zero-copy.
#[cfg(feature = "buffer-allocator")]
pub(crate) fn into_registered(bytes: Bytes) -> Bytes {
    if crate::allocator::allocator().is_none()
        || bytes.property() != burn_std::AllocationProperty::Native
    {
        return bytes;
    }
    let mut copy = alloc_bytes(bytes.len(), bytes.align());
    copy.spare_capacity_mut()
        .iter_mut()
        .zip(bytes.iter())
        .for_each(|(slot, &byte)| *slot = MaybeUninit::new(byte));
    // SAFETY: The `bytes.len()` bytes were written above.
    unsafe { copy.set_len(bytes.len()) };
    copy
}

/// A growable buffer of `E`, like `Vec<E>`, whose memory comes from the Flex allocator.
///
/// Build the data with [`push`](Self::push), [`extend_from_slice`](Self::extend_from_slice), or
/// writes to [`spare_capacity_mut`](Self::spare_capacity_mut) followed by
/// [`set_len`](Self::set_len). Then [`into_bytes`](Self::into_bytes) gives the storage of a
/// tensor without a copy.
pub struct FlexBuf<E> {
    bytes: Bytes,
    /// The start of the memory of `bytes`. It stays valid while `bytes` does not grow.
    ptr: NonNull<E>,
    len: usize,
    capacity: usize,
    _elem: PhantomData<E>,
}

// SAFETY: `FlexBuf` owns its memory, as a `Vec<E>` does.
unsafe impl<E: Send> Send for FlexBuf<E> {}
// SAFETY: Shared access only reads the elements.
unsafe impl<E: Sync> Sync for FlexBuf<E> {}

impl<E: bytemuck::Pod + Send + Sync> FlexBuf<E> {
    /// An empty buffer with memory for `capacity` elements.
    ///
    /// # Panics
    ///
    /// When the allocator has no memory for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let size = capacity
            .checked_mul(size_of::<E>())
            .expect("burn-flex: buffer size overflows usize");
        let mut bytes = alloc_bytes(size, align_of::<E>());
        let ptr = if size == 0 {
            NonNull::dangling()
        } else {
            NonNull::new(bytes.spare_capacity_mut().as_mut_ptr().cast::<E>())
                .expect("allocated memory is not null")
        };
        debug_assert!(
            ptr.is_aligned(),
            "the memory is not aligned for the element"
        );
        Self {
            bytes,
            ptr,
            len: 0,
            capacity,
            _elem: PhantomData,
        }
    }

    /// `n` copies of `value`, like `vec![value; n]`.
    pub fn from_elem(value: E, n: usize) -> Self {
        if bytemuck::bytes_of(&value).iter().all(|&byte| byte == 0) {
            return Self::zeroed(n);
        }
        Self::filled(value, n)
    }

    fn filled(value: E, n: usize) -> Self {
        let mut buf = Self::with_capacity(n);
        buf.spare_capacity_mut()
            .iter_mut()
            .for_each(|slot| *slot = MaybeUninit::new(value));
        // SAFETY: The `n` elements were written above.
        unsafe { buf.set_len(n) };
        buf
    }

    /// `n` zero elements.
    pub fn zeroed(n: usize) -> Self {
        #[cfg(feature = "buffer-allocator")]
        if crate::allocator::allocator().is_some() {
            return Self::filled(E::zeroed(), n);
        }
        // `vec!` asks the global allocator for zeroed memory, which can skip the writes.
        let mut bytes = Bytes::from_elems(alloc::vec![E::zeroed(); n]);
        let ptr = NonNull::new(bytes.as_mut_ptr().cast::<E>()).expect("memory is not null");
        Self {
            bytes,
            ptr,
            len: n,
            capacity: n,
            _elem: PhantomData,
        }
    }

    /// A copy of `data`.
    pub fn from_slice(data: &[E]) -> Self {
        let mut buf = Self::with_capacity(data.len());
        buf.extend_from_slice(data);
        buf
    }

    /// `f(0)`, `f(1)`, ..., `f(n - 1)`.
    pub fn from_fn(n: usize, mut f: impl FnMut(usize) -> E) -> Self {
        let mut buf = Self::with_capacity(n);
        buf.spare_capacity_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = MaybeUninit::new(f(i)));
        // SAFETY: The `n` elements were written above.
        unsafe { buf.set_len(n) };
        buf
    }

    /// The elements.
    #[inline]
    pub fn as_slice(&self) -> &[E] {
        self
    }

    /// The elements, mutably.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [E] {
        self
    }

    /// The number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of elements the buffer holds without growing.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The memory after the elements, up to the capacity.
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<E>] {
        // SAFETY: The memory holds `capacity` elements, and `len <= capacity`.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr().add(self.len).cast(),
                self.capacity - self.len,
            )
        }
    }

    /// Sets the number of elements.
    ///
    /// # Safety
    ///
    /// `len` must not be larger than the capacity, and the first `len` elements must be
    /// initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(
            len <= self.capacity,
            "the length is larger than the capacity"
        );
        self.len = len;
    }

    /// Makes room for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let required = self
            .len
            .checked_add(additional)
            .expect("burn-flex: buffer size overflows usize");
        if required > self.capacity {
            self.grow_to(required.max(self.capacity * 2).max(4));
        }
    }

    #[cold]
    fn grow_to(&mut self, capacity: usize) {
        let mut grown = Self::with_capacity(capacity);
        grown.extend_from_slice(self);
        *self = grown;
    }

    /// Appends `value`.
    #[inline]
    pub fn push(&mut self, value: E) {
        if self.len == self.capacity {
            self.reserve(1);
        }
        // SAFETY: `len < capacity`, so the slot is in the memory.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Appends a copy of `data`.
    pub fn extend_from_slice(&mut self, data: &[E]) {
        self.reserve(data.len());
        let spare = &mut self.spare_capacity_mut()[..data.len()];
        spare
            .iter_mut()
            .zip(data)
            .for_each(|(slot, &value)| *slot = MaybeUninit::new(value));
        self.len += data.len();
    }

    /// Shortens the buffer to `len` elements. Does nothing when it is already shorter.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The elements as tensor storage, without a copy.
    pub fn into_bytes(self) -> Bytes {
        let mut bytes = self.bytes;
        // SAFETY: The first `len` elements are initialized, and the memory holds them.
        unsafe { bytes.set_len(self.len * size_of::<E>()) };
        bytes
    }
}

#[cfg(feature = "rayon")]
impl<E: bytemuck::Pod + Send + Sync> FlexBuf<E> {
    /// The elements of an indexed parallel iterator, in order, like `collect::<Vec<_>>()`.
    pub fn par_collect(iter: impl rayon::iter::IndexedParallelIterator<Item = E>) -> Self {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

        let len = iter.len();
        let mut buf = Self::with_capacity(len);
        buf.spare_capacity_mut()[..len]
            .par_iter_mut()
            .zip(iter)
            .for_each(|(slot, value)| *slot = MaybeUninit::new(value));
        // SAFETY: The zip wrote all `len` elements: both sides have `len` of them.
        unsafe { buf.set_len(len) };
        buf
    }
}

/// `(0..n).map(f).unzip()` into two buffers, in parallel when `parallel` and the `rayon`
/// feature are on.
pub(crate) fn unzip_from_fn<A, B, F>(n: usize, parallel: bool, f: F) -> (FlexBuf<A>, FlexBuf<B>)
where
    A: bytemuck::Pod + Send + Sync,
    B: bytemuck::Pod + Send + Sync,
    F: Fn(usize) -> (A, B) + Send + Sync,
{
    let mut first = FlexBuf::with_capacity(n);
    let mut second = FlexBuf::with_capacity(n);
    let write = |(i, (a, b)): (usize, (&mut MaybeUninit<A>, &mut MaybeUninit<B>))| {
        let (value_a, value_b) = f(i);
        *a = MaybeUninit::new(value_a);
        *b = MaybeUninit::new(value_b);
    };
    #[cfg(feature = "rayon")]
    if parallel {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
        first.spare_capacity_mut()[..n]
            .par_iter_mut()
            .zip(second.spare_capacity_mut()[..n].par_iter_mut())
            .enumerate()
            .for_each(write);
        // SAFETY: Every one of the `n` pairs was written above.
        unsafe {
            first.set_len(n);
            second.set_len(n);
        }
        return (first, second);
    }
    #[cfg(not(feature = "rayon"))]
    let _ = parallel;
    first.spare_capacity_mut()[..n]
        .iter_mut()
        .zip(&mut second.spare_capacity_mut()[..n])
        .enumerate()
        .for_each(write);
    // SAFETY: Every one of the `n` pairs was written above.
    unsafe {
        first.set_len(n);
        second.set_len(n);
    }
    (first, second)
}

/// `collect` into a [`FlexBuf`] for indexed parallel iterators.
#[cfg(feature = "rayon")]
pub(crate) trait ParCollectBuf: rayon::iter::IndexedParallelIterator
where
    Self::Item: bytemuck::Pod + Send + Sync,
{
    /// The elements, in order, like `collect::<Vec<_>>()`.
    fn collect_buf(self) -> FlexBuf<Self::Item> {
        FlexBuf::par_collect(self)
    }
}

#[cfg(feature = "rayon")]
impl<I> ParCollectBuf for I
where
    I: rayon::iter::IndexedParallelIterator,
    I::Item: bytemuck::Pod + Send + Sync,
{
}

impl<E> Deref for FlexBuf<E> {
    type Target = [E];

    #[inline]
    fn deref(&self) -> &[E] {
        // SAFETY: The first `len` elements are initialized.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<E> DerefMut for FlexBuf<E> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [E] {
        // SAFETY: The first `len` elements are initialized, and `self` is borrowed mutably.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<E: bytemuck::Pod + Send + Sync> Extend<E> for FlexBuf<E> {
    fn extend<I: IntoIterator<Item = E>>(&mut self, iter: I) {
        let mut iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        self.reserve(lower);
        if upper == Some(lower) {
            // An exact size: a loop without capacity checks, which the compiler vectorizes as
            // it does `Vec`'s `collect`.
            let written = self.spare_capacity_mut()[..lower]
                .iter_mut()
                .zip(iter)
                .map(|(slot, value)| *slot = MaybeUninit::new(value))
                .count();
            self.len += written;
        } else {
            let written = self.spare_capacity_mut()[..lower]
                .iter_mut()
                .zip(iter.by_ref())
                .map(|(slot, value)| *slot = MaybeUninit::new(value))
                .count();
            self.len += written;
            iter.for_each(|value| self.push(value));
        }
    }
}

impl<E: bytemuck::Pod + Send + Sync> FromIterator<E> for FlexBuf<E> {
    fn from_iter<I: IntoIterator<Item = E>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut buf = Self::with_capacity(iter.size_hint().0);
        buf.extend(iter);
        buf
    }
}

impl<'a, E> IntoIterator for &'a FlexBuf<E> {
    type Item = &'a E;
    type IntoIter = core::slice::Iter<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, E> IntoIterator for &'a mut FlexBuf<E> {
    type Item = &'a mut E;
    type IntoIter = core::slice::IterMut<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<E: PartialEq> PartialEq for FlexBuf<E> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<E: PartialEq> PartialEq<[E]> for FlexBuf<E> {
    fn eq(&self, other: &[E]) -> bool {
        **self == *other
    }
}

impl<E: PartialEq, const N: usize> PartialEq<[E; N]> for FlexBuf<E> {
    fn eq(&self, other: &[E; N]) -> bool {
        **self == other[..]
    }
}

impl<E: PartialEq> PartialEq<alloc::vec::Vec<E>> for FlexBuf<E> {
    fn eq(&self, other: &alloc::vec::Vec<E>) -> bool {
        **self == other[..]
    }
}

impl<E: PartialEq> PartialEq<FlexBuf<E>> for alloc::vec::Vec<E> {
    fn eq(&self, other: &FlexBuf<E>) -> bool {
        self[..] == **other
    }
}

impl<E: core::fmt::Debug> core::fmt::Debug for FlexBuf<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_past_its_capacity_and_keeps_the_elements() {
        let mut buf = FlexBuf::<u32>::with_capacity(2);
        (0..10).for_each(|i| buf.push(i));
        assert_eq!(&*buf, &(0..10).collect::<alloc::vec::Vec<_>>()[..]);
    }

    #[test]
    fn into_bytes_holds_exactly_the_elements() {
        let mut buf = FlexBuf::<f32>::with_capacity(8);
        buf.extend_from_slice(&[1.0, 2.0, 3.0]);
        let bytes = buf.into_bytes();
        assert_eq!(bytes.len(), 3 * size_of::<f32>());
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&bytes), &[1.0, 2.0, 3.0]);
    }
}
