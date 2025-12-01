use burn_tensor::ops::TransactionOps;

use crate::{FloatTchElement, IntTchElement, LibTorch, TchElement};

impl<E: TchElement, F: FloatTchElement, I: IntTchElement> TransactionOps<Self> for LibTorch<E, F,I> {}
