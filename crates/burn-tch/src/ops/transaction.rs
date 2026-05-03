use burn_backend::ops::TransactionOps;

use crate::{FloatTchElement, LibTorch};

impl<E: FloatTchElement> TransactionOps<Self> for LibTorch<E> {}
