use core::marker::PhantomData;

use super::{PrecisionSettings, Record};
use burn_tensor::{Bool, DType, Element, Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};

use alloc::format;

/// Deserialize the value into [`TensorData`].
fn deserialize_data<'de, E, De>(deserializer: De) -> Result<TensorData, De::Error>
where
    E: Element + Deserialize<'de>,
    De: serde::Deserializer<'de>,
{
    let data = TensorData::deserialize(deserializer).map_err(|e| {
        serde::de::Error::custom(format!(
            "{e:?}\nThe internal data format has changed since version 0.14.0. If you are trying to load a record saved in a previous version, use the `record-backward-compat` feature flag with a previous version (<=0.16.0). Once you have saved the record in the new format, you can upgrade back to the current version.\n"
        ))
    })?;
    let data = if let DType::QFloat(_) = data.dtype {
        data // do not convert quantized tensors
    } else {
        data.convert::<E>()
    };
    Ok(data)
}

/// This struct implements serde to lazily serialize and deserialize a float tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct FloatTensorSerde<S: PrecisionSettings> {
    data: TensorData,
    _e: PhantomData<S::FloatElem>,
}

/// This struct implements serde to lazily serialize and deserialize an int tensor
/// using the given [record settings](RecordSettings).
#[derive(new, Clone, Debug)]
pub struct IntTensorSerde<S: PrecisionSettings> {
    data: TensorData,
    _e: PhantomData<S::IntElem>,
}

/// This struct implements serde to lazily serialize and deserialize an bool tensor.
#[derive(new, Clone, Debug)]
pub struct BoolTensorSerde {
    data: TensorData,
}

// --- SERDE IMPLEMENTATIONS --- //

impl<S: PrecisionSettings> Serialize for FloatTensorSerde<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, S: PrecisionSettings> Deserialize<'de> for FloatTensorSerde<S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = deserialize_data::<S::FloatElem, De>(deserializer)?;

        Ok(Self::new(data))
    }
}

impl<S: PrecisionSettings> Serialize for IntTensorSerde<S> {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, S: PrecisionSettings> Deserialize<'de> for IntTensorSerde<S> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = deserialize_data::<S::IntElem, De>(deserializer)?;

        Ok(Self::new(data))
    }
}

impl Serialize for BoolTensorSerde {
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BoolTensorSerde {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = deserialize_data::<bool, De>(deserializer)?;

        Ok(Self::new(data))
    }
}

// --- RECORD IMPLEMENTATIONS --- //

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D> {
    type Item<S: PrecisionSettings> = FloatTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        let data = self.into_data();
        let data = if let DType::QFloat(_) = data.dtype {
            data // do not convert quantized tensors
        } else {
            data.convert::<S::FloatElem>()
        };
        FloatTensorSerde::new(data)
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        let data = if let DType::QFloat(_) = item.data.dtype {
            item.data // do not convert quantized tensors
        } else {
            item.data.convert::<B::FloatElem>()
        };
        Tensor::from_data(data, device)
    }
}

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Int> {
    type Item<S: PrecisionSettings> = IntTensorSerde<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        IntTensorSerde::new(self.into_data().convert::<S::IntElem>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data.convert::<B::IntElem>(), device)
    }
}

impl<B: Backend, const D: usize> Record<B> for Tensor<B, D, Bool> {
    type Item<S: PrecisionSettings> = BoolTensorSerde;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        BoolTensorSerde::new(self.into_data())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Tensor::from_data(item.data, device)
    }
}
