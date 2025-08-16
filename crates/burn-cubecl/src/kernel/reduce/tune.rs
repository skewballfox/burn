#![allow(missing_docs)]

use burn_tensor::ElementConversion;
use cubecl::{
    client::ComputeClient,
    reduce::{ReduceFamily, tune_key::ReduceAutotuneKey},
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};

use crate::{
    CubeAutotuneKey, CubeElement, CubeRuntime, CubeTuneId, kernel::prng::random_like_uniform,
    tensor::CubeTensor,
};

use super::SumAutotuneKey;

/// Executes autotune on reduce operations.
pub fn autotune_reduce<
    Run: CubeRuntime,
    In: CubeElement,
    Out: CubeElement,
    Acc: CubeElement,
    Rd: cubecl::reduce::ReduceFamily,
>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: CubeTensor<Run>,
    output: CubeTensor<Run>,
    dim: usize,
    config: Rd::Config,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, CubeTuneId> = local_tuner!("reduce-dim");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<Run, Acc, Rd>, reduce_input_gen::<Run, Rd>)
            .with(Tunable::new(reduce::<Run, In, Out, Acc, Rd>))
            .with(Tunable::new(reduce_shared::<Run, In, Out, Acc, Rd>))
            .with(Tunable::new(reduce_plane::<Run, In, Out, Acc, Rd>))
            .with(Tunable::new(reduce_shared_plane::<Run, In, Out, Acc, Rd>))
    });

    TUNER.execute(
        &CubeTuneId::new::<Run>(&input.client, &input.device),
        client,
        tunables,
        (input, output, dim, config),
    );
}

pub(crate) fn create_key<Run: CubeRuntime, Acc: CubeElement, Rd: ReduceFamily>(
    input: &CubeTensor<Run>,
    output: &CubeTensor<Run>,
    axis: &usize,
    _config: &Rd::Config,
) -> ReduceAutotuneKey {
    let elem_input = input.dtype.into();
    let elem_output = output.dtype.into();
    let elem_acc = Acc::dtype().into();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        &input.shape.dims,
        input.strides[*axis] == 1,
        *axis,
    )
}

mod reduce_ops {
    #![allow(missing_docs)]

    use cubecl::reduce::ReduceFamily;

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime, Rd: ReduceFamily>(
        _key: &ReduceAutotuneKey,
        input: &CubeTensor<Run>,
        output: &CubeTensor<Run>,
        dim: &usize,
        config: &Rd::Config,
    ) -> (CubeTensor<Run>, CubeTensor<Run>, usize, Rd::Config) {
        (input.clone(), output.copy(), *dim, *config)
    }

    pub(crate) fn reduce<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Acc: CubeElement,
        Rd: cubecl::reduce::ReduceFamily,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, (In, Acc), Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: false,
            }),
            config,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Acc: CubeElement,
        Rd: cubecl::reduce::ReduceFamily,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, (In, Acc), Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: false,
            }),
            config,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_plane<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Acc: CubeElement,
        Rd: cubecl::reduce::ReduceFamily,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, (In, Acc), Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: true,
            }),
            config,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared_plane<
        Run: CubeRuntime,
        In: CubeElement,
        Out: CubeElement,
        Acc: CubeElement,
        Rd: cubecl::reduce::ReduceFamily,
    >(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, (In, Acc), Out, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: true,
            }),
            config,
        )
        .map_err(|e| format!("{e}"))
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
pub fn autotune_sum<Run: CubeRuntime, E: CubeElement>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: CubeTensor<Run>,
) -> CubeTensor<Run> {
    use sum_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!("autotune-sum");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key_sum::<Run>, sum_input_gen::<Run, E>)
            .with(Tunable::new(sum_chained::<Run, E>))
            .with(Tunable::new(sum_one_shot::<Run, E, 1>))
            .with(Tunable::new(sum_one_shot::<Run, E, 2>))
            .with(Tunable::new(sum_one_shot::<Run, E, 4>))
            .with(Tunable::new(sum_one_shot::<Run, E, 8>))
            .with(Tunable::new(sum_one_shot::<Run, E, 16>))
            .with(Tunable::new(sum_one_shot::<Run, E, 32>))
            .with(Tunable::new(sum_one_shot::<Run, E, 64>))
    });

    TUNER.execute(
        &CubeTuneId::new::<Run>(&input.client, &input.device),
        client,
        tunables,
        input,
    )
}

pub(crate) fn create_key_sum<Run: CubeRuntime>(input: &CubeTensor<Run>) -> CubeAutotuneKey {
    CubeAutotuneKey::Sum(SumAutotuneKey::generate(input))
}

impl SumAutotuneKey {
    #[allow(unused)]
    pub(crate) fn generate<Run: CubeRuntime>(input: &CubeTensor<Run>) -> Self {
        let dtype = input.dtype;
        let length = input.shape.num_elements();
        Self { dtype, length }
    }
}
mod sum_ops {
    #![allow(missing_docs)]
    use super::*;

    pub(crate) fn sum_input_gen<Run: CubeRuntime, E: CubeElement>(
        _key: &CubeAutotuneKey,
        input: &CubeTensor<Run>,
    ) -> CubeTensor<Run> {
        let random_bounds: (E, E) = ((-10.0_f32).elem::<E>(), (10.0_f32).elem::<E>());
        random_like_uniform(input, random_bounds.0, random_bounds.1)
    }

    pub(crate) fn sum_one_shot<Run: CubeRuntime, E: CubeElement, const C: u32>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let handle = client.create(E::as_bytes(&[E::from_int(0)]));
        let output = CubeTensor::new_contiguous(client, device, [1].into(), handle, E::dtype());

        cubecl::reduce::shared_sum::<Run, E>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            C,
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }

    #[cfg(feature = "autotune")]
    pub(crate) fn sum_chained<Run: CubeRuntime, E: CubeElement>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        crate::kernel::reduce::reduce::<Run, E, E, E>(
            input,
            crate::kernel::reduce::ReduceStrategy::Autotune,
            cubecl::reduce::instructions::ReduceFnConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}
