import torch
import logging

from pathlib import Path
from typing import Callable, Optional

import hermes.quiver as qv

from deploy.libs import gwak_logger
from deploy.libs import scale_model, add_streaming_input_preprocessor


def export(
    project: Path,
    model_dir: Path,
    output_dir: Path,
    clean: bool,
    background_batch_size: int, 
    stride_batch_size: int, 
    num_ifos: int, 
    gwak_instances: int, 
    psd_length: float,
    kernel_length: float,
    fduration: float,
    fftlength: int,
    inference_sampling_rate: int,
    sample_rate: int,
    preproc_instances: int,
    # highpass: Optional[float] = None,
    # streams_per_gpu: int,
    platform: qv.Platform = qv.Platform.ONNX,
    **kwargs,
):
    
    weights = model_dir / project / "model_JIT.pt"
    output_dir = output_dir / project
    batch_size = background_batch_size * stride_batch_size
    kernel_size = int(kernel_length * sample_rate)

    with open(weights, "rb") as f:
        graph = torch.jit.load(f)

    graph.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = qv.ModelRepository(output_dir, clean=clean)

    gwak_logger(output_dir / "export.log")
    try:
        gwak = repo.models[f"gwak-{project}"]
    except KeyError:
        gwak = repo.add(f"gwak-{project}", platform)

    if gwak_instances is not None:
        scale_model(gwak, gwak_instances)

    # input_shape = (batch_size, kernel_size, num_ifos) # Apply this for gwak_1
    input_shape = (batch_size, num_ifos, kernel_size) 
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        gwak.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = False

    logging.info(f"Export trained model with {platform} format")
    logging.info(f"GWAK Model iuput shape:")
    logging.info(f"    Batch size: {input_shape[0]}")
    logging.info(f"    Nums Ifo: {input_shape[1]}")
    logging.info(f"    Sample Kernel: {input_shape[-1]}")

    gwak.export_version(
        graph,
        input_shapes={"INPUT__0": input_shape}, 
        output_names=["OUTPUT__0"],
        **kwargs,
    )

    ensemble_name = f"gwak-{project}-streamer"

    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

        logging.info(f"Adding snappershotter and whitener.")
        whitened = add_streaming_input_preprocessor(
            ensemble,
            gwak.inputs["INPUT__0"],
            background_batch_size=background_batch_size,
            stride_batch_size=stride_batch_size,
            num_ifos=num_ifos,
            psd_length=psd_length,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            fftlength=fftlength,
            preproc_instances=preproc_instances,
        )

        logging.info(f"Ensemble model.")
        ensemble.pipe(whitened, gwak.inputs["INPUT__0"])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(gwak.outputs["OUTPUT__0"])
        ensemble.export_version(None)

    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has gwak
        # and the snapshotter as a part of its models
        if gwak not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'gwak'".format(ensemble_name)
            )
        # TODO: checks for snapshotter and preprocessor

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()