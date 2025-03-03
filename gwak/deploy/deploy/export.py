import torch
import logging

from pathlib import Path
from typing import Callable, Optional, List

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
    model_files: List[str],
    combiner_model_file: str,
    platform: qv.Platform = qv.Platform.ONNX,
    **kwargs,
):

    output_dir = output_dir / project
    batch_size = background_batch_size * stride_batch_size
    kernel_size = int(kernel_length * sample_rate)
    input_shape = (batch_size, num_ifos, kernel_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    repo = qv.ModelRepository(output_dir, clean=clean)

    gwak_logger(output_dir / "export.log")
    logging.info(f"Exporting {len(model_files)} individual models for project {project}.")

    # loop over each model and load
    individual_models = []
    for idx, model_file in enumerate(model_files):
        weights = model_dir / project / model_file
        with open(weights, "rb") as f:
            graph = torch.jit.load(f)
        graph.eval()

        # IMPORTANT: create a unique name for each model.
        model_name = f"model_{idx}-{project}"
        try:
            model_instance = repo.models[model_name]
        except KeyError:
            model_instance = repo.add(model_name, platform)

        if gwak_instances is not None:
            scale_model(model_instance, gwak_instances)
        local_kwargs = {}
        if platform == qv.Platform.ONNX:
            local_kwargs["opset_version"] = 13
            # turn off graph optimization because of this error
            # https://github.com/triton-inference-server/server/issues/3418
            model_instance.config.optimization.graph.level = -1 
        elif platform == qv.Platform.TENSORRT:
            local_kwargs["use_fp16"] = False

        logging.info(f"Exporting model '{model_name}' with input shape {input_shape}.")
        model_instance.export_version(
            graph,
            input_shapes={"INPUT__0": input_shape},
            output_names=["OUTPUT__0"],
            **local_kwargs,
        )
        individual_models.append(model_instance)

    ensemble_name = f"ensemble-{project}"
    try:
        ensemble = repo.models[ensemble_name]
    except KeyError:
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

    logging.info(f"Adding snappershotter and whitener.")
    whitened = add_streaming_input_preprocessor(
        ensemble,
        individual_models[0].inputs["INPUT__0"],
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

    # Pipe the preprocessed input to each individual model.
    for model in individual_models:
        ensemble.pipe(whitened, model.inputs["INPUT__0"])

    # Load and add the combiner model.
    combiner_weights = model_dir / project / combiner_model_file
    with open(combiner_weights, "rb") as f:
        combiner_graph = torch.jit.load(f)
    combiner_graph.eval()

    combiner_name = f"combiner-{project}"
    try:
        combiner = repo.models[combiner_name]
    except KeyError:
        combiner = repo.add(combiner_name, platform)

    combiner_input_shapes = {
        f"INPUT__{idx}": input_shape for idx in range(len(individual_models))
    }
    
    logging.info(f"Exporting combiner model '{combiner_name}' with input shapes: {combiner_input_shapes}.")
    combiner.export_version(
        combiner_graph,
        input_shapes=combiner_input_shapes,
        output_names=["OUTPUT__0"],
        **local_kwargs,
    )

    # Pipe each individual model's output into the combiner.
    for idx, model in enumerate(individual_models):
        ensemble.pipe(model.outputs["OUTPUT__0"], combiner.inputs[f"INPUT__{idx}"])

    # Set the ensemble's final output from the combiner.
    ensemble.add_output(combiner.outputs["OUTPUT__0"])
    ensemble.export_version(None)

    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(6e10)
    snapshotter.config.write()