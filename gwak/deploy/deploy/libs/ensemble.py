import torch
import logging
from typing import Callable, Optional, Tuple

from hermes.quiver.model import EnsembleModel, ExposedTensor


from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from hermes.quiver.model import EnsembleModel, ExposedTensor

from .whiten_utils import BackgroundSnapshotter, BatchWhitener

Tensor = torch.Tensor
def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def add_streaming_input_preprocessor(
    ensemble: "EnsembleModel",
    input: "ExposedTensor",
    background_batch_size: int,
    stride_batch_size: int,
    num_ifos: int,
    psd_length: float,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    fftlength: float,
    highpass: Optional[float] = None,
    preproc_instances: Optional[int] = None,
    streams_per_gpu: int = 1,
) -> "ExposedTensor":
    """Create a snapshotter model and add it to the repository"""

    # _, _, *kernel_size = input.shape # batch_size, num_ifos, *kernel_size
    # # batch_size, *kernel_size, num_ifos  = input.shape # Apply for gwak1
    background_batch_size
    # * stride_batch_size
    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    stride = int(sample_rate / inference_sampling_rate)
    state_shape = (background_batch_size, num_ifos, snapshotter.state_size)
    input_shape = (background_batch_size, num_ifos, stride_batch_size * stride)
    logging.info(f"Snappshot kerenl shape: ")
    logging.info(f"    Batch Size: {state_shape[0]}")
    logging.info(f"    Nums Ifo: {state_shape[1]}")
    logging.info(f"    Input Update: {input_shape[-1]}: ({input_shape[-1]/sample_rate} Sec)")
    logging.info(f"    State Size: {state_shape[-1]}: ({state_shape[-1]/sample_rate} Sec)")
    kernel_value = state_shape[-1] + input_shape[-1]
    logging.info(f"    Sample Kernel: {kernel_value}: ({kernel_value/sample_rate} Sec)")
    # state_shape = (background_batch_size, snapshotter.state_size, num_ifos) # Apply for gwak1

    streaming_model = streaming_utils.add_streaming_model(
        ensemble.repository,
        streaming_layer=snapshotter,
        name="snapshotter",
        input_name="stream",
        input_shape=input_shape,
        state_names=["snapshot"],
        state_shapes=[state_shape],
        output_names=["strain"],
        streams_per_gpu=streams_per_gpu,
    )
    ensemble.add_input(streaming_model.inputs["stream"])

    preprocessor = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=stride_batch_size,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
    )
    preproc_model = ensemble.repository.add(
        "preprocessor", platform=Platform.TORCHSCRIPT
    )
    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if preproc_instances is not None:
        scale_model(preproc_model, preproc_instances)

    input_shape = streaming_model.outputs["strain"].shape
    preproc_model.export_version(
        preprocessor,
        input_shapes={"strain": input_shape},
        output_names=["whitened"],
    )
    ensemble.pipe(
        streaming_model.outputs["strain"],
        preproc_model.inputs["strain"],
    )
    return preproc_model.outputs["whitened"]