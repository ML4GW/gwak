import h5py
import yaml
import logging
import numpy as np

from typing import Optional

import matplotlib.pyplot as plt
from deploy.libs import gwak_logger
from deploy.libs.infer_utils import noise_runs_list
from deploy.libs.trigger_io import (
    unpack_timeslide, 
    select_threshold, 
    find_outlier_by_segmets
)
from deploy.libs import gwak_output_dir, gwak_louvre_dir, gwak_logging_dir


def threshold_lock(
    cl_config: str, 
    fm_config: str,
    ifo_mode: str, 
    run_name: str,
    infer_sample_rate: int,
    psd_length: float,
    threshold_level: float,
    **kwargs
):

    if run_name not in noise_runs_list:
        print(f"Warning! Run name {run_name} not in {noise_runs_list}")

    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    log_dir = gwak_logging_dir(suffix=f"infer/{model}/{run_name}")()
    log_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_dir / "threshold_lock.log")

    tslide_data_dir = gwak_output_dir(
        suffix=f"infer/{model}/{run_name}/inference_result"
    )()
    threshold_file = gwak_output_dir(suffix=f"infer/{model}")(
        append_path="threshold.h5"
    )

    # Main operation
    tslide_dict, tslide_data = unpack_timeslide(
        psd_length=psd_length,
        infer_sample_rate=infer_sample_rate,
        tslide_data_dir=tslide_data_dir
    )

    threshold = select_threshold(
        tslide_dict=tslide_dict,
        tslide_data=tslide_data,
        threshold_level=threshold_level,
        run_name=run_name,
        model=model
    )

    # Save threshold data
    with h5py.File(threshold_file, "a") as h:

        h.pop(run_name, None)
        h.create_dataset(run_name, data=threshold)

    logging.info(f"Therehold generated at: {threshold_file}.")

    # # Plotting
    # if threshold_level >= 1: 
    #     hist_label = f"Max Outlier: {np.min(tslide_data):.2f} \
    #     \n{ordinal(threshold_level)} Outlier: {threshold:.2f}"

    # elif threshold_level < 1: 
    #     hist_label = f"Max Outlier: {np.min(tslide_data):.2f} \
    #     \n{threshold_level*100}% Outlier: {threshold:.2f}"

    # # Plot Timeslide outputs
    # plt.title(f"{run_name.capitalize()} \n{model} \nTimeslide Output distribution")
    # plt.hist(tslide_data, bins=100, zorder=2, label=hist_label,)
    # plt.legend()
    # plt.axvline(threshold, color="red")
    # plt.grid(zorder=0)
    # plt.yscale("log")
    # plt.xlabel("Metric")
    # plt.ylabel("Data Counts")
    # plt.savefig(model_louvre_dir/"TS_ana.png", dpi=300, bbox_inches='tight')
    # plt.close()


def scan_outlier(
    cl_config: str,
    fm_config: str,
    ifo_mode: str,
    run_name: str,
    threshold_setting: str,
    infer_sample_rate: int,
    psd_length: float,
    accumlation_length:float,
    pad:float,
    threshold_value: Optional[float]=None,
    **kwargs
):

    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    stream_cut = int(infer_sample_rate*psd_length)

    # Initialize file paths
    log_dir = gwak_logging_dir(
        suffix=f"{model}/{run_name}_{threshold_setting}"
    )()
    log_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_dir / "scan_outlier.log")

    threshold_file = gwak_output_dir(suffix=f"infer/{model}")(
        append_path="threshold.h5"
    )
    tslide_data_dir = gwak_output_dir(
        suffix=f"infer/{model}/{run_name}/inference_result"
    )()
    outlier_file = gwak_output_dir(
        suffix=f"infer/{model}/{run_name}"
    )(append_path="outlier_config.h5")

    # Determine threshold
    if threshold_value is not None:
        threshold = threshold_value
    else:
        logging.info(f"{threshold_file}")
        with h5py.File(threshold_file, "r") as h5:
            threshold = float(h5[f"{threshold_setting}"][()])
    
    # Main operation
    tslide_dict, tslide_data = unpack_timeslide(
        infer_sample_rate=infer_sample_rate,
        psd_length=psd_length,
        tslide_data_dir=tslide_data_dir,
    )

    outlier_dict = find_outlier_by_segmets(
        tslide_dict=tslide_dict,
        threshold=threshold,
        infer_sample_rate=infer_sample_rate,
        psd_length=psd_length,
        accumlation_length=accumlation_length,
        pad=pad,
    )

    # Data saving
    logging_list = []
    with h5py.File(outlier_file, "w") as h:
        for key, item in outlier_dict.items():
            logging_list.append(key)
            h.create_dataset(
                key, 
                data=np.concatenate(outlier_dict[key])
            )
    logging.info(f" ")
    logging.info(f"Outlier information saved at: {outlier_file}.")
    # logging.info(f"Contained infomation includes:")
    # for key in logging_list:
    #     logging.info(f"{key}")
    logging.info(f" ")
