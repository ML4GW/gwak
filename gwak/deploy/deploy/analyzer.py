import h5py
import yaml
import logging
import numpy as np

from typing import Optional

import matplotlib.pyplot as plt
from pathlib import Path
from deploy.libs import gwak_logger
from deploy.libs.infer_utils import noise_runs_list
from deploy.libs.trigger_io import (
    lovure_file_handler,
    unpack_timeslide,
    select_threshold,
    find_outlier_by_segmets,
    resolve_oulier_config
)
from deploy.libs import gwak_output_dir, gwak_louvre_dir, gwak_logging_dir, ordinal
from deploy.libs.analysis_utils import get_bbc_inj_names, bbc_inj_info, find_valid_triggers


def threshold_lock(
    ifo_mode: str,
    ana_ver: str,
    data_ver: str,
    cl_config: str,
    coh_mode: str,
    fm_config: str,
    run_name: str,
    infer_sample_rate: int,
    psd_length: float,
    threshold_level: float,
    **kwargs
):

    if run_name not in noise_runs_list:
        print(f"Warning! Run name {run_name} not in {noise_runs_list}")

    ana_mode = f"{ifo_mode}/{ana_ver}"
    model = f"{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    log_dir = gwak_logging_dir(
        suffix=f"infer/{ana_mode}/{model}/{run_name}"
    )()
    log_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_dir / "threshold_lock.log")
    louvre_dir = gwak_louvre_dir(
        suffix=f"{ana_mode}/{model}/{run_name}"
    )()
    model_louvre_dir, model_snapshot_dir = lovure_file_handler(
        model_louvre_dir=louvre_dir, model=model
    )

    tslide_data_dir = gwak_output_dir(
        suffix=f"infer/{ana_mode}/{model}/{run_name}/inference_result"
    )()

    threshold_file = gwak_output_dir(
        suffix=f"infer/{ana_mode}/{model}"
    )(append_path="threshold.h5")

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

    # Plotting
    if threshold_level >= 1:
        hist_label = f"Max Outlier: {np.min(tslide_data):.2f} \
        \n{ordinal(threshold_level)} Outlier: {threshold:.2f}"

    elif threshold_level < 1:
        hist_label = f"Max Outlier: {np.min(tslide_data):.2f} \
        \n{threshold_level*100}% Outlier: {threshold:.2f}"

    # Plot Timeslide outputs
    plt.title(f"{run_name.capitalize()} \n{model} \nTimeslide Output distribution")
    plt.hist(tslide_data, bins=100, zorder=2, label=hist_label,)
    plt.legend()
    plt.axvline(threshold, color="red")
    plt.grid(zorder=0)
    plt.yscale("log")
    plt.xlabel("Metric")
    plt.ylabel("Data Counts")
    plt.savefig(model_louvre_dir/"TS_ana.png", dpi=300, bbox_inches='tight')
    plt.close()


def scan_outlier(
    ifo_mode: str,
    ana_ver: str,
    data_ver: str,
    cl_config: str,
    coh_mode: str,
    fm_config: str,
    run_name: str,
    threshold_setting: str,
    infer_sample_rate: int,
    psd_length: float,
    accumlation_length:float,
    pad:float,
    threshold_value: Optional[float]=None,
    **kwargs
):

    ana_mode = f"{ifo_mode}/{ana_ver}"
    model = f"{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    stream_cut = int(infer_sample_rate*psd_length)

    # Initialize file paths
    log_dir = gwak_logging_dir(
        suffix=f"{ana_mode}/{model}/{run_name}_{threshold_setting}"
    )()
    log_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_dir / "scan_outlier.log")

    threshold_file = gwak_output_dir(
        suffix=f"infer/{ana_mode}/{model}"
    )(append_path="threshold.h5")

    tslide_data_dir = gwak_output_dir(
        suffix=f"infer/{ana_mode}/{model}/{run_name}/inference_result"
    )()
    outlier_file = gwak_output_dir(
        suffix=f"infer/{ana_mode}/{model}/{run_name}"
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


def bbc_benchmark(
    ifo_mode: str,
    ana_ver: str,
    data_ver: str,
    cl_config: str,
    coh_mode: str,
    fm_config: str,
    foreground: str,
    threshold_setting: str,
    buffer_dur: float = 1,
    # This needs versioning, and should propergate to plot_bbc_benchmark
    outlier_cfg_name: str = "outlier_config.h5",
    **kwargs
):

    # Init
    # model = f"{cl_config}_{coh_mode}_{fm_config}_{ifo_mode}"
    ana_mode = f"{ifo_mode}/{ana_ver}"
    model = f"{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
    log_dir = gwak_logging_dir(
        suffix=f"{ana_mode}/{model}/{foreground}_{threshold_setting}"
    )()
    gwak_logger(log_dir / "benchmark.log")

    # Input setting
    O4_bbc_dir = Path(
        "/home/burst.benchmark/unblinded_o4b-2_injections/injections/"
    )
    unbind_file_dict = {
        "bbc-short-0": O4_bbc_dir / "burst_benchmark_short-0.h5",
        "bbc-short-1": O4_bbc_dir / "burst_benchmark_short-1.h5"
    }
    unblind_file = unbind_file_dict[foreground]
    signal_groups = get_bbc_inj_names(unblind_file)

    outlier_config = gwak_output_dir(suffix=f"infer/{ana_mode}/{model}")(
        append_path=f"{foreground}/{outlier_cfg_name}"
    )
    # threshold_file = gwak_output_dir(suffix=f"infer/{model}")(
    #     append_path="threshold.h5"
    # )
    # with h5py.File(threshold_file, "r") as h5:
    #     threshold = float(h5[f"{threshold_setting}"][()])

    # Output setting
    louvre_dir = gwak_louvre_dir(suffix=f"{ana_mode}/{model}/{foreground}")()
    output_dir = gwak_output_dir(suffix=f"infer/{ana_mode}/{model}/{foreground}")()
    model_louvre_dir, model_snapshot_dir = lovure_file_handler(
        model_louvre_dir=louvre_dir,
        model=model,
    )

    benchmark_result = output_dir / "bbc-unpack.h5"
    false_trigger_config = output_dir / "false_triggers.h5"

    # Output inits
    valid_arrays = []
    inj_total_count = 0
    gwak_triggered_count = 0
    performance= {}

    # Main process
    logging.info(f"Unpacking {foreground}")

    data_config = resolve_oulier_config(outlier_config)
    # outlier_keys = data_config.key_list
    outlier_keys = ["seg_start", "seg_end", "event_start", "event_end"]
    outlier_info = data_config.get_result_by_key(outlier_keys)
    trigger_time = (outlier_info["event_start"] + outlier_info["event_end"])/2

    # Apply scanning logic
    bbc_info_generator = bbc_inj_info(unblind_file, signal_groups, buffer_dur)
    for inj_time_buffer, bbc_inj_count, signal in bbc_info_generator:

        # Collecting performance
        valid = find_valid_triggers(inj_time_buffer, trigger_time)
        triggered_count = sum(valid)
        performance[signal] = np.array([triggered_count, bbc_inj_count])

        # Meta data
        valid_arrays.append(valid)
        inj_total_count += bbc_inj_count
        gwak_triggered_count += triggered_count

    has_duplicates = np.any(np.count_nonzero(valid_arrays, axis=0) > 1)
    valid_triggers = np.logical_or.reduce(valid_arrays)
    logging.info(f"    Has duplicate triggers: {has_duplicates}")

    triggered_raito = gwak_triggered_count/inj_total_count
    err_count = len(trigger_time) - gwak_triggered_count
    err_ratio = err_count / gwak_triggered_count
    logging.info(f"    Correct_triggered_count = {gwak_triggered_count}")
    logging.info(f"    Error report count (ratio): {err_count} ({err_ratio:.02f})")
    logging.info(f"    bbc_inj_total_count = {inj_total_count}")
    logging.info(f"    triggered_raito ={triggered_raito:.4f}")
    logging.info("")

    # Saving result
    with h5py.File(benchmark_result, "w") as h:
        for name, values in performance.items():
            h.create_dataset(name, data=values)
    logging.info(f"Benchmark_result saved at: {benchmark_result}")
    if sum(~valid_triggers) > 0:
        false_triggers = {
            key: outlier_info[key][~valid_triggers] for key in outlier_keys
        }

        with h5py.File(false_trigger_config, "w") as h:

            for key in outlier_keys:
                h.create_dataset(key, data=false_triggers[key])
