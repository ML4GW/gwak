import h5py
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path
from deploy.libs import gwak_logger
from collections import defaultdict
from deploy.libs import gwak_output_dir, gwak_louvre_dir, gwak_logging_dir

from deploy.libs.trigger_io import lovure_file_handler, resolve_oulier_config
from deploy.libs.analysis_utils import get_bbc_inj_names, bbc_inj_info, find_valid_triggers

def find_outlier_segs(
    cl_config: str,
    fm_config: str,
    ifo_mode: str,
    threshold_setting: str,
    seg_count: int,
    **kwargs
):

    # Init
    run_name = threshold_setting    
    outlier_ticks = np.arange(1, seg_count+1)
    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    model_louvre_dir = gwak_louvre_dir(suffix=f"{model}/{run_name}")()
    # model_louvre_dir, model_snapshot_dir = lovure_file_handler(
    #     model_louvre_dir=model_louvre_dir, model=model
    # )

    log_dir = gwak_logging_dir(
        suffix=f"{model}/{threshold_setting}"
    )()
    log_dir.mkdir(parents=True, exist_ok=True)
    gwak_logger(log_dir / "find_outlier_segs.log")
    outlier_file = gwak_output_dir(suffix=f"infer/{model}/{run_name}")(
        append_path="outlier_config.h5"
    )

    # Get data
    data_config = resolve_oulier_config(outlier_file)
    outlier_segs_by_count = data_config.find_exotic_segs_by_count(seg_count)
    outlier_segs_by_rate = data_config.find_exotic_segs_by_rate(seg_count)

    # Plotting
    segments = [seg for seg, _, _ in outlier_segs_by_count][::-1]
    counts = np.array([count for _, count, _ in outlier_segs_by_count])[::-1]
    rates = np.array([rate for _, _, rate in outlier_segs_by_count])[::-1]

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_title(
        f"Top {seg_count} Problematic Segments \n by Outlier Event Counts", 
        fontsize=13, fontweight='bold'
    )
    hbars = ax.barh(
        outlier_ticks, counts,
        color="black", alpha=0.7, height=0.5, zorder=2
    )
    for bar, rate in zip(hbars, rates):
        ax.text(
            0.002,bar.get_y() + bar.get_height()*1.3,
            f"{rate:.02f} per sec", va="center", ha="left", color="black"
        )
    ax.set_yticks(outlier_ticks, segments)
    ax.set_xlabel("Outlier count", fontweight="bold")
    plt.ylabel("Segments",  fontweight="bold")
    plt.grid(zorder=0)
    plt.savefig(
        model_louvre_dir / "Scaned_Segments-counts.png", 
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    segments = [seg for seg, _, _ in outlier_segs_by_rate][::-1]
    counts = np.array([count for _, count, _ in outlier_segs_by_rate])[::-1]
    rates = np.array([rate for _, _,rate in outlier_segs_by_rate])[::-1]

    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_title(
        f"Top {seg_count} Problematic Segments \n by Outlier Event Rate",
        fontsize=13, fontweight="bold"
    )
    hbars = ax.barh(
        outlier_ticks, rates*60,
        color="black", alpha=0.7, height=0.5, zorder=2
    )
    for bar, count in zip(hbars, counts):
        ax.text(
            0.002,bar.get_y() + bar.get_height()*1.3,
            f"{count} triggers", va="center", ha="left", color="black"
        )
    ax.set_yticks(outlier_ticks, segments)
    ax.set_xlabel("Outlier rate (1/minutes)", fontweight="bold")
    plt.ylabel("Segments",  fontweight="bold")
    plt.grid(zorder=0)
    plt.savefig(
        model_louvre_dir/"Scaned_Segments-rate.png", 
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    logging.info(f"Plots saved at: {model_louvre_dir}")





def plot_bbc_benchmark(
    cl_config: str,
    fm_config: str,
    ifo_mode: str,
    threshold_setting: str,
    **kwargs
):

    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    log_dir = gwak_logging_dir(
        suffix=f"{model}/{threshold_setting}"
    )()
    gwak_logger(log_dir / "benchmark.log")

    O4_bbc_dir = Path(
        "/home/burst.benchmark/unblinded_o4b-2_injections/injections/"
    )
    unbind_file_dict = {
        "bbc-short-0": O4_bbc_dir / "burst_benchmark_short-0.h5",
        "bbc-short-1": O4_bbc_dir / "burst_benchmark_short-1.h5"
    }
    unblind_file = unbind_file_dict["bbc-short-0"]
    signal_groups = get_bbc_inj_names(unblind_file)

    # infer_result_dir = Path("/home/hongyin.chen/anti_gravity/gwak/gwak/output/infer/ResNet_6d_NF_from_file_conditioning_HL")
    louvre_dir = gwak_louvre_dir(suffix=f"{model}/{threshold_setting}")()
    infer_result_dir = gwak_output_dir(suffix=f"infer/{model}")()

    unpack_file_1 = infer_result_dir / "bbc-short-0/bbc-unpack.h5"
    unpack_file_2 = infer_result_dir / "bbc-short-1/bbc-unpack.h5"

    h5_info_1 = h5py.File(unpack_file_1, "r")
    h5_info_2 = h5py.File(unpack_file_2, "r")

    total_trigger_count = 0
    plt.figure(figsize=(18, 8), dpi=400)
    for signal_type, signals in signal_groups.items():
        keys = []
        values = []

        for signal in signals:
            trigger_count = h5_info_1[signal][0] + h5_info_2[signal][0]
            total_count = h5_info_1[signal][1] + h5_info_2[signal][1]

            keys.append(signal)
            values.append(trigger_count)

        plt.bar(keys, values, label=signal_type)
        total_trigger_count += sum(values)

    plt.title(f"GWAK ({model}) \nBBC O4b performance \nRecovered events: {total_trigger_count}")
    # Step 3: formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Wavefrom type")
    plt.ylabel("Recovered Event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(louvre_dir / "trigger-rate.png")
    plt.close()
    h5_info_1.close()
    h5_info_2.close()
