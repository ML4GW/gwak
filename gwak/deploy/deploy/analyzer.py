import re
import h5py
import shutil
import yaml

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict
from deploy.libs import accumlator, Pathfinder
from matplotlib import pyplot as plt

# from bokeh.plotting import figure 
# from bokeh.io import output_notebook, save, show, reset_output, export_png
from deploy.libs import gwak_dir, gwak_output_dir, gwak_louvre_dir, O4_bbc_short_0_data_dir, O4_bbc_short_1_data_dir

def lovure_file_handler(
    model_louvre_dir: Path,
    model,
    remake: bool=False,
    caching: bool=True,
):

    model_snapshot_dir = model_louvre_dir / "snapshot"
    if (model_louvre_dir).exists() and remake:

        model_snapshot_dir.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(model_louvre_dir)
        return model_louvre_dir, model_snapshot_dir

    # Check if cache exists
    if (model_louvre_dir/"cache").exists():
        shutil.rmtree(model_louvre_dir/"cache")

    if caching:
        cache_dir = model_louvre_dir / "cache"
        (cache_dir / "snapshot").mkdir(parents=True, exist_ok=True) 

        for png_file in model_louvre_dir.glob("*.png"):
            shutil.move(str(png_file), str(cache_dir / png_file.name))
        for png_file in model_louvre_dir.glob("snapshot/*.png"):
            shutil.move(str(png_file), str(cache_dir / "snapshot" /png_file.name))

    model_snapshot_dir.mkdir(parents=True, exist_ok=True)

    return model_louvre_dir, model_snapshot_dir


def thereshold_selection(
    psd_length,
    infer_sample_rate,
    tslide_data_dir,
    thereshold_level,
    run_name,
    model
):

    tslide_data = []
    tslide_dict = {}

    stream_cut = int(infer_sample_rate*psd_length)
    file_list = list(sorted(tslide_data_dir.glob("*.h5")))

    # The stream_cut will be effected stride_batch_size is too small/large
    for fname in tqdm(file_list):

        with h5py.File(fname, "r") as h5_file: 
            
            gwk_stream = h5_file["gwak_value"][stream_cut:]
            tslide_dict[fname] = gwk_stream
            tslide_data.append(gwk_stream)

    # Merge all the nan-truncated timeslide in the list to 
    # one numpy array with the shape of (x_n,) and find the thereshold. 
    tslide_data = np.concatenate(tslide_data)

    if thereshold_level >= 1: 
        thereshold = np.sort(tslide_data)[int(thereshold_level)]
        print()
        print(f"    The top {int(thereshold_level)}th of {run_name} outlier of {model} is at : {round(thereshold, 2)}.")
        print()

    if thereshold_level < 1: 
        thereshold = np.quantile(tslide_data, thereshold_level)
        print()
        print(f"    The {run_name} {thereshold_level} thereshold of {model} is at : {round(thereshold, 2)}.")
        print()

    return tslide_dict, stream_cut, tslide_data, thereshold


def scan(
    # louvre_dir: Pathfinder,
    cl_config: str, 
    fm_config: str,
    ifo_mode: str, 
    run_name: str,
    seg_num: int,
    thereshold_level: float,
    infer_sample_rate: int,
    psd_length: float,
    plot_padding: int,
    plotting: bool,
    **kwargs
):

    anomaly_dict = {}
    anomaly_data = {}


    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    louvre_dir = gwak_louvre_dir(suffix=f"{model}/{run_name}")()
    tslide_data_dir = gwak_output_dir()(
        append_path=f"infer/{model}/{run_name}/inference_result"
    )

    model_louvre_dir, model_snapshot_dir = lovure_file_handler(
        model_louvre_dir=louvre_dir,
        model=model
    )

    tslide_dict, stream_cut, tslide_data, thereshold = thereshold_selection(
        psd_length=psd_length,
        infer_sample_rate=infer_sample_rate,
        tslide_data_dir=tslide_data_dir,
        thereshold_level=thereshold_level,
        run_name=run_name,
        model=model
    )

    for fname, ts_data in tslide_dict.items():

        fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)_(?P<shift>\d+\.*\d*)")
        match = fname_re.search(str(fname))

        if match is None:
            print(f"Couldn't parse file {fname.path}")
            # logging.warning(f"Couldn't parse file {fname.path}")

        start = int(match.group("t0"))
        length = int(match.group("length"))
        shift = int(float(match.group("shift")))

        if length <= int(psd_length): # Skip data that are too short

            print(f"Skip {fname}")
            continue
        try:
            np.min(ts_data) < thereshold
        except:
            continue
        if np.min(ts_data) < thereshold:
            segment_name = f"{start}-{length}"

            indices = np.where(ts_data < thereshold)[0]

            if anomaly_dict.get(segment_name) is None:
                anomaly_dict[segment_name] = []
            if anomaly_data.get(segment_name) is None:
                anomaly_data[segment_name] = {}
            anomaly_data[segment_name][f"{shift}"] = []

            indices = np.where(ts_data < thereshold)[0]

            start_pad = plot_padding
            end_pad = ts_data.shape[0] - (start_pad + 1)
            anomaly_dict[segment_name].append((shift, indices, ts_data[indices]))
            indices = indices[(indices > start_pad) & (indices < end_pad)]

            for idx in indices:
                ts_data[indices]
                if ts_data[idx - start_pad: idx + (start_pad + 1)].shape[0] != (start_pad + start_pad + 1):
                    continue
                anomaly_data[segment_name][f"{shift}"].append(ts_data[idx - start_pad: idx + (start_pad + 1)])

    # Make Problematic Segments data
    outlier_segments_name = []
    outlier_counts_per_seg = []
    outlier_rate_per_seg = []
    if len(anomaly_dict.keys()) < seg_num:
        seg_num = len(anomaly_dict.keys())

    for count, seg_name in enumerate(anomaly_dict.keys()):

        fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)")
        match = fname_re.search(seg_name)
        duration = int(match.group("length")) - psd_length
        outlier_segments_name.append(seg_name)
        
        indices_counts = 0
        for i in range(len(anomaly_dict[seg_name])):

            indices_counts += len(anomaly_dict[seg_name][i][1])
            if indices_counts == 0:
                print(f"Insifficent value for outlier {seg_name} {anomaly_dict[seg_name][i][1]}")
                break
        outlier_counts_per_seg.append(indices_counts)
        outlier_rate_per_seg.append(indices_counts/duration)


    sort_idx_count = np.argsort(outlier_counts_per_seg)[-seg_num:]
    sort_idx_rate = np.argsort(outlier_rate_per_seg)[-seg_num:]
    outlier_ticks = np.linspace(1, seg_num, seg_num)

    # Count analysis
    outlier_counts_segments_name = np.array(outlier_segments_name)[sort_idx_count]
    outlier_counts_per_seg_ = np.array(outlier_counts_per_seg)[sort_idx_count]
    rate_per_seg = np.array(outlier_rate_per_seg)[sort_idx_count]
    
    # Rate analysis
    outlier_rate_segments_name = np.array(outlier_segments_name)[sort_idx_rate]
    outlier_rate_per_seg_ = np.array(outlier_rate_per_seg)[sort_idx_rate]
    counts_per_seg = np.array(outlier_counts_per_seg)[sort_idx_rate]

    outlier_config = model_louvre_dir / f"outlier_config.h5"
    with h5py.File(outlier_config, "w") as outlier_h5:
        outlier_data = []
        for seg_name, anomaly_infos in anomaly_dict.items():

            t0 = int(seg_name[:10])
            length = int(seg_name[11:])

            for shift, indices, value in anomaly_infos:

                H1_time = (indices + stream_cut)/infer_sample_rate + t0

                outlier_seg = accumlator(H1_time, value, accumlation_length=16, pad=0.5)

                incre_len = outlier_seg.shape[0]
                meta_data = np.ones((incre_len,3))
                meta_data[:, 0] *= t0
                meta_data[:, 1] *= length
                meta_data[:, 2] *= shift

                outlier_data.append(np.concatenate((meta_data, outlier_seg), axis=1))
        outlier_h5.create_dataset("data", data=np.vstack(outlier_data))

    if plotting:
        # Plot Timeslide outputs
        plt.title(f"{run_name.capitalize()} \n{model} \nTimeslide Output distribution")
        plt.hist(
            tslide_data, 
            bins=100,
            zorder=2,
            label=f"Max Outlier: {np.min(tslide_data):.2f} \n{thereshold_level*100}% Outlier: {thereshold:.2f}",
        )
        plt.legend()
        plt.axvline(thereshold, color="red")
        plt.grid(zorder=0)
        plt.yscale("log")
        plt.xlabel("Metric")
        plt.ylabel("Data Counts")
        plt.savefig(model_louvre_dir/"TS_ana.png", dpi=300, bbox_inches='tight')
        plt.close()


        fig, ax = plt.subplots(figsize=(4, 8))
        ax.set_title(
            f"Top {seg_num} Problematic Segments \n by Outlier Event Rate",
            fontsize=13, fontweight="bold"
        )
        hbars = ax.barh(
            outlier_ticks,
            outlier_rate_per_seg_,
            color="black",
            alpha=0.7, height=0.5, zorder=2
        )
        for bar, count in zip(hbars, counts_per_seg):
            ax.text(
                0.002,bar.get_y() + bar.get_height()*1.3,
                f"{count} triggers",
                va="center",
                ha="left",
                color="black"
            )
        ax.set_yticks(outlier_ticks, outlier_rate_segments_name)
        ax.set_xlabel("Outlier rate per seconds", fontweight="bold")
        plt.ylabel("Segments",  fontweight="bold")
        plt.grid(zorder=0)
        plt.savefig(
            model_louvre_dir/"Scaned_Segments-rate.png", 
            dpi=300, bbox_inches='tight'
        )
        plt.close()


        fig, ax = plt.subplots(figsize=(4, 8))
        ax.set_title(
            f"Top {seg_num} Problematic Segments \n by Outlier Event Counts", 
            fontsize=13, fontweight='bold'
        )
        hbars = ax.barh(
            outlier_ticks,
            outlier_counts_per_seg_,
            color="black",
            alpha=0.7,height=0.5,zorder=2
        )
        for bar, rate in zip(hbars, rate_per_seg):
            ax.text(
                0.002,bar.get_y() + bar.get_height()*1.3,
                f"{rate:.02f} per sec",
                va="center",
                ha="left",
                color="black"
            )
        ax.set_yticks(outlier_ticks, outlier_counts_segments_name)
        ax.set_xlabel("Outlier rate per seconds", fontweight="bold")
        plt.ylabel("Segments",  fontweight="bold")
        plt.grid(zorder=0)
        plt.savefig(
            model_louvre_dir/"Scaned_Segments-counts.png", 
            dpi=300, bbox_inches='tight'
        )
        plt.close()


        snap_time = np.arange(0, 10+1/infer_sample_rate, 1/infer_sample_rate)
        for seg_name, shift_dict in anomaly_data.items():
            if seg_name in outlier_segments_name[-5:]:
                plt.figure(figsize=(10, 4))
                plt.title("GWAK Stream snapshot")
                for shift, data_list in shift_dict.items():
                    for snapshot in data_list:
                        plt.plot(snap_time, snapshot)
                plt.xlabel("Time(s)")
                plt.savefig(model_snapshot_dir / f"GWAK-Stream_{seg_name}.png", dpi=300, bbox_inches='tight')
                plt.close()


        for seg_name, anomaly_infos in anomaly_dict.items():
            if seg_name in outlier_segments_name[-5:]:
                t0 = int(seg_name[:10])
                length = int(seg_name[11:])
                outlier_values = []
                h1_outlier_times = []
                l1_outlier_times = []

                for shift, indices, value in anomaly_infos:

                    H1_time = (indices + stream_cut)/infer_sample_rate + t0 * 0
                    L1_time = (indices + stream_cut)/infer_sample_rate + shift + t0 * 0 

                    outlier_values.append(value)
                    h1_outlier_times.append(H1_time)
                    l1_outlier_times.append(L1_time)

                h1_timestamps = np.sort(np.concatenate(h1_outlier_times)) * infer_sample_rate
                l1_timestamps = np.sort(np.concatenate(l1_outlier_times)) * infer_sample_rate

                h1_second_indices = h1_timestamps.astype(int)
                l1_second_indices = l1_timestamps.astype(int)
                h1_counts = np.bincount(h1_second_indices, minlength=length*infer_sample_rate)
                l1_counts = np.bincount(l1_second_indices, minlength=length*infer_sample_rate)
                
                plt.figure(figsize=(10, 4))
                plt.title(f"Segment: {seg_name} Error Rate")
                plt.plot(np.arange(0, length, 1/infer_sample_rate), h1_counts, label="H1")
                plt.plot(np.arange(0, length, 1/infer_sample_rate), l1_counts, label="L1")
                plt.xlabel("Time(s)")
                plt.ylabel("Error count")
                plt.legend()
                plt.savefig(model_snapshot_dir / f"GWAK-Stream_{seg_name}_outlier_rate.png", dpi=300, bbox_inches='tight')
                plt.close()

    print(f"Plots saved at: {louvre_dir}")