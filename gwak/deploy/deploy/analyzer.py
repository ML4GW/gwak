import re
import h5py
import shutil
import yaml

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from deploy.libs import accumlator, Pathfinder
from matplotlib import pyplot as plt

# from bokeh.plotting import figure 
# from bokeh.io import output_notebook, save, show, reset_output, export_png

def lovure_file_handler(
    louvre_dir,
    model,
    remake: bool=False,
    caching: bool=True,
):

    
    model_louvre_dir = louvre_dir(model)
    model_snapshot_dir = model_louvre_dir / "snapshot"

    if (model_louvre_dir).exists() and remake:
        shutil.rmtree(model_louvre_dir)

        model_snapshot_dir.mkdir(parents=True, exist_ok=True)

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


def scan(
    out_dir: Pathfinder,
    cl_config: str, 
    fm_config: str,
    ifo_mode: str, 
    project: str,
    seg_num: int,
    thereshold_level: float,
    infer_sample_rate: int,
    psd_length: float,
    plot_padding: int,
    plotting: bool,
    **kwargs
):
    
    model = f"{cl_config}_{fm_config}_{ifo_mode}"
    tslide_data_dir = Path(f"/fred/oz016/Andy/Output/gwak-bbc/{model}/{project}/inference_result")
    model_louvre_dir, model_snapshot_dir = lovure_file_handler(
        louvre_dir=out_dir,
        model=model
    )
    stream_cut = int(infer_sample_rate*psd_length)
    anomaly_dict = {}
    anomaly_data = {}

    file_list = list(sorted(tslide_data_dir.glob("*.h5")))

    tslide_data_list = []
    # # The stream_cut will be effected stride_batch_size is too small/large
    for fname in tqdm(file_list):

        with h5py.File(fname, "r") as h5_file: 

            tslide_data_list.append(h5_file["data"][0, stream_cut:])

    # Merge all the nan-truncated timeslide in the list to 
    # one numpy array with the shape of (x_n,) and find the thereshold. 
    tslide_data = np.concatenate(tslide_data_list)

    if thereshold_level >= 1: 
        thereshold = np.sort(tslide_data)[int(thereshold_level)]
        print()
        print(f"    The top {int(thereshold_level)}th of {project} outlier of {model} is at : {round(thereshold, 2)}.")
        print()

    if thereshold_level < 1: 
        thereshold = np.quantile(tslide_data, thereshold_level)
        print()
        print(f"    The {project} {thereshold_level} thereshold of {model} is at : {round(thereshold, 2)}.")
        print()
    for ts_data, fname in zip(tslide_data_list, file_list):

        fname_re = re.compile(r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)_(?P<shift>\d+\.*\d*)")
        match = fname_re.search(str(fname))

        if match is None:
            print(f"Couldn't parse file {fname.path}")
            # logging.warning(f"Couldn't parse file {fname.path}")

        start = int(match.group("t0"))
        length = int(match.group("length"))
        shift = int(float(match.group("shift")))

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
                    breakpoint()
                anomaly_data[segment_name][f"{shift}"].append(ts_data[idx - start_pad: idx + (start_pad + 1)])

    # Make Problematic Segments data
    error_segments_name = []
    error_counts_per_seg = []
    if len(anomaly_dict.keys()) < seg_num:
        seg_num = len(anomaly_dict.keys())

    for count, seg_name in enumerate(anomaly_dict.keys()):

        error_segments_name.append(seg_name)
        indices_counts = 0
        for i in range(len(anomaly_dict[seg_name])):

            indices_counts += len(anomaly_dict[seg_name][i][1])
            if indices_counts == 0:
                print(f"Insifficent value for outlier {seg_name} {anomaly_dict[seg_name][i][1]}")
                break
        error_counts_per_seg.append(indices_counts)

    sort_idx = np.argsort(error_counts_per_seg)[-seg_num:]

    error_ticks = np.linspace(1, seg_num, seg_num)
    error_segments_name = np.array(error_segments_name)[sort_idx]
    error_counts_per_seg = np.array(error_counts_per_seg)[sort_idx]


    error_config = model_louvre_dir / f"error_config.h5"
    with h5py.File(error_config, "w") as error_h5:
        error_data = []
        for seg_name, anomaly_infos in anomaly_dict.items():

            t0 = int(seg_name[:10])
            length = int(seg_name[11:])

            for shift, indices, value in anomaly_infos:

                H1_time = (indices + stream_cut)/infer_sample_rate + t0

                error_seg = accumlator(H1_time, value, accumlation_length=16, pad=0.5)

                incre_len = error_seg.shape[0]
                meta_data = np.ones((incre_len,3))
                meta_data[:, 0] *= t0
                meta_data[:, 1] *= length
                meta_data[:, 2] *= shift

                error_data.append(np.concatenate((meta_data, error_seg), axis=1))
        error_h5.create_dataset("data", data=np.vstack(error_data))

    if plotting:
        # Plot Timeslide outputs
        plt.title(f"Timeslide Output distribution: {project}")
        plt.hist(
            tslide_data, 
            bins=100,
            zorder=2,
            # label=f"Most extrem ourlier value{np.min(tslide_data):.2f}",
        )
        # plt.legend()
        plt.axvline(thereshold, color="red")
        plt.grid(zorder=0)
        plt.yscale("log")
        plt.xlabel("Metric")
        plt.ylabel("Data Counts")
        plt.savefig(model_louvre_dir/"TS_ana.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot Problematic Segments
        plt.figure(figsize=(4, 8))
        plt.title(f"Top {seg_num} Problematic Segments", fontsize=13, fontweight='bold')
        plt.barh(
            error_ticks, 
            error_counts_per_seg, 
            color="black", 
            alpha=0.7,
            height=0.5,
            zorder=2
        )
        plt.yticks(error_ticks, error_segments_name)
        plt.xscale("log")
        plt.xlabel("Anomaly counts", fontweight="bold")
        plt.ylabel("Segments",  fontweight="bold")
        plt.grid(zorder=0)
        plt.savefig(model_louvre_dir/"Scaned_Segments.png", dpi=300, bbox_inches='tight')
        plt.close()


        snap_time = np.arange(0, 10+1/infer_sample_rate, 1/infer_sample_rate)
        for seg_name, shift_dict in anomaly_data.items():
            if seg_name in error_segments_name[-5:]:
                plt.figure(figsize=(10, 4))
                plt.title("GWAK Stream snapshot")
                for shift, data_list in shift_dict.items():
                    for snapshot in data_list:
                        plt.plot(snap_time, snapshot)
                plt.xlabel("Time(s)")
                plt.savefig(model_snapshot_dir / f"GWAK-Stream_{seg_name}.png", dpi=300, bbox_inches='tight')
                plt.close()


        for seg_name, anomaly_infos in anomaly_dict.items():
            if seg_name in error_segments_name[-5:]:
                t0 = int(seg_name[:10])
                length = int(seg_name[11:])
                error_values = []
                h1_error_times = []
                l1_error_times = []

                for shift, indices, value in anomaly_infos:

                    H1_time = (indices + stream_cut)/infer_sample_rate + t0 * 0
                    L1_time = (indices + stream_cut)/infer_sample_rate + shift + t0 * 0 

                    error_values.append(value)
                    h1_error_times.append(H1_time)
                    l1_error_times.append(L1_time)

                h1_timestamps = np.sort(np.concatenate(h1_error_times)) * infer_sample_rate
                l1_timestamps = np.sort(np.concatenate(l1_error_times)) * infer_sample_rate

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
                plt.savefig(model_snapshot_dir / f"GWAK-Stream_{seg_name}_error_rate.png", dpi=300, bbox_inches='tight')
                plt.close()

