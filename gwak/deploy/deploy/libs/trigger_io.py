import h5py
import re
import shutil
import logging

import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from deploy.libs import accumlator
from deploy.libs.loggers import ordinal


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


def unpack_timeslide(
    infer_sample_rate,
    psd_length,
    tslide_data_dir,
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
    # one numpy array with the shape of (x_n,) and find the threshold. 
    tslide_data = np.concatenate(tslide_data, axis=0).ravel()

    return tslide_dict, tslide_data

def select_threshold(
    tslide_dict,
    tslide_data,
    threshold_level,
    run_name,
    model,
    verbose=True
):

    if threshold_level >= 1: 
        threshold = np.sort(tslide_data)[int(threshold_level)]

        text_1 = f"    The {ordinal(threshold_level)} most outlier trigger "
        text_2 = f"of {run_name} distribution of {model} is: "
        text_3 = f"{round(threshold, 2)}."
        full_message = text_1 + text_2 + text_3

    if threshold_level < 1: 
        threshold = np.quantile(tslide_data, threshold_level)

        text_1 = f"    The {int(threshold_level*100)}% outlier from "
        text_2 = f"{run_name} distribution of {model} is: "
        text_3 = f"{round(threshold, 2)}."
        full_message = text_1 + text_2 + text_3

    if verbose:
        logging.info(f"")
        logging.info(full_message)
        logging.info(f"")

    return threshold

def find_outlier_by_segmets(
    tslide_dict,
    threshold,
    infer_sample_rate,
    psd_length,
    accumlation_length,
    pad,
):

    stream_cut = int(infer_sample_rate*psd_length)
    outlier_dict = defaultdict(list)
    for fname, ts_data in tslide_dict.items():

        fname_re = re.compile(
            r"(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*)_(?P<shift>\d+\.*\d*)"
        )
        match = fname_re.search(str(fname))

        if match is None:
            logging.error(f"Couldn't parse file {fname.path}")
            # logging.warning(f"Couldn't parse file {fname.path}")

        start = int(match.group("t0"))
        length = int(match.group("length"))
        shift = int(float(match.group("shift")))


        # Data checking logic
        if length <= int(psd_length): # Skip data that are too short
            logging.debug(f"Data frame too short skip {fname.name}")
            continue
        if ts_data.size == 0:
            continue
        if np.any(np.isnan(ts_data)):
            logging.error(f"Found nan value in {fname}")
            continue
        if not np.any(ts_data < threshold):
            continue

        indices = np.where(ts_data < threshold)[0]
        H1_time = (indices + stream_cut)/infer_sample_rate + start
        value = ts_data[indices]
        outlier_info = accumlator(
            H1_time, value, 
            accumlation_length=accumlation_length, pad=pad
        )
        event_counts = outlier_info.shape[0]

        outlier_dict["seg_start"].append(np.repeat(start, event_counts))
        outlier_dict["seg_end"].append(np.repeat(length, event_counts))
        outlier_dict["shifts"].append(np.repeat(shift, event_counts))
        outlier_dict["event_start"].append(outlier_info[:, 0])
        outlier_dict["event_end"].append(outlier_info[:, 1])
        outlier_dict["dur"].append(outlier_info[:, 2])
        outlier_dict["max_value"].append(outlier_info[:, 3])
    return outlier_dict
