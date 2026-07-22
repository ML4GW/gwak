import h5py
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from deploy.libs import gwak_logger
short0_data_dir = Path.home() / "anti_gravity/gwak/gwak/output/data/O4_MDC_short-0/HL"

def get_bbc_inj_names(
    file,
):
    signal_groups = defaultdict(list)

    with h5py.File(file, "r") as h5_file:
        signal_list = list(h5_file.keys())
        for signal in signal_list:
            signal_type = signal.split('_')[0]
            signal_groups[signal_type].append(signal)

    return signal_groups


def bbc_inj_info(
    unblind_file,
    signal_groups,
    buffer_dur = 1
):

    with h5py.File(unblind_file, "r") as unblind_info:
        for key, signals in signal_groups.items():
            for signal in signals:
                inj_time = unblind_info[signal]["PARAMETERS"][:]["time"][:]
                inj_count = inj_time.shape[0]

                inj_time_buffer = np.empty((inj_count, 2))
                inj_time_buffer[:, 0] = inj_time - buffer_dur/2
                inj_time_buffer[:, 1] = inj_time + buffer_dur/2

                yield inj_time_buffer, inj_count, signal

def unpack_outlier_config(file):

    with h5py.File(file, "r") as h5_file:

        outlier_data = h5_file["data"][:]
        # Sort the meta data based on the outlier_start.
        sort_indices = outlier_data[:, 3].argsort()
        outlier_data = outlier_data[sort_indices]
        
    seg_starts, seg_counts = np.unique(
        outlier_data[:, 0], 
        return_counts=True
    )
    i_str = 0
    outlier_time = []
    for seg_count in seg_counts:

        i_end = i_str + seg_count
        seg_outlier = outlier_data[i_str:i_end, :]
        outlier_start, outlier_end, ourlier_value = seg_outlier[:, 3], seg_outlier[:, 4], seg_outlier[:, 6]

        outlier_time.append((outlier_end + outlier_start)/2)
        i_str += seg_count
    triggered_time = np.concatenate(outlier_time)
    return triggered_time    
    
def read_triggered_time(file, threshold, apply_cuts=False):

    data = pd.read_csv(file)
    gwak_value = data["gwak_value"].to_numpy()
    if apply_cuts == False:
        gwak_idx = np.where(gwak_value < threshold)[0]
    else:
        false_idx = data.index[data["vetoed"] == False].to_numpy()
        gwak_idx = false_idx[gwak_value[false_idx] < threshold]

    err_start = data["error_start"].to_numpy()[gwak_idx]
    err_end = data["error_end"].to_numpy()[gwak_idx]

    t0 = data["t0"].to_numpy()[gwak_idx]
    length = data["length"].to_numpy()[gwak_idx]
    # err_start = data["error_start"].to_numpy()
    # err_end = data["error_end"].to_numpy()
    triggered_time = (err_start + err_end ) / 2

    return triggered_time, t0, length


def plot_trigger_rate(
    signal_groups: dict,
    model: str,
    benchmark_dir:Path,
):

    unpack_file = benchmark_dir/ "bbc-unpack.h5"
    total_trigger_count = 0
    plt.figure(figsize=(18, 8), dpi=400)
    h5_info = h5py.File(unpack_file, "r")
    for signal_type, signals in signal_groups.items():
        keys = []
        values = []

        for signal in signals:
            
            trigger_count = h5_info["short-0"][signal][0] + h5_info["short-1"][signal][0]
            total_count = h5_info["short-0"][signal][1] + h5_info["short-1"][signal][1]
            keys.append(signal)
            values.append(trigger_count)
        total_trigger_count += sum(values)

    plt.bar(keys, values, label=signal_type)
    plt.title(f"GWAK ({model}) BBC dataset performance \n Recovered events: {total_trigger_count}")
    # Step 3: formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Wavefrom type")
    plt.ylabel("Recovered Event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(benchmark_dir / "trigger-rate.png")
    h5_info.close()

def read_false_triggers(
    file: Path,
    benchmark_result_dir: Path
):
    with  h5py.File(file, "r") as h5_data:

        t0 = h5_data["short-0"]["t0"][:]
        length = h5_data["short-0"]["length"][:]
        triggered_time = h5_data["short-0"]["triggered_time"][:]

    gwak_h1_glitch = []
    gwak_l1_glitch = []
    for i in range(2):
        data_path = short0_data_dir / f"background-{int(t0[i])}-{int(length[i])}.h5"
        with h5py.File(data_path, "r") as f:
            
            center_time = int((triggered_time[i] - t0[i]) * 4096)
            idx_start = center_time - 4096*3
            idx_end = center_time + 4096*3
            gwak_h1_glitch.append(f["H1"][idx_start:idx_end])
            gwak_l1_glitch.append(f["L1"][idx_start:idx_end])

    gwak_h1_glitch = np.stack(gwak_h1_glitch)
    gwak_l1_glitch = np.stack(gwak_l1_glitch)
    with h5py.File(benchmark_result_dir / "gwak_glitch/false_triggers_strain.h5", "w") as h:

        h.create_dataset("H1", data=gwak_h1_glitch)
        h.create_dataset("L1", data=gwak_l1_glitch)


def read_error_strain(
    file: Path,
    benchmark_result_dir: Path
):
    with h5py.File(file, "r") as h:
        for i in range(2):
            h1 = h["H1"][i]
            l1 = h["L1"][i]
            time = np.arange(h1.shape[0])/4096
            plt.title(f"gwak_glitch-{i}")
            plt.plot(time, h1, label="H1", alpha=0.6)
            plt.plot(time, l1, label="L1", alpha=0.6)
            plt.xlabel("Time(s)")
            plt.legend()
            plt.savefig(benchmark_result_dir / f"gwak_glitch/glitch-{i}.png")
            plt.close()
            # print(benchmark_result_dir / f"gwak_glitch/glitch-{i}.png")


def bbc_benchmark(
    unblind_files: list[Path],
    model:str,
    threshold: float, 
    apply_cuts: bool,
    benchmark_dir: Path,
    buffer_dur: float, 
    **kwargs
):
    print(f"Performance of model: {model}")
    threshold = float(threshold)
    benchmark_dir = Path(benchmark_dir)
    benchmark_result_dir = benchmark_dir / model

    # Input
    ana_files = [
        benchmark_result_dir / "short-0_correlation_cuts.csv",
        benchmark_result_dir / "short-1_correlation_cuts.csv"
    ]
    # Output files
    false_triggers_file = benchmark_result_dir / "gwak_glitch/false_triggers.h5"
    false_triggers_strain =  benchmark_result_dir / "gwak_glitch/false_triggers_strain.h5"
    log_file = benchmark_result_dir / "gwak_glitch/log.log"
    (benchmark_result_dir / "gwak_glitch").mkdir(parents=True, exist_ok=True)
    gwak_logger(log_file)
    loggin_phrase = "before"
    if apply_cuts:
        loggin_phrase = "after"
    signal_groups = get_bbc_inj_names(unblind_files[0])

    with h5py.File(benchmark_result_dir / "bbc-unpack.h5", "w") as unpack_info:
        for bbc_group_count, (unblind_file, ana_file) in enumerate(zip(unblind_files, ana_files)):
            logging.info(f"Short-{bbc_group_count}:")
            valid_arrays = []
            inj_total_count = 0
            gwak_triggered_count = 0
            bbc_group = unpack_info.create_group(f"short-{bbc_group_count}")

            # triggered_time = unpack_outlier_config(ana_file)
            triggered_time, t0, length = read_triggered_time(
                ana_file, 
                threshold=threshold, 
                apply_cuts=apply_cuts
            )
            logging.info(f"    Triggers remained {loggin_phrase} cuts: {len(triggered_time)}")
            
            # READ bbc_inj_info
            bbc_info_generator = bbc_inj_info(unblind_file, signal_groups, buffer_dur)
            for inj_time_buffer, inj_count, signal in bbc_info_generator:

                idx = np.searchsorted(inj_time_buffer[:, 0], triggered_time, side="right") - 1
                valid = (idx >= 0) & (triggered_time <= inj_time_buffer[idx, 1])

                triggered_count = sum(valid)
                valid_arrays.append(valid)

                inj_total_count += inj_count
                gwak_triggered_count += triggered_count

                bbc_group.create_dataset(
                    name=signal,
                    data=np.array([triggered_count, inj_count])
                )
            valid_triggers = np.logical_or.reduce(valid_arrays)
            has_duplicates = np.any(np.count_nonzero(valid_arrays, axis=0) > 1)
            logging.info(f"    Has duplicate = {has_duplicates}")
            if triggered_time[~valid_triggers].shape[0] > 0:
                with h5py.File(benchmark_result_dir / "gwak_glitch/false_triggers.h5", "a") as f:
                    group_name = f"short-{bbc_group_count}"

                    if group_name in f:
                        del f[group_name]

                    group = f.create_group(group_name)
                    group.create_dataset("triggered_time", data=triggered_time[~valid_triggers])
                    group.create_dataset("t0", data=t0[~valid_triggers])
                    group.create_dataset("length", data=length[~valid_triggers])


            triggered_raito = gwak_triggered_count/inj_total_count
            logging.info(f"    Correct_triggered_count = {gwak_triggered_count}")
            err_count = len(triggered_time) - gwak_triggered_count
            err_ratio = err_count / gwak_triggered_count
            logging.info(f"    Error report count (ratio): {err_count} ({err_ratio:.02f})")
            logging.info(f"    bbc_inj_total_count = {inj_total_count}")    
            logging.info(f"    triggered_raito ={triggered_raito:.4f}")
            logging.info("")
    logging.info(f"Data saved at: {benchmark_result_dir}/gwak_glitch/false_triggers.h5")

    plot_trigger_rate(signal_groups, model, benchmark_result_dir,)
    read_false_triggers(false_triggers_file, benchmark_result_dir)
    read_error_strain(false_triggers_strain, benchmark_result_dir)







