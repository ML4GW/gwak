from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np



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


def find_valid_triggers(inj_time_buffer, trigger_time):

    """
    inj_time_buffer = np.array([[1, 2], [4, 5]])
    triggered_time = np.array([0.9, 1.5, 3.7, 4.5])
    return np.array([False,  True, False,  True])
    """

    idx = np.searchsorted(inj_time_buffer[:, 0], trigger_time, side="right") - 1
    valid = (idx >= 0) & (trigger_time <= inj_time_buffer[idx, 1])

    return valid


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
            
            trigger_count = h5_info["short-0"][signal][0] + h5_info["short-1"][signal][0]*0
            total_count = h5_info["short-0"][signal][1] + h5_info["short-1"][signal][1]*0

            keys.append(signal)
            values.append(trigger_count)

        plt.bar(keys, values, label=signal_type)
        total_trigger_count += sum(values)

    plt.title(f"GWAK ({model}) BBC short-0 performance \n Recovered events: {total_trigger_count}")
    # Step 3: formatting
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Wavefrom type")
    plt.ylabel("Recovered Event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(benchmark_dir / "trigger-rate.png")
    plt.close()
    h5_info.close()





# CSV version of the outliers (cuts)
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
