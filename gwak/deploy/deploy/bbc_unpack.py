import h5py
import torch

import numpy as np
from pathlib import Path

# short_0_path = "/home/burst.benchmark/unblinded_o4b-2_injections/injections/burst_benchmark_short-0.h5"
# short_1_path = "/home/burst.benchmark/unblinded_o4b-2_injections/injections/burst_benchmark_short-1.h5"

unblind_files = [
    "/home/burst.benchmark/unblinded_o4b-2_injections/injections/burst_benchmark_short-0.h5",
    "/home/burst.benchmark/unblinded_o4b-2_injections/injections/burst_benchmark_short-1.h5"
]


outlier_files = [
    "/home/hongyin.chen/Outputs/GWAK/LLO/GWAK/torch_rbw_zp_resnet_do6_dcs128_epoch25_NF_from_file_conditioning_HL-O4_MDC_short-0/error_config.h5",
    "/home/hongyin.chen/Outputs/GWAK/LLO/GWAK/torch_rbw_zp_resnet_do6_dcs128_epoch25_NF_from_file_conditioning_HL-O4_MDC_short-1/error_config.h5"
]

def hrrs_value(
    h_plus: torch.Tensor,
    h_cross: torch.Tensor,
):
    """
    Returns:
        sqrt(sum(h_plus^2 + h_cross^2))
    """
    h_plus_s = torch.square(h_plus)
    h_cross_s = torch.square(h_cross)

    hrrs = torch.sqrt(
        torch.sum(h_plus_s + h_cross_s)
    )

    return hrrs



for outlier_file, unblind_file in zip(outlier_files, unblind_files):

    with h5py.File(outlier_file, "r") as h5_file:

        outlier_data = h5_file["data"][:]
        # Sort the meta data based on the outlier_start.
        sort_indices = outlier_data[:, 3].argsort()
        outlier_data = outlier_data[sort_indices]

    seg_starts, seg_counts = np.unique(
        outlier_data[:, 0], 
        return_counts=True
    )


    """
    read bbc outlier
    seg_counts,
    outlier_data,
    GWAK_BBC_SHORT_0_PATH,
    inference_sampling_rate
    """
    
    outlier_time = []
    h1_data = []
    l1_data = []

    sample_rate = 4096
    inference_sampling_rate = 4
    i_str = 0
    psd_len = 64

    fft_len = 1
    buffer_len = 1
    kernel_len = 1
    walk_in_len = (inference_sampling_rate - 1)/inference_sampling_rate
    kernel_len_h = kernel_len / 2
    
    stream_buffer = fft_len + buffer_len/2
    stream_shift = walk_in_len + kernel_len_h
    
    for seg_count in seg_counts:

        i_end = i_str + seg_count
        seg_outlier = outlier_data[i_str:i_end, :]
        seg_start, seg_dur = int(seg_outlier[0,0]), int(seg_outlier[0,1])
        outlier_start, outlier_end, ourlier_value = seg_outlier[:, 3], seg_outlier[:, 4], seg_outlier[:, 6]

        outlier_time.append((outlier_end + outlier_start)/2)
        i_str += seg_count




    triggered_values = np.concatenate(outlier_time)

    # print()
    # print()
    print(f"GWAK outlier count: {triggered_values.shape[0]}")
    # print()
    # print()


    counts = 2

    type_name = []
    inj_total_count = 0
    gwak_triggered_count = 0
    # for unblind_file in unblind_files:
    # unblind_file = unblind_files
    h5_file = h5py.File(unblind_file, "r")

    for key in list(h5_file.keys()):

        type_name.append(h5_file[key].attrs["type"])
        inj_time = h5_file[key]["PARAMETERS"][:]["time"][:]
        inj_count = inj_time.shape[0]
        inj_total_count += inj_count
        inj_time_buffer = np.empty((inj_count, 2))
        inj_time_buffer[:, 0] = inj_time - 1
        inj_time_buffer[:, 1] = inj_time + 1
        # for name in h5_file[key]["PARAMETERS"].dtype.names:
        #     print(f"    {name}")

        # for time_stamp in h5_file[key]["PARAMETERS"][:]["time"][:counts]:
        #     print(f"{time_stamp = }")
        # hrss_norm = h5_file[key].attrs.get("hrss_norm")
        # if hrss_norm:
        #     print(h5_file[key]["PARAMETERS"][:]["amplitude"][:counts]*hrss_norm)
        # else:
        #     print(h5_file[key]["PARAMETERS"][:]["amplitude"][:counts])
        # print(h5_file[key].attrs.get("hrss_norm"))
        ranges = inj_time_buffer

        idx = np.searchsorted(
            ranges[:, 0], 
            triggered_values, 
            side="right"
        ) - 1

        valid = (idx >= 0) & (triggered_values <= ranges[idx, 1])

        triggered_idx = np.unique(idx[valid])
        triggered_ranges = ranges[triggered_idx]
        
        if len(triggered_ranges) > 0:
            print(f"{key}: {len(triggered_ranges)/inj_count:.2f}")
            # print(h5_file[key].attrs["type"])
        #     print(triggered_ranges)
            gwak_triggered_count += len(triggered_ranges)

        # print()

    h5_file.close()
    print()
    print(f"{gwak_triggered_count = }")
    print(f"{inj_total_count = }")
    triggered_raito = gwak_triggered_count/inj_total_count
    print(f"triggered_raito ={triggered_raito:.4f}")
    print()
    print()
# inj_count = inj_time.shape[0]
# inj_time_buffer = np.empty((inj_count, 2))



# import numpy as np

# triggered = np.array([
#     1402053391.5,
#     1402058541.0,
#     1402065086.125,
#     1402080563.5,
#     1402080644.25,
#     1402080847.75,
#     1402082963.0,
#     1402104875.875,
#     1402107891.0,
#     1402112296.25,
# ])

# values = triggered
# ranges = np.array([
#     [2, 4],
#     [6, 8],
#     [10, 12],
#     # [1402053391,14020533912]
# ])
# # ranges = inj_time_buffer

# idx = np.searchsorted(ranges[:, 0], values, side="right") - 1

# valid = (idx >= 0) & (values <= ranges[idx, 1])

# triggered_idx = np.unique(idx[valid])
# triggered_ranges = ranges[triggered_idx]

# if len(triggered_ranges) > 0:
#     print(triggered_ranges)
#     print("======================================================")

# breakpoint()
