import os
import glob
import numpy as np
import h5py
import argparse

def merge_h5_files(folder_path, dataset_key='data'):
    """
    Searches for all .h5 files in the given folder, reads the specified dataset,
    and concatenates them along the first axis.
    
    Parameters:
        folder_path (str): Path to the folder containing .h5 files.
        dataset_key (str): Key for the dataset to extract from each file (default 'data').
    
    Returns:
        np.ndarray: The merged array from all files.
    """
    file_pattern = os.path.join(folder_path, '*.h5')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        raise ValueError("No .h5 files found in the provided folder.")
    
    arrays = []
    for file in file_list:
        with h5py.File(file, 'r') as f:
            data = f[dataset_key][:]
            arrays.append(data)
            print(f"Loaded {data.shape} from {file}")
    
    merged_array = np.concatenate(arrays, axis=0)
    return merged_array

def score_to_far(threshold, scores, total_time, direction='negative'):
    """
    Calculate the False Alarm Rate (FAR) for a given score threshold.
    
    In this context, since more negative scores are considered more signal-like,
    FAR is computed as the number of events with scores <= threshold divided by the total background time.
    
    Parameters:
        threshold (float): The score threshold.
        scores (np.ndarray): 1D array of background scores.
        total_time (float): Total background time in seconds.
        direction (str): 'negative' (default) uses <= threshold; use 'positive' for >= threshold.
    
    Returns:
        float: FAR (events per second).
    """
    if direction == 'negative':
        count = np.sum(scores <= threshold)
    else:
        count = np.sum(scores >= threshold)
    return count / total_time

def calculate_far_metrics(merged_array, duration, num_thresholds=10, direction='negative'):
    """
    Calculate FAR metrics over a range of thresholds from the merged HDF5 array.
    
    Parameters:
        merged_array (np.ndarray): Merged array from HDF5 files of shape (N, M, 1, 1).
        duration (float): Duration (in seconds) per score entry.
        num_thresholds (int): Number of thresholds to evaluate (default: 10).
        direction (str): 'negative' if lower scores are more signal-like (default), or 'positive'.
    
    Returns:
        np.ndarray: A NumPy array of shape (num_thresholds, 2) where column 0 is threshold and column 1 is FAR.
    """
    # Determine total background time from merged array shape (N, M, 1, 1)
    n, m, _, _ = merged_array.shape
    total_time = n * m * duration

    # Flatten the array to 1D scores
    scores = merged_array.flatten()

    # Generate evenly spaced thresholds from min to max score
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)
    metrics = []
    for thresh in thresholds:
        far = score_to_far(thresh, scores, total_time, direction=direction)
        metrics.append((thresh, far))
    
    return np.array(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge HDF5 files, compute FAR metrics, and save the conversion as a NumPy array.'
    )
    parser.add_argument('folder', type=str, help='Path to the folder containing .h5 files.')
    parser.add_argument('--dataset', type=str, default='data', help='Dataset key to extract from each file (default: "data").')
    parser.add_argument('--duration', type=float, default=0.5, help='Duration (in seconds) per score entry (default: 0.5).')
    parser.add_argument('--num_thresholds', type=int, default=100, help='Number of thresholds for FAR metrics (default: 100).')
    parser.add_argument('--outfile', type=str, default='far_metrics.npy', help='Output file name for the NumPy array (default: far_metrics.npy).')
    parser.add_argument('--direction', type=str, default='negative', choices=['negative', 'positive'],
                        help='Scoring direction; use "negative" if more negative scores are signal-like (default).')
    
    args = parser.parse_args()
    
    # Merge HDF5 files
    merged_array = merge_h5_files(args.folder, args.dataset)
    print('Merged array shape:', merged_array.shape)
    
    # Calculate FAR metrics and save them as a NumPy array.
    metrics_array = calculate_far_metrics(merged_array, args.duration, args.num_thresholds, direction=args.direction)
    np.save(args.outfile, metrics_array)
    print(f"FAR metrics saved to {args.outfile}")

    # Example output of the metrics array:
    print("Threshold\tFAR (per second)")
    for thresh, far in metrics_array:
        print(f"{thresh:.4f}\t\t{far:.4e}")

    # FOR KATYA Example usage:
    # python save_far.py /path/to/h5_files_from_infer --dataset data --duration 0.5 --num_thresholds 100 --outfile far_metrics.npy --direction positive
