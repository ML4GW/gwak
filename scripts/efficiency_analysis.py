#!/usr/bin/env python3
"""
Calculate signal detection efficiencies from embedding triggers.

This script:
1. Converts embedding indices to GPS times
2. Matches triggers to injections
3. Calculates detection efficiency vs SNR for each signal type
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_triggers(trigger_indices, gps_start, segment_duration=1.0):
    """
    Convert trigger indices to GPS times.

    Args:
        trigger_indices: Array of embedding indices (0, 1, 2, ...)
        gps_start: GPS start time of the data
        segment_duration: Duration of each segment in seconds (default: 1.0)

    Returns:
        Array of GPS times
    """
    return gps_start + trigger_indices * segment_duration


def load_injections(injection_file):
    """
    Load all injection parameters from the injection file.

    Args:
        injection_file: Path to injection h5 file

    Returns:
        Dictionary with signal types as keys, each containing:
        - 'times': GPS times of injections
        - 'parameters': Full parameter arrays
        - 'group_attrs': Group attributes (type, hrss_norm, etc.)
    """
    injections = {}

    with h5py.File(injection_file, 'r') as f:
        for signal_name in f.keys():
            if isinstance(f[signal_name], h5py.Group):
                # Load parameters
                if 'PARAMETERS' in f[signal_name]:
                    params = f[signal_name]['PARAMETERS'][:]

                    # Get group attributes
                    attrs = dict(f[signal_name].attrs)

                    injections[signal_name] = {
                        'times': params['time'],
                        'parameters': params,
                        'group_attrs': attrs,
                        'signal_type': attrs.get('type', signal_name)
                    }

    return injections


def match_triggers_to_injections(trigger_times, injection_times, time_window=0.5):
    """
    Match trigger times to injection times.

    Args:
        trigger_times: Array of GPS times for triggers
        injection_times: Array of GPS times for injections
        time_window: Time window for matching (seconds)

    Returns:
        Array of booleans indicating which injections were recovered
    """
    recovered = np.zeros(len(injection_times), dtype=bool)

    for i, inj_time in enumerate(injection_times):
        # Check if any trigger falls within time_window of this injection
        time_diffs = np.abs(trigger_times - inj_time)
        if np.any(time_diffs <= time_window):
            recovered[i] = True

    return recovered


def calculate_snr(parameters, signal_type):
    """
    Calculate or extract SNR from parameters.

    For many signals, amplitude is proportional to SNR.
    This is a simplified version - you may need to adjust based on your signal types.

    Args:
        parameters: Parameter array from injections
        signal_type: Type of signal

    Returns:
        Array of SNR values
    """
    # Use amplitude as proxy for SNR
    # You may want to calculate actual SNR based on signal type
    amplitudes = parameters['amplitude']

    # Convert amplitude to approximate SNR
    # This is signal-type dependent; adjust as needed
    if signal_type in ['SG', 'WNB', 'GA']:
        # For burst signals, amplitude often relates to hrss
        # You might need to scale this appropriately
        snr = amplitudes / 1e-22  # Normalize by hrss_norm
    else:
        # For CBC and others
        snr = amplitudes

    return snr


def calculate_efficiency_vs_snr(recovered, snr_values, snr_bins=None):
    """
    Calculate detection efficiency as a function of SNR.

    Args:
        recovered: Boolean array indicating which injections were recovered
        snr_values: SNR values for each injection
        snr_bins: SNR bin edges (if None, auto-generate)

    Returns:
        bin_centers, efficiency, efficiency_error, counts_per_bin
    """
    if snr_bins is None:
        # Auto-generate logarithmic bins
        snr_min = max(0.1, np.min(snr_values[snr_values > 0]))
        snr_max = np.max(snr_values)
        snr_bins = np.logspace(np.log10(snr_min), np.log10(snr_max), 15)

    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    efficiency = np.zeros(len(bin_centers))
    efficiency_error = np.zeros(len(bin_centers))
    counts = np.zeros(len(bin_centers))

    for i in range(len(bin_centers)):
        in_bin = (snr_values >= snr_bins[i]) & (snr_values < snr_bins[i+1])
        n_total = np.sum(in_bin)

        if n_total > 0:
            n_recovered = np.sum(recovered[in_bin])
            efficiency[i] = n_recovered / n_total
            # Binomial error
            efficiency_error[i] = np.sqrt(efficiency[i] * (1 - efficiency[i]) / n_total)
            counts[i] = n_total

    return bin_centers, efficiency, efficiency_error, counts


def analyze_triggers(
    trigger_indices,
    gps_start,
    injection_file,
    time_window=0.5,
    output_dir='.',
    segment_duration=1.0
):
    """
    Full analysis: match triggers to injections and calculate efficiencies.

    Args:
        trigger_indices: Array of embedding indices that triggered
        gps_start: GPS start time of the embedding data
        injection_file: Path to injection h5 file
        time_window: Time window for matching triggers to injections (seconds)
        output_dir: Directory to save plots
        segment_duration: Duration of each embedding segment (seconds)

    Returns:
        Dictionary of results per signal type
    """
    # Convert triggers to GPS times
    trigger_times = load_triggers(trigger_indices, gps_start, segment_duration)
    print(f"Loaded {len(trigger_times)} triggers")
    print(f"GPS time range: {trigger_times.min():.1f} - {trigger_times.max():.1f}")

    # Load injections
    injections = load_injections(injection_file)
    print(f"\nLoaded {len(injections)} signal types")

    # Match and calculate efficiencies for each signal type
    results = {}

    for signal_name, inj_data in injections.items():
        print(f"\nProcessing {signal_name}...")

        # Match triggers to injections
        recovered = match_triggers_to_injections(
            trigger_times,
            inj_data['times'],
            time_window
        )

        # Calculate SNR
        snr = calculate_snr(inj_data['parameters'], inj_data['signal_type'])

        # Calculate efficiency vs SNR
        bin_centers, efficiency, eff_error, counts = calculate_efficiency_vs_snr(
            recovered, snr
        )

        n_recovered = np.sum(recovered)
        n_total = len(recovered)

        print(f"  Total injections: {n_total}")
        print(f"  Recovered: {n_recovered} ({100*n_recovered/n_total:.1f}%)")

        results[signal_name] = {
            'recovered': recovered,
            'snr': snr,
            'bin_centers': bin_centers,
            'efficiency': efficiency,
            'efficiency_error': eff_error,
            'counts': counts,
            'signal_type': inj_data['signal_type'],
            'n_total': n_total,
            'n_recovered': n_recovered
        }

    return results


def plot_efficiency_curves(results, output_file='efficiency_vs_snr.png'):
    """
    Plot efficiency vs SNR for all signal types.

    Args:
        results: Dictionary from analyze_triggers()
        output_file: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Group by signal type
    signal_types = {}
    for signal_name, data in results.items():
        sig_type = data['signal_type']
        if sig_type not in signal_types:
            signal_types[sig_type] = []
        signal_types[sig_type].append((signal_name, data))

    # Plot efficiency curves
    for sig_type, signals in signal_types.items():
        for signal_name, data in signals:
            # Only plot if we have data
            mask = data['counts'] > 0
            if np.any(mask):
                ax1.errorbar(
                    data['bin_centers'][mask],
                    data['efficiency'][mask],
                    yerr=data['efficiency_error'][mask],
                    marker='o',
                    label=f"{signal_name} ({data['n_recovered']}/{data['n_total']})",
                    alpha=0.7
                )

    ax1.set_xlabel('SNR (or Amplitude)', fontsize=12)
    ax1.set_ylabel('Detection Efficiency', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_title('Detection Efficiency vs SNR')

    # Plot total recovery per signal type
    signal_type_stats = defaultdict(lambda: {'recovered': 0, 'total': 0})
    for sig_type, signals in signal_types.items():
        for signal_name, data in signals:
            signal_type_stats[sig_type]['recovered'] += data['n_recovered']
            signal_type_stats[sig_type]['total'] += data['n_total']

    types = list(signal_type_stats.keys())
    efficiencies = [
        signal_type_stats[t]['recovered'] / signal_type_stats[t]['total']
        for t in types
    ]
    totals = [signal_type_stats[t]['total'] for t in types]

    bars = ax2.bar(range(len(types)), efficiencies, alpha=0.7)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, rotation=45, ha='right')
    ax2.set_ylabel('Overall Efficiency', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Overall Efficiency by Signal Type')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{int(height*total)}/{total}',
            ha='center', va='bottom', fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate signal detection efficiencies from embedding triggers'
    )

    parser.add_argument(
        '--triggers',
        type=str,
        required=True,
        help='Path to .npy file with trigger indices or CSV file with trigger times'
    )

    parser.add_argument(
        '--gps-start',
        type=float,
        required=True,
        help='GPS start time of the embedding data'
    )

    parser.add_argument(
        '--injection-file',
        type=str,
        required=True,
        help='Path to injection h5 file'
    )

    parser.add_argument(
        '--time-window',
        type=float,
        default=0.5,
        help='Time window for matching triggers to injections (seconds)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='efficiency_vs_snr.png',
        help='Output plot filename'
    )

    parser.add_argument(
        '--segment-duration',
        type=float,
        default=1.0,
        help='Duration of each embedding segment (seconds)'
    )

    args = parser.parse_args()

    # Load trigger indices
    trigger_file = Path(args.triggers)
    if trigger_file.suffix == '.npy':
        trigger_indices = np.load(trigger_file)
    else:
        # Assume CSV or text file
        trigger_indices = np.loadtxt(trigger_file)

    # Run analysis
    results = analyze_triggers(
        trigger_indices=trigger_indices,
        gps_start=args.gps_start,
        injection_file=args.injection_file,
        time_window=args.time_window,
        segment_duration=args.segment_duration
    )

    # Plot results
    plot_efficiency_curves(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for signal_name, data in sorted(results.items()):
        eff = data['n_recovered'] / data['n_total']
        print(f"{signal_name:40s}: {data['n_recovered']:4d}/{data['n_total']:4d} ({100*eff:5.1f}%)")
