#!/usr/bin/env python3
"""
Plot cWB Metric Distributions with Veto Thresholds

Reads the output CSV from cwb_veto_simple.py and creates histograms
for each of the 8 metrics with veto thresholds shown as vertical lines.

Features:
- Recalculates pass/fail at runtime based on thresholds (allows testing new thresholds)
- Plots distributions of all 8 cWB metrics with veto lines
- Plots GWAK value distribution for events that pass all vetoes
- Generates both combined dashboard and individual plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def recalculate_vetoes(df, thresholds):
    """
    Recalculate veto decisions based on current thresholds.

    This allows using different thresholds than what was used
    to generate the original results.csv.

    Parameters
    ----------
    df : DataFrame
        Results dataframe with metric values
    thresholds : dict
        Threshold dictionary from thresholds.json

    Returns
    -------
    DataFrame with updated 'vetoed_recalc' column
    """
    df = df.copy()

    metrics = ["rho", "edr", "cc", "scc", "coh_max", "coh_mean",
               "corr_mag_fd", "corr_real_fd"]

    # Initialize as not vetoed
    df['vetoed_recalc'] = False

    for metric in metrics:
        if metric not in df.columns:
            continue

        metric_values = df[metric].values
        thresh_info = thresholds[metric]

        min_thresh = thresh_info.get("min")
        max_thresh = thresh_info.get("max")

        # Apply min threshold
        if min_thresh is not None:
            fails_min = (~np.isfinite(metric_values)) | (metric_values < min_thresh)
            df.loc[fails_min, 'vetoed_recalc'] = True

        # Apply max threshold
        if max_thresh is not None:
            fails_max = (~np.isfinite(metric_values)) | (metric_values > max_thresh)
            df.loc[fails_max, 'vetoed_recalc'] = True

    return df


def plot_metric_distributions(results_csv, thresholds_json, output_dir=None,
                              bins=50, figsize=(15, 10)):
    """
    Plot distributions of all 8 cWB metrics with veto thresholds.

    Parameters
    ----------
    results_csv : Path
        Path to results CSV from cwb_veto_simple.py
    thresholds_json : Path
        Path to thresholds.json
    output_dir : Path, optional
        Directory to save plots (if None, just shows them)
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size (width, height)
    """
    # Load data
    df = pd.read_csv(results_csv)

    with open(thresholds_json, "r") as f:
        thresholds = json.load(f)

    # Filter to successful segments only
    df_success = df[df["success"] == True].copy()

    if len(df_success) == 0:
        print("ERROR: No successfully processed segments found!")
        return

    # Recalculate vetoes based on current thresholds
    df_success = recalculate_vetoes(df_success, thresholds)

    print(f"Loaded {len(df_success)} successfully processed segments")
    print(f"Vetoed (recalculated): {df_success['vetoed_recalc'].sum()}")
    print(f"Passed (recalculated): {(~df_success['vetoed_recalc']).sum()}")

    # Define the 8 metrics
    metrics = ["rho", "edr", "cc", "scc", "coh_max", "coh_mean",
               "corr_mag_fd", "corr_real_fd"]

    # Create subplot grid (3 rows x 3 columns for 8 metrics + GWAK values)
    fig, axes = plt.subplots(3, 3, figsize=(figsize[0], figsize[1] * 1.2))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get metric values (filter out NaN/inf)
        values = df_success[metric].values
        finite_mask = np.isfinite(values)
        values_finite = values[finite_mask]

        if len(values_finite) == 0:
            ax.text(0.5, 0.5, f"No finite values\nfor {metric}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric)
            continue

        # Plot histogram
        ax.hist(values_finite, bins=bins, alpha=0.7, edgecolor='black',
               color='steelblue', label='Data')

        # Get threshold info
        thresh_info = thresholds[metric]
        min_thresh = thresh_info.get("min")
        max_thresh = thresh_info.get("max")

        # Plot threshold lines
        ylim = ax.get_ylim()

        if min_thresh is not None:
            ax.axvline(min_thresh, color='red', linestyle='--',
                      linewidth=2, label=f'Min: {min_thresh:.4f}')
            # Shade vetoed region (below min)
            ax.axvspan(values_finite.min(), min_thresh,
                      alpha=0.2, color='red')

        if max_thresh is not None:
            ax.axvline(max_thresh, color='darkred', linestyle='--',
                      linewidth=2, label=f'Max: {max_thresh:.4f}')
            # Shade vetoed region (above max)
            ax.axvspan(max_thresh, values_finite.max(),
                      alpha=0.2, color='red')

        # Labels and legend
        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')

        # Add legend if thresholds exist
        if min_thresh is not None or max_thresh is not None:
            ax.legend(loc='best', fontsize=9)

        # Add statistics text
        stats_text = (f'Mean: {np.mean(values_finite):.4f}\n'
                     f'Median: {np.median(values_finite):.4f}\n'
                     f'Std: {np.std(values_finite):.4f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot GWAK values for PASSING events (9th subplot)
    ax_gwak = axes[8]

    # Get passing events only
    df_passed = df_success[~df_success['vetoed_recalc']].copy()

    if len(df_passed) > 0 and 'gwak_value' in df_passed.columns:
        gwak_values = df_passed['gwak_value'].values
        finite_gwak = gwak_values[np.isfinite(gwak_values)]

        if len(finite_gwak) > 0:
            ax_gwak.hist(finite_gwak, bins=bins, alpha=0.7, edgecolor='black',
                        color='green', label='Passed events')

            ax_gwak.set_xlabel('GWAK Value', fontsize=11)
            ax_gwak.set_ylabel('Count', fontsize=11)
            ax_gwak.set_title('GWAK Values (Passing Events Only)',
                             fontsize=12, fontweight='bold')

            # Statistics
            gwak_stats = (f'N = {len(finite_gwak)}\n'
                         f'Mean: {np.mean(finite_gwak):.4f}\n'
                         f'Median: {np.median(finite_gwak):.4f}\n'
                         f'Std: {np.std(finite_gwak):.4f}\n'
                         f'Min: {np.min(finite_gwak):.4f}\n'
                         f'Max: {np.max(finite_gwak):.4f}')
            ax_gwak.text(0.02, 0.98, gwak_stats, transform=ax_gwak.transAxes,
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            ax_gwak.text(0.5, 0.5, 'No finite GWAK values\nfor passing events',
                        ha='center', va='center', transform=ax_gwak.transAxes)
            ax_gwak.set_title('GWAK Values (Passing Events Only)',
                             fontsize=12, fontweight='bold')
    else:
        ax_gwak.text(0.5, 0.5, f'No passing events\n(N={len(df_passed)})',
                    ha='center', va='center', transform=ax_gwak.transAxes, fontsize=12)
        ax_gwak.set_title('GWAK Values (Passing Events Only)',
                         fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save or show
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "cwb_metric_distributions.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()

    return fig, axes


def plot_individual_metrics(results_csv, thresholds_json, output_dir,
                           bins=50, figsize=(10, 6)):
    """
    Create individual plots for each metric (8 separate files + GWAK plot).

    Parameters
    ----------
    results_csv : Path
        Path to results CSV
    thresholds_json : Path
        Path to thresholds.json
    output_dir : Path
        Directory to save individual plots
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size for each plot
    """
    # Load data
    df = pd.read_csv(results_csv)

    with open(thresholds_json, "r") as f:
        thresholds = json.load(f)

    # Filter to successful segments
    df_success = df[df["success"] == True].copy()

    if len(df_success) == 0:
        print("ERROR: No successfully processed segments found!")
        return

    # Recalculate vetoes based on current thresholds
    df_success = recalculate_vetoes(df_success, thresholds)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["rho", "edr", "cc", "scc", "coh_max", "coh_mean",
               "corr_mag_fd", "corr_real_fd"]

    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get metric values
        values = df_success[metric].values
        finite_mask = np.isfinite(values)
        values_finite = values[finite_mask]

        if len(values_finite) == 0:
            print(f"WARNING: No finite values for {metric}, skipping...")
            plt.close(fig)
            continue

        # Plot histogram
        ax.hist(values_finite, bins=bins, alpha=0.7, edgecolor='black',
               color='steelblue', label='Data')

        # Get thresholds
        thresh_info = thresholds[metric]
        min_thresh = thresh_info.get("min")
        max_thresh = thresh_info.get("max")
        description = thresh_info.get("description", "")

        # Plot threshold lines
        ylim = ax.get_ylim()

        if min_thresh is not None:
            ax.axvline(min_thresh, color='red', linestyle='--',
                      linewidth=2.5, label=f'Min threshold: {min_thresh:.6f}')
            ax.axvspan(values_finite.min(), min_thresh,
                      alpha=0.15, color='red', label='Vetoed region')

        if max_thresh is not None:
            ax.axvline(max_thresh, color='darkred', linestyle='--',
                      linewidth=2.5, label=f'Max threshold: {max_thresh:.6f}')
            ax.axvspan(max_thresh, values_finite.max(),
                      alpha=0.15, color='red')

        # Labels
        ax.set_xlabel(f'{metric} Value', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title(f'{metric.upper()} Distribution\n{description}',
                    fontsize=14, fontweight='bold')

        # Legend
        ax.legend(loc='best', fontsize=11)

        # Statistics
        stats_text = (f'N = {len(values_finite)}\n'
                     f'Mean = {np.mean(values_finite):.6f}\n'
                     f'Median = {np.median(values_finite):.6f}\n'
                     f'Std = {np.std(values_finite):.6f}\n'
                     f'Min = {np.min(values_finite):.6f}\n'
                     f'Max = {np.max(values_finite):.6f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        # Save
        output_file = output_dir / f"{metric}_distribution.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close(fig)

    # Plot GWAK values for passing events
    df_passed = df_success[~df_success['vetoed_recalc']].copy()

    if len(df_passed) > 0 and 'gwak_value' in df_passed.columns:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        gwak_values = df_passed['gwak_value'].values
        finite_gwak = gwak_values[np.isfinite(gwak_values)]

        if len(finite_gwak) > 0:
            ax.hist(finite_gwak, bins=bins, alpha=0.7, edgecolor='black',
                   color='green', label='Passing events')

            ax.set_xlabel('GWAK Value', fontsize=13)
            ax.set_ylabel('Count', fontsize=13)
            ax.set_title('GWAK Value Distribution (Passing Events Only)',
                        fontsize=14, fontweight='bold')

            ax.legend(loc='best', fontsize=11)

            # Statistics
            gwak_stats = (f'N = {len(finite_gwak)}\n'
                         f'Mean = {np.mean(finite_gwak):.6f}\n'
                         f'Median = {np.median(finite_gwak):.6f}\n'
                         f'Std = {np.std(finite_gwak):.6f}\n'
                         f'Min = {np.min(finite_gwak):.6f}\n'
                         f'Max = {np.max(finite_gwak):.6f}')
            ax.text(0.98, 0.98, gwak_stats, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

            plt.tight_layout()

            output_file = output_dir / "gwak_value_distribution_passing.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close(fig)
        else:
            print("WARNING: No finite GWAK values for passing events")
            plt.close(fig)
    else:
        print(f"WARNING: No passing events (N={len(df_passed)}) or 'gwak_value' column missing")


def print_summary_statistics(results_csv, thresholds_json):
    """Print summary statistics for all metrics with recalculated vetoes."""
    df = pd.read_csv(results_csv)

    with open(thresholds_json, "r") as f:
        thresholds = json.load(f)

    df_success = df[df["success"] == True].copy()

    # Recalculate vetoes based on current thresholds
    df_success = recalculate_vetoes(df_success, thresholds)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS (using recalculated vetoes)")
    print("="*80)

    metrics = ["rho", "edr", "cc", "scc", "coh_max", "coh_mean",
               "corr_mag_fd", "corr_real_fd"]

    for metric in metrics:
        values = df_success[metric].values
        finite_mask = np.isfinite(values)
        values_finite = values[finite_mask]

        thresh_info = thresholds[metric]
        min_thresh = thresh_info.get("min")
        max_thresh = thresh_info.get("max")

        # Count vetoes for this metric
        n_below_min = np.sum(values_finite < min_thresh) if min_thresh is not None else 0
        n_above_max = np.sum(values_finite > max_thresh) if max_thresh is not None else 0
        n_vetoed = n_below_min + n_above_max

        print(f"\n{metric.upper()}:")
        print(f"  N (finite): {len(values_finite)}")
        print(f"  Mean: {np.mean(values_finite):.6f}")
        print(f"  Median: {np.median(values_finite):.6f}")
        print(f"  Std: {np.std(values_finite):.6f}")
        print(f"  Min: {np.min(values_finite):.6f}")
        print(f"  Max: {np.max(values_finite):.6f}")

        if min_thresh is not None:
            print(f"  Min threshold: {min_thresh:.6f} (vetoes {n_below_min} segments)")
        if max_thresh is not None:
            print(f"  Max threshold: {max_thresh:.6f} (vetoes {n_above_max} segments)")

        print(f"  Vetoed by this metric: {n_vetoed}/{len(values_finite)}")

    print("\n" + "="*80)
    print(f"OVERALL: {df_success['vetoed_recalc'].sum()}/{len(df_success)} segments vetoed")
    print(f"PASSED: {(~df_success['vetoed_recalc']).sum()}/{len(df_success)} segments")

    # GWAK value statistics for passing events
    df_passed = df_success[~df_success['vetoed_recalc']].copy()
    if len(df_passed) > 0 and 'gwak_value' in df_passed.columns:
        gwak_values = df_passed['gwak_value'].values
        finite_gwak = gwak_values[np.isfinite(gwak_values)]
        if len(finite_gwak) > 0:
            print("\nGWAK VALUES (passing events only):")
            print(f"  N: {len(finite_gwak)}")
            print(f"  Mean: {np.mean(finite_gwak):.6f}")
            print(f"  Median: {np.median(finite_gwak):.6f}")
            print(f"  Std: {np.std(finite_gwak):.6f}")
            print(f"  Min: {np.min(finite_gwak):.6f}")
            print(f"  Max: {np.max(finite_gwak):.6f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cWB metric distributions with veto thresholds"
    )
    parser.add_argument("--results", type=Path, required=True,
                       help="Path to results CSV from cwb_veto_simple.py")
    parser.add_argument("--thresholds", type=Path, required=True,
                       help="Path to thresholds.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Directory to save plots (default: show only)")
    parser.add_argument("--individual", action="store_true",
                       help="Also create individual plots for each metric")
    parser.add_argument("--bins", type=int, default=50,
                       help="Number of histogram bins (default: 50)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Don't print summary statistics")

    args = parser.parse_args()

    # Check inputs exist
    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        return

    if not args.thresholds.exists():
        print(f"ERROR: Thresholds file not found: {args.thresholds}")
        return

    # Print summary statistics
    if not args.no_stats:
        print_summary_statistics(args.results, args.thresholds)

    # Create combined plot
    print("\nCreating combined distribution plot...")
    plot_metric_distributions(args.results, args.thresholds,
                             output_dir=args.output_dir, bins=args.bins)

    # Create individual plots if requested
    if args.individual:
        if args.output_dir is None:
            print("\nWARNING: --individual requires --output-dir, skipping individual plots")
        else:
            print("\nCreating individual metric plots...")
            plot_individual_metrics(args.results, args.thresholds,
                                   args.output_dir, bins=args.bins)

    print("\nDone!")


if __name__ == "__main__":
    main()
