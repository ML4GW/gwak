#!/usr/bin/env python3
"""
Test script to compare computed embeddings with reference embeddings.

Compares mean and standard deviation of each embedding dimension to ensure
the embedding computation is working correctly.
"""

import argparse
import h5py
import numpy as np
import sys
from pathlib import Path


def load_computed_embeddings(h5_path):
    """Load embeddings from HDF5 file"""
    print(f"Loading computed embeddings from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]

    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    return embeddings


def load_reference_embeddings(npy_path):
    """Load reference embeddings from .npy file"""
    print(f"Loading reference embeddings from: {npy_path}")

    embeddings = np.load(npy_path)

    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    return embeddings


def compute_statistics(embeddings, name="Embeddings"):
    """Compute mean and std for each dimension"""
    print(f"\n{name} statistics:")

    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)

    print(f"  Mean per dimension: min={mean.min():.6f}, max={mean.max():.6f}, avg={mean.mean():.6f}")
    print(f"  Std per dimension:  min={std.min():.6f}, max={std.max():.6f}, avg={std.mean():.6f}")

    # Overall statistics
    print(f"  Overall mean: {embeddings.mean():.6f}")
    print(f"  Overall std:  {embeddings.std():.6f}")
    print(f"  Overall min:  {embeddings.min():.6f}")
    print(f"  Overall max:  {embeddings.max():.6f}")

    return mean, std


def compare_statistics(mean1, std1, mean2, std2, name1="Computed", name2="Reference"):
    """Compare statistics between two sets of embeddings"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")

    # Mean comparison
    mean_diff = mean1 - mean2
    mean_rel_diff = mean_diff / (np.abs(mean2) + 1e-10)  # Avoid division by zero

    print(f"\nMean differences:")
    print(f"  Absolute difference:")
    print(f"    Min:  {mean_diff.min():.6f}")
    print(f"    Max:  {mean_diff.max():.6f}")
    print(f"    Mean: {mean_diff.mean():.6f}")
    print(f"    Std:  {mean_diff.std():.6f}")

    print(f"  Relative difference (%):")
    print(f"    Min:  {mean_rel_diff.min()*100:.2f}%")
    print(f"    Max:  {mean_rel_diff.max()*100:.2f}%")
    print(f"    Mean: {mean_rel_diff.mean()*100:.2f}%")
    print(f"    Std:  {mean_rel_diff.std()*100:.2f}%")

    # Std comparison
    std_diff = std1 - std2
    std_rel_diff = std_diff / (np.abs(std2) + 1e-10)

    print(f"\nStd differences:")
    print(f"  Absolute difference:")
    print(f"    Min:  {std_diff.min():.6f}")
    print(f"    Max:  {std_diff.max():.6f}")
    print(f"    Mean: {std_diff.mean():.6f}")
    print(f"    Std:  {std_diff.std():.6f}")

    print(f"  Relative difference (%):")
    print(f"    Min:  {std_rel_diff.min()*100:.2f}%")
    print(f"    Max:  {std_rel_diff.max()*100:.2f}%")
    print(f"    Mean: {std_rel_diff.mean()*100:.2f}%")
    print(f"    Std:  {std_rel_diff.std()*100:.2f}%")

    # Correlation between means and stds
    mean_corr = np.corrcoef(mean1, mean2)[0, 1]
    std_corr = np.corrcoef(std1, std2)[0, 1]

    print(f"\nCorrelations:")
    print(f"  Mean correlation: {mean_corr:.6f}")
    print(f"  Std correlation:  {std_corr:.6f}")

    # Check if they're "similar" (within reasonable tolerance)
    mean_close = np.allclose(mean1, mean2, rtol=0.1, atol=0.01)  # 10% relative or 0.01 absolute
    std_close = np.allclose(std1, std2, rtol=0.1, atol=0.01)

    print(f"\nSimilarity check (rtol=0.1, atol=0.01):")
    print(f"  Means are close: {mean_close}")
    print(f"  Stds are close:  {std_close}")

    return mean_close and std_close


def print_dimension_comparison(mean1, std1, mean2, std2, num_dims=10):
    """Print detailed comparison for first few dimensions"""
    print(f"\n{'='*80}")
    print(f"DIMENSION-BY-DIMENSION COMPARISON (first {num_dims} dimensions)")
    print(f"{'='*80}")
    print(f"{'Dim':<5} {'Comp Mean':<12} {'Ref Mean':<12} {'Diff':<12} {'Comp Std':<12} {'Ref Std':<12} {'Diff':<12}")
    print(f"{'-'*80}")

    for i in range(min(num_dims, len(mean1))):
        mean_diff = mean1[i] - mean2[i]
        std_diff = std1[i] - std2[i]
        print(f"{i:<5} {mean1[i]:>11.6f} {mean2[i]:>11.6f} {mean_diff:>11.6f} {std1[i]:>11.6f} {std2[i]:>11.6f} {std_diff:>11.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare computed embeddings with reference embeddings'
    )

    parser.add_argument(
        '--computed',
        type=Path,
        help='Path to computed embeddings (HDF5 or .npy file)'
    )

    parser.add_argument(
        '--reference',
        type=Path,
        default=Path('/home/katya.govorkova/gwak2/gwak/output/plots/ResNet_NF_from_file_conditioning_HL_precomputed/embeddings_50000.npy'),
        help='Path to reference embeddings (.npy file)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to compare (default: all available, limited by smaller dataset)'
    )

    parser.add_argument(
        '--num-dims',
        type=int,
        default=10,
        help='Number of dimensions to show in detail (default: 10)'
    )

    args = parser.parse_args()

    # Auto-detect computed embeddings if not specified
    if args.computed is None:
        # Try to find the most recent embeddings
        possible_paths = [
            Path('gwak/output/embeddings/o4_test_embeddings.h5'),
            Path('gwak/output/embeddings/bbc_background_embeddings.h5'),
        ]

        for path in possible_paths:
            if path.exists():
                args.computed = path
                print(f"Auto-detected computed embeddings: {args.computed}")
                break

        if args.computed is None:
            print("ERROR: Could not find computed embeddings. Please specify with --computed")
            sys.exit(1)

    # Load embeddings
    if args.computed.suffix == '.h5':
        computed_emb = load_computed_embeddings(args.computed)
    else:
        computed_emb = load_reference_embeddings(args.computed)

    reference_emb = load_reference_embeddings(args.reference)

    # Check dimensions match
    if computed_emb.shape[1] != reference_emb.shape[1]:
        print(f"\nWARNING: Embedding dimensions don't match!")
        print(f"  Computed: {computed_emb.shape[1]}")
        print(f"  Reference: {reference_emb.shape[1]}")
        print(f"  Using minimum dimension size for comparison")
        min_dim = min(computed_emb.shape[1], reference_emb.shape[1])
        computed_emb = computed_emb[:, :min_dim]
        reference_emb = reference_emb[:, :min_dim]

    # Limit number of samples if requested
    n_computed = len(computed_emb)
    n_reference = len(reference_emb)

    if args.num_samples is not None:
        n_samples = min(args.num_samples, n_computed, n_reference)
    else:
        n_samples = min(n_computed, n_reference)

    print(f"\nComparing {n_samples} samples from each dataset")

    # Use first n_samples from each
    computed_emb = computed_emb[:n_samples]
    reference_emb = reference_emb[:n_samples]

    # Compute statistics
    comp_mean, comp_std = compute_statistics(computed_emb, "Computed embeddings")
    ref_mean, ref_std = compute_statistics(reference_emb, "Reference embeddings")

    # Compare
    similar = compare_statistics(comp_mean, comp_std, ref_mean, ref_std)

    # Print detailed dimension comparison
    print_dimension_comparison(comp_mean, comp_std, ref_mean, ref_std, num_dims=args.num_dims)

    # Final verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}")
    if similar:
        print("✓ Embeddings are SIMILAR - statistics match within tolerance")
        sys.exit(0)
    else:
        print("✗ Embeddings are DIFFERENT - statistics differ beyond tolerance")
        print("  This could indicate:")
        print("  - Different preprocessing")
        print("  - Different model versions")
        print("  - Different data sources")
        print("  - Bugs in the embedding computation")
        sys.exit(1)


if __name__ == '__main__':
    main()
