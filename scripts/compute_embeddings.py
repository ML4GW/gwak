#!/usr/bin/env python3
"""
Compute embeddings using ResNet model on BBC background and O4_test signal data.

This script:
1. Loads the JIT-compiled ResNet embedding model
2. Processes continuous strain data from h5 files
3. Applies whitening and bandpassing (same as training dataloader)
4. Segments data into 1-second windows
5. Computes embeddings for each window
6. Saves embeddings efficiently to HDF5 files
"""

import argparse
import h5py
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add gwak to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml4gw.transforms import SpectralDensity, Whiten
from gwak.train.dataloader import TorchBandpassFIR


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(model_path, device):
    """Load the JIT-compiled model"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path}")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model


def get_preprocessing_transforms(sample_rate, fduration, fftlength, device):
    """
    Create preprocessing transforms matching the training dataloader.

    Args:
        sample_rate: Sampling rate (typically 4096 Hz)
        fduration: Duration for PSD estimation
        fftlength: FFT length for spectral density
        device: torch device

    Returns:
        bandpass, whitener, spectral_density transforms
    """
    bandpass = TorchBandpassFIR(
        lowcut=30,
        highcut=2047,
        sample_rate=sample_rate
    ).to(device)

    whitener = Whiten(
        fduration,
        sample_rate,
        highpass=30,
    ).to(device)

    spectral_density = SpectralDensity(
        sample_rate,
        fftlength,
        average='median'
    ).to(device)

    return bandpass, whitener, spectral_density


def load_valid_times(h5_path):
    """
    Load valid clean segment times from corresponding .npy file.

    Args:
        h5_path: Path to the h5 file

    Returns:
        Array of valid start times, or None if file doesn't exist
    """
    npy_path = Path(str(h5_path).replace('.h5', '_clean_valid.npy'))

    if npy_path.exists():
        return np.load(npy_path)
    return None


def read_strain_data(h5_path, ifos=['H1', 'L1']):
    """
    Read strain data from h5 file.

    Args:
        h5_path: Path to h5 file
        ifos: List of interferometer names

    Returns:
        strain_data: Dictionary with ifo names as keys
        start_time: GPS start time
        sample_rate: Sampling rate
    """
    logger = logging.getLogger(__name__)

    with h5py.File(h5_path, 'r') as f:
        strain_data = {}

        for ifo in ifos:
            if ifo in f:
                strain_data[ifo] = f[ifo][:]
            else:
                logger.warning(f"IFO {ifo} not found in {h5_path}")

        # Try to get metadata
        start_time = None
        sample_rate = None

        # Common metadata locations
        for ifo in ifos:
            if ifo in f and hasattr(f[ifo], 'attrs'):
                if 'start_time' in f[ifo].attrs:
                    start_time = f[ifo].attrs['start_time']
                if 'sample_rate' in f[ifo].attrs:
                    sample_rate = f[ifo].attrs['sample_rate']
                if start_time is not None and sample_rate is not None:
                    break

    return strain_data, start_time, sample_rate


def preprocess_data(data, psd_data, bandpass, whitener, spectral_density, device):
    """
    Apply bandpassing and whitening to data.

    Args:
        data: Tensor of shape [batch_size, num_ifos, kernel_length * sample_rate]
        psd_data: Tensor for PSD estimation [batch_size, num_ifos, psd_length]
        bandpass: Bandpass filter
        whitener: Whitening transform
        spectral_density: PSD estimator
        device: torch device

    Returns:
        Whitened data
    """
    # Apply bandpass filter
    data = bandpass(data)

    # Calculate PSDs
    psds = spectral_density(psd_data.double())

    # Whiten the data
    whitened = whitener(data.double(), psds.double())

    return whitened.float()


def segment_and_embed(
    strain_data,
    model,
    bandpass,
    whitener,
    spectral_density,
    sample_rate=4096,
    kernel_length=1.0,
    psd_length=64,
    fduration=1.0,
    batch_size=32,
    device='cuda',
    valid_times=None
):
    """
    Segment continuous strain data into 1-second windows and compute embeddings.

    Args:
        strain_data: Dictionary with ifo keys and continuous strain arrays
        model: The embedding model
        bandpass, whitener, spectral_density: Preprocessing transforms
        sample_rate: Sampling rate in Hz
        kernel_length: Length of each segment in seconds
        psd_length: Length for PSD estimation in seconds
        fduration: Duration for whitening
        batch_size: Batch size for processing
        device: torch device
        valid_times: Array of valid start times (for background data), or None

    Returns:
        embeddings: Array of embeddings [num_segments, embedding_dim]
        segment_times: Array of GPS times for each segment
    """
    logger = logging.getLogger(__name__)

    # Stack IFO data into [num_ifos, total_samples]
    ifos = sorted(strain_data.keys())
    strain_arrays = [strain_data[ifo] for ifo in ifos]

    # Check all ifos have same length
    lengths = [len(s) for s in strain_arrays]
    if len(set(lengths)) > 1:
        logger.warning(f"IFO data have different lengths: {lengths}. Using minimum.")
        min_length = min(lengths)
        strain_arrays = [s[:min_length] for s in strain_arrays]

    continuous_data = np.stack(strain_arrays, axis=0)  # [num_ifos, total_samples]

    # Calculate sizes
    kernel_samples = int(kernel_length * sample_rate)
    psd_samples = int(psd_length * sample_rate)
    fduration_samples = int(fduration * sample_rate)

    # Total samples needed per segment: psd_data + kernel + fduration
    total_samples_per_segment = psd_samples + kernel_samples + fduration_samples

    total_samples = continuous_data.shape[1]

    # Determine which segments to process
    if valid_times is not None:
        # For background: only process segments starting at valid times
        # valid_times should be in sample indices or GPS times
        # Assuming they are GPS start times in seconds
        logger.info(f"Processing {len(valid_times)} valid segments")
        segment_starts = []

        # Convert valid times to sample indices
        # This assumes valid_times are relative to the start of this file
        for vt in valid_times:
            # vt could be GPS time or sample index
            # If it's a GPS time, we'd need the file start time
            # For now, assume it's sample index
            if isinstance(vt, (int, np.integer)):
                sample_idx = int(vt)
            else:
                # If float, might be time in seconds - convert to samples
                sample_idx = int(vt * sample_rate)

            # Need psd_samples BEFORE sample_idx and kernel+fduration AFTER
            if sample_idx >= psd_samples and sample_idx + kernel_samples + fduration_samples <= total_samples:
                segment_starts.append(sample_idx)

        logger.info(f"Valid segments after filtering: {len(segment_starts)}")
    else:
        # For signals: process all non-overlapping 1-second segments
        # Each segment needs: psd_samples before + kernel_samples + fduration_samples
        # First segment starts at psd_samples (so PSD goes from 0 to psd_samples)
        # Calculate how many complete segments we can fit
        available_for_segments = total_samples - psd_samples
        num_segments = (available_for_segments - fduration_samples) // kernel_samples

        segment_starts = []
        for i in range(num_segments):
            start_idx = psd_samples + i * kernel_samples
            # Verify this segment has enough data
            if start_idx + kernel_samples + fduration_samples <= total_samples:
                segment_starts.append(start_idx)

        logger.info(f"Processing {len(segment_starts)} segments (of {num_segments} calculated)")

    embeddings_list = []
    segment_times_list = []

    # Process in batches
    num_batches = (len(segment_starts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(segment_starts))

            batch_segments = []
            batch_psd_data = []
            batch_times = []

            for seg_idx in range(batch_start, batch_end):
                start_sample = segment_starts[seg_idx]

                # Extract PSD data (before the segment)
                psd_start = start_sample - psd_samples
                psd_end = start_sample
                psd_segment = continuous_data[:, psd_start:psd_end]

                # Extract segment data (kernel + fduration)
                seg_end = start_sample + kernel_samples + fduration_samples
                segment = continuous_data[:, start_sample:seg_end]

                # Validate segment sizes
                expected_psd_size = psd_samples
                expected_seg_size = kernel_samples + fduration_samples

                if psd_segment.shape[1] != expected_psd_size:
                    logger.warning(
                        f"Skipping segment {seg_idx}: PSD has {psd_segment.shape[1]} samples, "
                        f"expected {expected_psd_size}"
                    )
                    continue

                if segment.shape[1] != expected_seg_size:
                    logger.warning(
                        f"Skipping segment {seg_idx}: segment has {segment.shape[1]} samples, "
                        f"expected {expected_seg_size}"
                    )
                    continue

                batch_segments.append(segment)
                batch_psd_data.append(psd_segment)
                batch_times.append(start_sample / sample_rate)

            # Skip empty batches
            if len(batch_segments) == 0:
                continue

            # Stack into tensors [batch, num_ifos, samples]
            batch_tensor = torch.tensor(
                np.stack(batch_segments, axis=0),
                dtype=torch.float32,
                device=device
            )

            psd_tensor = torch.tensor(
                np.stack(batch_psd_data, axis=0),
                dtype=torch.float32,
                device=device
            )

            # Preprocess: bandpass and whiten
            try:
                # Split into data to whiten and psd data is already done
                # We just need to apply preprocessing
                whitened_batch = preprocess_data(
                    batch_tensor,
                    psd_tensor,
                    bandpass,
                    whitener,
                    spectral_density,
                    device
                )

                # Extract just the kernel_length portion (first kernel_samples)
                # The whitened_batch has shape [batch, num_ifos, kernel + fduration]
                # We want just [batch, num_ifos, kernel_samples]
                whitened_batch = whitened_batch[:, :, :kernel_samples]

                # Compute embeddings
                embeddings_batch = model(whitened_batch)

                # Move to CPU and store
                embeddings_list.append(embeddings_batch.cpu().numpy())
                segment_times_list.extend(batch_times)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Skip this batch
                continue

    if len(embeddings_list) == 0:
        logger.warning("No embeddings computed!")
        return np.array([]), np.array([])

    embeddings = np.concatenate(embeddings_list, axis=0)
    segment_times = np.array(segment_times_list)

    logger.info(f"Computed {len(embeddings)} embeddings")

    return embeddings, segment_times


def process_files(
    file_paths,
    model,
    bandpass,
    whitener,
    spectral_density,
    output_path,
    is_background=False,
    sample_rate=4096,
    kernel_length=1.0,
    psd_length=64,
    fduration=1.0,
    batch_size=32,
    device='cuda',
    max_seconds=None
):
    """
    Process multiple h5 files and save embeddings.

    Args:
        file_paths: List of paths to h5 files
        model: The embedding model
        bandpass, whitener, spectral_density: Preprocessing transforms
        output_path: Path to save embeddings
        is_background: Whether this is background data (uses valid times)
        sample_rate: Sampling rate
        kernel_length: Segment length in seconds
        psd_length: PSD length in seconds
        fduration: Whitening duration
        batch_size: Batch size
        device: torch device
        max_seconds: Maximum number of seconds to process (None for all)
    """
    logger = logging.getLogger(__name__)

    all_embeddings = []
    all_times = []
    all_file_ids = []

    total_seconds = 0

    for file_idx, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
        logger.info(f"Processing {file_path}")

        # Load valid times if background
        valid_times = None
        if is_background:
            valid_times = load_valid_times(file_path)
            if valid_times is None:
                logger.warning(f"No valid times found for {file_path}, skipping")
                continue

        # Read strain data
        try:
            strain_data, start_time, file_sample_rate = read_strain_data(file_path)

            if not strain_data:
                logger.warning(f"No strain data found in {file_path}, skipping")
                continue

            # Check sample rate matches
            if file_sample_rate is not None and file_sample_rate != sample_rate:
                logger.warning(
                    f"File sample rate {file_sample_rate} != expected {sample_rate}"
                )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

        # Segment and embed
        try:
            embeddings, segment_times = segment_and_embed(
                strain_data,
                model,
                bandpass,
                whitener,
                spectral_density,
                sample_rate=sample_rate,
                kernel_length=kernel_length,
                psd_length=psd_length,
                fduration=fduration,
                batch_size=batch_size,
                device=device,
                valid_times=valid_times
            )

            if len(embeddings) == 0:
                logger.warning(f"No embeddings from {file_path}")
                continue

            all_embeddings.append(embeddings)
            all_times.append(segment_times)
            all_file_ids.extend([file_idx] * len(embeddings))

            total_seconds += len(embeddings)
            logger.info(f"Total seconds processed so far: {total_seconds}")

            # Check if we've hit the limit
            if max_seconds is not None and total_seconds >= max_seconds:
                logger.info(f"Reached max_seconds limit: {max_seconds}")
                break

        except Exception as e:
            logger.error(f"Error computing embeddings for {file_path}: {e}")
            continue

    if len(all_embeddings) == 0:
        logger.error("No embeddings computed from any files!")
        return

    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_times = np.concatenate(all_times, axis=0)
    final_file_ids = np.array(all_file_ids)

    # Truncate if needed
    if max_seconds is not None and len(final_embeddings) > max_seconds:
        final_embeddings = final_embeddings[:max_seconds]
        final_times = final_times[:max_seconds]
        final_file_ids = final_file_ids[:max_seconds]

    logger.info(f"Saving {len(final_embeddings)} embeddings to {output_path}")

    # Save to HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=final_embeddings, compression='gzip')
        f.create_dataset('times', data=final_times, compression='gzip')
        f.create_dataset('file_ids', data=final_file_ids, compression='gzip')

        # Store file paths as metadata
        f.attrs['num_files'] = len(file_paths)
        for i, fp in enumerate(file_paths[:len(set(final_file_ids))]):
            f.attrs[f'file_{i}'] = str(fp)

    logger.info(f"Saved embeddings: shape={final_embeddings.shape}")
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute embeddings on BBC and O4_test data'
    )

    parser.add_argument(
        '--model',
        type=Path,
        default='gwak/output/ResNet_HK/model_JIT.pt',
        help='Path to JIT model'
    )

    parser.add_argument(
        '--bbc-dir',
        type=Path,
        default='gwak/output/BBC_AnalysisReady_Cat12/HL',
        help='Directory containing BBC background h5 files'
    )

    parser.add_argument(
        '--o4-dir',
        type=Path,
        default='gwak/output/O4_test_MDC_short-0/HL',
        help='Directory containing O4 test signal h5 files'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default='gwak/output/embeddings',
        help='Output directory for embeddings'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=4096,
        help='Sample rate in Hz'
    )

    parser.add_argument(
        '--kernel-length',
        type=float,
        default=1.0,
        help='Segment length in seconds'
    )

    parser.add_argument(
        '--psd-length',
        type=float,
        default=64.0,
        help='PSD estimation length in seconds'
    )

    parser.add_argument(
        '--fduration',
        type=float,
        default=1.0,
        help='Duration for whitening'
    )

    parser.add_argument(
        '--fftlength',
        type=float,
        default=2.0,
        help='FFT length for spectral density'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )

    parser.add_argument(
        '--ifos',
        type=str,
        default='HL',
        help='Interferometers (e.g., HL for H1,L1)'
    )

    args = parser.parse_args()

    logger = setup_logging()

    # Parse IFOs
    ifos = [f'{ifo}1' for ifo in args.ifos]
    logger.info(f"Processing interferometers: {ifos}")

    # Load model
    device = torch.device(args.device)
    model = load_model(args.model, device)

    # Setup preprocessing
    bandpass, whitener, spectral_density = get_preprocessing_transforms(
        args.sample_rate,
        args.fduration,
        args.fftlength,
        device
    )

    # Get O4 files and count seconds
    logger.info("Processing O4 test data...")
    o4_files = sorted(args.o4_dir.glob('*.h5'))

    if len(o4_files) == 0:
        logger.error(f"No h5 files found in {args.o4_dir}")
        sys.exit(1)

    logger.info(f"Found {len(o4_files)} O4 files")

    # Process O4 files (all of them)
    o4_output = args.output_dir / 'o4_test_embeddings.h5'

    process_files(
        o4_files,
        model,
        bandpass,
        whitener,
        spectral_density,
        o4_output,
        is_background=False,
        sample_rate=args.sample_rate,
        kernel_length=args.kernel_length,
        psd_length=args.psd_length,
        fduration=args.fduration,
        batch_size=args.batch_size,
        device=device,
        max_seconds=None  # Process all O4 data
    )

    # Count how many seconds were in O4
    with h5py.File(o4_output, 'r') as f:
        o4_seconds = len(f['embeddings'])

    logger.info(f"O4 test data: {o4_seconds} seconds")

    # Now process BBC with roughly the same amount
    logger.info(f"Processing BBC background data (targeting ~{o4_seconds} seconds)...")
    bbc_files = sorted(args.bbc_dir.glob('*.h5'))

    if len(bbc_files) == 0:
        logger.error(f"No h5 files found in {args.bbc_dir}")
        sys.exit(1)

    logger.info(f"Found {len(bbc_files)} BBC files")

    bbc_output = args.output_dir / 'bbc_background_embeddings.h5'

    process_files(
        bbc_files,
        model,
        bandpass,
        whitener,
        spectral_density,
        bbc_output,
        is_background=True,
        sample_rate=args.sample_rate,
        kernel_length=args.kernel_length,
        psd_length=args.psd_length,
        fduration=args.fduration,
        batch_size=args.batch_size,
        device=device,
        max_seconds=o4_seconds  # Match O4 amount
    )

    logger.info("Done!")
    logger.info(f"O4 embeddings: {o4_output}")
    logger.info(f"BBC embeddings: {bbc_output}")


if __name__ == '__main__':
    main()
