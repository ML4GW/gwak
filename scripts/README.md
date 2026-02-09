# Embedding Computation Scripts

## compute_embeddings.py

Computes embeddings using a trained ResNet model on BBC background and O4 test data.

### Usage

Basic usage with default paths:
```bash
python scripts/compute_embeddings.py
```

With custom paths:
```bash
python scripts/compute_embeddings.py \
    --model gwak/output/ResNet_HK/model_JIT.pt \
    --bbc-dir gwak/output/BBC_AnalysisReady_Cat12/HL \
    --o4-dir gwak/output/O4_test_MDC_short-0/HL \
    --output-dir gwak/output/embeddings \
    --batch-size 64
```

### Arguments

- `--model`: Path to JIT-compiled model (default: `gwak/output/ResNet_HK/model_JIT.pt`)
- `--bbc-dir`: Directory with BBC background h5 files (default: `gwak/output/BBC_AnalysisReady_Cat12/HL`)
- `--o4-dir`: Directory with O4 test signal h5 files (default: `gwak/output/O4_test_MDC_short-0/HL`)
- `--output-dir`: Output directory for embeddings (default: `gwak/output/embeddings`)
- `--sample-rate`: Sampling rate in Hz (default: 4096)
- `--kernel-length`: Segment length in seconds (default: 1.0)
- `--psd-length`: PSD estimation length in seconds (default: 64.0)
- `--fduration`: Duration for whitening (default: 1.0)
- `--fftlength`: FFT length for spectral density (default: 2.0)
- `--batch-size`: Batch size for processing (default: 32)
- `--device`: Device to use, `cuda` or `cpu` (default: auto-detect)
- `--ifos`: Interferometers to process (default: `HL` for H1,L1)

### How It Works

1. **Loads the model**: Loads the JIT-compiled ResNet embedding model
2. **Processes O4 test data**:
   - Reads all h5 files from O4 directory
   - Segments continuous strain into non-overlapping 1-second windows
   - Applies bandpass (30-2047 Hz) and whitening
   - Computes embeddings for each second
   - Counts total seconds processed

3. **Processes BBC background data**:
   - Reads h5 files from BBC directory
   - For each file, loads corresponding `*_clean_valid.npy` file
   - Only processes segments at valid clean times
   - Applies same preprocessing as O4
   - Stops after processing roughly the same number of seconds as O4

4. **Saves embeddings**:
   - Saves to HDF5 files with compression
   - `o4_test_embeddings.h5`: Contains O4 signal embeddings
   - `bbc_background_embeddings.h5`: Contains BBC background embeddings
   - Each file contains:
     - `embeddings`: Array of shape `[num_seconds, embedding_dim]`
     - `times`: GPS times (in seconds) for each embedding
     - `file_ids`: Which source file each embedding came from
     - File paths stored as attributes

### Data Format

Input h5 files should contain:
- `H1`: Hanford strain data (continuous array)
- `L1`: Livingston strain data (continuous array)

For BBC background files, should have corresponding:
- `background-{gps}-{duration}_clean_valid.npy`: Array of valid start times

### Output Format

HDF5 files with:
```
embeddings/
├── o4_test_embeddings.h5
│   ├── embeddings [N, D]  # N=num_seconds, D=embedding_dim
│   ├── times [N]          # GPS time for each second
│   └── file_ids [N]       # Source file index
└── bbc_background_embeddings.h5
    ├── embeddings [M, D]
    ├── times [M]
    └── file_ids [M]
```

### Example

On the cluster:
```bash
# Pull the latest code
git pull origin for-nplm

# Run the script
python scripts/compute_embeddings.py \
    --batch-size 128 \
    --device cuda
```

The script will automatically:
1. Process all O4 test files
2. Count how many seconds were processed
3. Process BBC files until reaching approximately the same count
4. Save both datasets to the output directory
