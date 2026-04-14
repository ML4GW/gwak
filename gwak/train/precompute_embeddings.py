import yaml
import argparse
import numpy as np
from pathlib import Path
import h5py

import torch

from ml4gw.distributions import PowerLaw
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

from gwak.train.dataloader import SignalDataloader
from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def frequency_cos_similarity(batch):
    H = torch.fft.rfft(batch[:, 0, :], dim=-1)
    L = torch.fft.rfft(batch[:, 1, :], dim=-1)
    numerator = torch.sum(H * torch.conj(L), dim=-1)
    norm_H = torch.linalg.norm(H, dim=-1)
    norm_L = torch.linalg.norm(L, dim=-1)
    rho_complex = numerator / (norm_H * norm_L + 1e-8)
    rho_real = torch.real(rho_complex).unsqueeze(-1)
    return rho_real

# ---------- helpers to load saved datasets ----------
def _pick_data_and_labels_from_npz(npz):
    candidates_data = ['data', 'X', 'inputs', 'waves', 'pulses', 'processed']
    candidates_labels = ['labels', 'y', 'targets', 'class', 'classes']
    data = None
    for k in candidates_data:
        if k in npz and npz[k].ndim == 3:
            data = npz[k]
            break
    if data is None:
        for k in npz.files:
            if npz[k].ndim == 3:
                data = npz[k]
                break
    labels = None
    for k in candidates_labels:
        if k in npz and npz[k].ndim == 1 and npz[k].shape[0] == data.shape[0]:
            labels = npz[k]
            break
    return data, labels

def _pick_data_and_labels_from_h5(h5):
    candidates_data = ['data', 'X', 'inputs', 'waves', 'pulses', 'processed']
    candidates_labels = ['labels', 'y', 'targets', 'class', 'classes']
    data = None
    for k in candidates_data:
        if k in h5 and len(h5[k].shape) == 3:
            data = h5[k]
            break
    if data is None:
        for k in h5.keys():
            if len(h5[k].shape) == 3:
                data = h5[k]
                break
    labels = None
    for k in candidates_labels:
        if k in h5 and len(h5[k].shape) == 1 and h5[k].shape[0] == data.shape[0]:
            labels = h5[k][:]
            break
    return data, labels

def load_saved_dataset(path):
    """
    Returns (data_like, labels_array_or_None, length, is_memmapped_like)
    data_like supports slicing: data_like[i0:i1] -> (B, 2, T)
    """
    ext = str(path).lower()
    if ext.endswith('.npz'):
        npz = np.load(path, mmap_mode='r')
        data, labels = _pick_data_and_labels_from_npz(npz)
        if data is None:
            raise ValueError("Could not find a 3D array in NPZ for data (expected shape (N,2,T)).")
        return data, (labels if labels is not None else None), data.shape[0], True
    elif ext.endswith('.h5') or ext.endswith('.hdf5'):
        h5 = h5py.File(path, 'r')
        data, labels = _pick_data_and_labels_from_h5(h5)
        if data is None:
            raise ValueError("Could not find a 3D dataset in H5 for data (expected shape (N,2,T)).")
        return (h5, data), (labels if labels is not None else None), data.shape[0], False
    else:
        raise ValueError(f"Unsupported dataset extension: {path}. Use .npz or .h5")

# ----------------------------------------------------

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Process and merge ROOT files into datasets.')
    parser.add_argument('--embedding-model', type=str, default=None)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--ifos', type=str)
    parser.add_argument('--embeddings', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--correlations', type=str)
    parser.add_argument('--means', type=str, default=None)
    parser.add_argument('--stds', type=str, default=None)
    parser.add_argument('--nevents', type=int, default=10000)
    parser.add_argument('--include-signals', default=None, help='Use signal_classes, priors, waveforms from config if set')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to a saved dataset (.npz or .h5) with shape (N,2,T). If provided, embeddings are computed on it instead of generating on the fly.')

    args = parser.parse_args()

    embed_model = torch.jit.load(args.embedding_model)
    embed_model.eval()
    embed_model.to(device=device)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    sample_rate = config['data']['init_args']['sample_rate']
    kernel_length = config['data']['init_args']['kernel_length']
    psd_length = config['data']['init_args']['psd_length']
    fduration = config['data']['init_args']['fduration']
    fftlength = config['data']['init_args']['fftlength']
    batch_size = 128
    batches_per_epoch = 2000000
    num_workers = config['data']['init_args']['num_workers']
    data_saving_file = config['data']['init_args']['data_saving_file']

    duration = fduration + kernel_length

    if args.include_signals in ['All', 'ALL', 'all']:
        signal_classes = [
            "SineGaussian",
            "BBH",
            "Gaussian",
            "Cusp",
            "Kink",
            "KinkKink",
            "WhiteNoiseBurst",
            "CCSN",
            "Background",
            "Glitch",
        ]
        priors = [
            SineGaussianBBC(),
            LAL_BBHPrior(),
            GaussianBBC(),
            CuspBBC(),
            KinkBBC(),
            KinkkinkBBC(),
            WhiteNoiseBurstBBC(),
            None,
            None,
            None,
        ]
        waveforms = [
            SineGaussian(sample_rate=sample_rate, duration=duration),
            IMRPhenomPv2(),
            Gaussian(sample_rate=sample_rate, duration=duration),
            GenerateString(sample_rate=sample_rate),
            GenerateString(sample_rate=sample_rate),
            GenerateString(sample_rate=sample_rate),
            WhiteNoiseBurst(sample_rate=sample_rate, duration=duration),
            None,
            None,
            None,
        ]
        extra_kwargs = [None, {"ringdown_duration": 0.9}, None, None, None, None, None, None, None, None]

    elif args.include_signals in ['WNB', 'wnb']:
        signal_classes = ["WhiteNoiseBurst", "Background", "Glitch"]
        priors = [WhiteNoiseBurstBBC(), None, None]
        waveforms = [WhiteNoiseBurst(sample_rate=sample_rate, duration=duration), None, None]
        extra_kwargs = [None, None, None]
    elif args.include_signals in ['SG', 'sg']:
        signal_classes = ["SineGaussian", "Background", "Glitch"]
        priors = [SineGaussianBBC(), None, None]
        waveforms = [SineGaussian(sample_rate=sample_rate, duration=duration), None, None]
        extra_kwargs = [None, None, None]
    elif args.include_signals is None:
        signal_classes = ['Glitch', 'Background']
        priors = [None, None]
        waveforms = [None, None]
        extra_kwargs = [None, None]

    all_labels = []
    all_embeddings = []
    all_correlations = []

    # ========== BRANCH: use saved dataset if provided ==========
    if args.dataset_path is not None:
        ds_path = Path(args.dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"--dataset-path not found: {ds_path}")

        loaded, labels_arr, n_samples, is_memmap = load_saved_dataset(str(ds_path))

        h5_handle = None
        data_ds = None
        if isinstance(loaded, tuple) and len(loaded) == 2:
            h5_handle, data_ds = loaded
        else:
            data_ds = loaded

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_np = np.asarray(data_ds[start:end])
            if batch_np.ndim != 3 or batch_np.shape[1] != 2:
                raise ValueError(f"Expected batch shape (B,2,T); got {batch_np.shape}")

            processed = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)

            labels_chunk = labels_arr[start:end] if labels_arr is not None else None

            embeddings = embed_model(processed).detach().cpu().numpy()
            if args.correlations:
                correlations = frequency_cos_similarity(processed).cpu().detach().numpy()

            if labels_chunk is None:
                labels_chunk = -1 * np.ones((embeddings.shape[0],), dtype=np.int32)

            all_labels.append(np.asarray(labels_chunk))
            all_embeddings.append(embeddings)
            if args.correlations:
                all_correlations.append(correlations)

            del processed, embeddings
            torch.cuda.empty_cache()

        if h5_handle is not None:
            h5_handle.close()

    # ========== BRANCH: generate on the fly ==========
    else:
        loader = SignalDataloader(
            signal_classes, priors, waveforms, extra_kwargs,
            data_dir=args.data_dir,
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            psd_length=psd_length,
            fduration=fduration,
            fftlength=fftlength,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            num_workers=num_workers,
            data_saving_file=data_saving_file,
            ifos=args.ifos,
            snr_prior=torch.distributions.Uniform(3, 30),
            glitch_root=f"/home/katya.govorkova/gwak2/gwak/output/O4b_AnalysisReady_Cat12/omicron/"
        )

        n_iter = args.nevents // batch_size
        train_loader = loader.train_dataloader()
        train_iter = iter(train_loader)
        for i in range(n_iter):

            if i % 10 == 0:
                print(f"Processed batch {i}/{n_iter}")

            clean_batch, glitch_batch = next(train_iter)
            clean_batch = clean_batch.to(device)
            glitch_batch = glitch_batch.to(device)

            processed, labels, _, _ = loader.on_after_batch_transfer([clean_batch, glitch_batch], None, local_test=True)

            if processed.shape[0] == 0:
                del clean_batch, glitch_batch
                torch.cuda.empty_cache()
                continue

            embeddings = embed_model(processed).cpu().detach().numpy()
            if args.correlations:
                correlations = frequency_cos_similarity(processed).cpu().detach().numpy()

            all_labels.append(labels.cpu().detach().numpy())
            all_embeddings.append(embeddings)
            if args.correlations:
                all_correlations.append(correlations)

            del clean_batch, glitch_batch, processed, embeddings
            torch.cuda.empty_cache()

    # ---------- save outputs ----------
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) else np.empty((0,), dtype=np.int32)
    all_embeddings = np.concatenate(all_embeddings, axis=0) if len(all_embeddings) else np.empty((0, 0), dtype=np.float32)
    if args.correlations:
        all_correlations = np.concatenate(all_correlations, axis=0) if len(all_correlations) else np.empty((0, 1), dtype=np.float32)

    np.save(f'{args.labels}', all_labels)
    print('Labels shape', all_labels.shape)

    np.save(f'{args.embeddings}', all_embeddings)
    print('Embeddings shape', all_embeddings.shape)

    if args.correlations:
        np.save(f'{args.correlations}', all_correlations)
        print('Correlation shape', all_correlations.shape)

    means = np.mean(all_embeddings, axis=0) if all_embeddings.size else np.array([])
    stds = np.std(all_embeddings, axis=0) if all_embeddings.size else np.array([])
    if args.means is not None:
        np.save(f'{args.means}', means)
    if args.stds is not None:
        np.save(f'{args.stds}', stds)
