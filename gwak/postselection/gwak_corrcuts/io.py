"""io.py — dataset streaming + feature-table write/read.

Streams the 16 GB HDF5 by class in small h5py slices (never loads a whole class). Writes the
feature table as CSV (slice default) with a sidecar <out>.config.json.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class EventBatch:
    data: np.ndarray        # (b, 2, Ns) float32
    snr: np.ndarray         # (b,) float32
    class_name: str
    row_indices: np.ndarray  # (b,) int64


def list_classes(path: str) -> List[str]:
    """Discover <Class>_data / <Class>_snrs pairs."""
    import h5py
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
    classes = []
    for k in keys:
        if k.endswith("_data"):
            base = k[: -len("_data")]
            if base + "_snrs" in keys:
                classes.append(base)
    return sorted(classes)


def class_shapes(path: str) -> Dict[str, Tuple[int, int, int]]:
    """(N, 2, Ns) per class without loading data."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for c in list_classes(path):
            out[c] = tuple(f[c + "_data"].shape)
    return out


def iter_event_batches(path: str, class_name: str, batch_size: int,
                       subsample: Optional[int] = None,
                       rng: Optional[np.random.RandomState] = None) -> Iterator[EventBatch]:
    """Yield EventBatches via h5py slice reads. subsample = deterministic random row subset."""
    import h5py
    dkey = class_name + "_data"
    skey = class_name + "_snrs"
    with h5py.File(path, "r") as f:
        N = f[dkey].shape[0]
        if subsample is not None and subsample < N:
            if rng is None:
                rng = np.random.RandomState(1234)
            rows = np.sort(rng.choice(N, size=subsample, replace=False))
        else:
            rows = np.arange(N)
        for start in range(0, rows.size, batch_size):
            sel = rows[start: start + batch_size]
            # h5py fancy indexing requires sorted, unique -> sel already sorted
            data = f[dkey][sel.tolist()]
            snr = f[skey][sel.tolist()]
            yield EventBatch(
                data=np.asarray(data, dtype=np.float32),
                snr=np.asarray(snr, dtype=np.float32),
                class_name=class_name,
                row_indices=np.asarray(sel, dtype=np.int64),
            )


def write_feature_table(rows: List[dict], config, out_path: str) -> None:
    """Write rows (list of dicts) to CSV + sidecar config.json."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if config.output_format == "parquet":
        try:
            df.to_parquet(out_path)
        except Exception:
            out_path = out_path.rsplit(".", 1)[0] + ".csv"
            df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
        if config.also_csv and not out_path.endswith(".csv"):
            df.to_csv(out_path.rsplit(".", 1)[0] + ".csv", index=False)
    with open(out_path + ".config.json", "w") as fh:
        json.dump(config.to_dict(), fh, indent=2)


def read_feature_table(path: str):
    """Return (DataFrame, Config) for the plotting stage."""
    import pandas as pd
    from .config import Config
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cfg = None
    cfg_path = path + ".config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = Config.from_dict(json.load(fh))
    return df, cfg
