from pathlib import Path
from pelicanfs.core import OSDFFileSystem
from gwpy.timeseries import TimeSeries
from time import time
import h5py


pelfs = OSDFFileSystem()

exp_file = '/igwn/cit/staging/hongyin.chen/Data/GWAK/HL/O4b_cat1_v2/background-1401696286-3614.h5'


def pelican_read(
    read_dir: Path = exp_file
):
    data_dir = {}
    # Open the file via pelicanfs and pass the context to h5py
    with pelfs.open(file_path, 'rb') as f:
        with h5py.File(f, 'r') as h5_file:
            for ifo in list(h5_file.keys()):
                data_dir[ifo] = h5_file[ifo]
            return data_dir

