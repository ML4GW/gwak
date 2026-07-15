import h5py

import numpy as np
from pathlib import Path


raw_data_path = Path("/fred/oz016/Andy/Data/gwak/HL")
new_data_path = Path("/fred/oz016/Andy/New_Data/gwak/HL")


files = raw_data_path.glob("*.h5")

for file in files:
    
    new_file = new_data_path / file.name
    
    with h5py.File(file, "r") as fin, h5py.File(new_file, "w") as fout:

        for name in ["H1", "L1"]:
            data = fin[name]

            dset = fout.create_dataset(
            name,
            dtype=np.float32,           # Reduce the data type to float32 
            data=data,
            shape=data.shape,
            chunks=(135168,),
            compression="lzf",          # or compression="gzip", compression_opts=1
        )