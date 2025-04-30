import os
import re
import torch
from torch.nn import functional as F 
import math
import h5py
import logging
import configparser

import numpy as np

from pathlib import Path
from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityDict



glitch_keys = [
    'time', 
    'frequency', 
    'tstart', 
    'tend', 
    'fstart', 
    'fend', 
    'snr', 
    'q', 
    'amplitude', 
    'phase'
]


########################
### File level utils ###
########################

def get_conincident_segs(
    ifos:list,
    start:int,
    stop:int,
    state_flag:list,
):

    query_flag = []
    for i, ifo in enumerate(ifos):
        query_flag.append(f"{ifo}:{state_flag[i]}")

    flags = DataQualityDict.query_dqsegdb(
        query_flag,
        start,
        stop
    )

    segs = []

    for contents in flags.intersection().active.to_table():
        print(f"Preparing segment data between {contents['start']} and {contents['end']}")
        segs.append((contents["start"], contents["end"]))

    return segs


def get_background(
    seg_start: int,
    seg_end: int, 
    ifos:list,
    frame_type:list,
    channels:list,
    sample_rate:int,
    verbose:bool=True
): 
    
    strains = {}
    logging.info(f"Collecting strain data from {seg_start} to {seg_end} at {channels}")
    for num, ifo in enumerate(ifos):
        
        if 'V' in ifo:  ### VIRGO uses frametype WITHOUT the IFO name
            files = find_urls(
                site=f"{ifo[0]}",
                frametype=f"{frame_type[num]}",
                gpsstart=seg_start,
                gpsend=seg_end,
                urltype="file",
            )

        else: ### LIGO uses frametype WITH the IFO name

            files = find_urls(
                site=f"{ifo[0]}",
                frametype=f"{ifo}_{frame_type[num]}",
                gpsstart=seg_start,
                gpsend=seg_end,
                urltype="file",
            )
        print(f"{ifo}_{frame_type[num]}")
        print(f"Found {len(files)} files for {ifo}")
        if len(files) == 0:
            raise ValueError(f"No files found for {ifo} between {seg_start} and {seg_end}")

        print(seg_start, seg_end)
        strains[ifo] = TimeSeries.read(
            files, 
            f"{ifo}:{channels[num]}", 
            start=seg_start, 
            end=seg_end, 
            nproc=8, 
            verbose=verbose
        ).resample(sample_rate).value
        print(f"Strain data for {ifo} collected")
        print(strains[ifo].shape)
        print()

    return strains


def find_files_with_ifo(directory, start_time, end_time, ifo):
    matching_files = []
    # Pattern to capture IFO, start time, duration
    pattern = re.compile(r".*-(\w\d)_BurstBenchmark-(\d+)-(\d+)\.gwf")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_ifo = match.group(1)
            file_start = int(match.group(2))
            duration = int(match.group(3))
            file_end = file_start + duration
            # Match requested IFO and time overlap
            if file_ifo == ifo and not (file_end <= start_time or file_start >= end_time):
                full_path = os.path.join(directory, filename)
                matching_files.append(full_path)

    return sorted(matching_files)

def get_injections(
    seg_start: int,
    seg_end: int,
    ifos:list,
    channels:list,
    sample_rate:int,
    injections_dir:str='/scratch/burst.benchmark/o4b-2/',
    verbose:bool=True
):

    strains = {}
    logging.info(f"Collecting strain data from {seg_start} to {seg_end} at {channels}")

    for num, ifo in enumerate(ifos):
        files = find_files_with_ifo(injections_dir, seg_start, seg_end, ifo)
        print(f"Found {len(files)} files for {ifo}")
        if len(files) == 0:
            raise ValueError(f"No files found for {ifo} between {seg_start} and {seg_end}")

        print(seg_start, seg_end)
        strains[ifo] = TimeSeries.read(
            files,
            f"{ifo}:{channels[num]}",
            start=seg_start,
            end=seg_end,
            nproc=8,
            verbose=verbose
        ).resample(sample_rate).value
        print(f"Strain data for {ifo} collected")
        print(strains[ifo].shape)
        print()
    strains['GPS_start'] = seg_start
    strains['GPS_stop'] = seg_end

    return strains


def create_lcs(
    ifo: str,
    frametype: str,
    start_time,
    end_time,
    output_dir,
    urltype="file"
):
    """
    Create lcs file for omicron to fetch strain data (*.gwf file) 
    for each interferometer. 
    """
    head = "file://localhost"
    empty = ""


    files = find_urls(
        site=ifo[0],
        frametype=f"{frametype}",
        gpsstart=start_time,
        gpsend=end_time,
        urltype=urltype,
    )
    # breakpoint()
    
    output_dir = output_dir / ifo
    output_dir.mkdir(parents=True, exist_ok=True)
    
    f = open(output_dir / "data_file.lcf", "a")
    for file in files:
        f.write(f"{file.replace(head, empty)}\n")
    f.close()



def glitch_merger(
    ifos,
    omicron_path: Path,
    channels,
    output_file=None,
    glitch_keys=glitch_keys
):

    
    for i, ifo in enumerate(ifos):

        glitch_dir = \
            Path(omicron_path) / f"{ifo}/trigger_output/merge/{ifo}:{channels[i]}"

        h5_name = {}
        for key in glitch_keys:

            h5_name[key] = []   

        for file in sorted(glitch_dir.glob("*.h5")):

            with h5py.File(file, "r") as h:
                
                for key in glitch_keys:
                    
                    h5_name[key].append(h["triggers"][key])
                    
        for key in glitch_keys:
            print(key, h5_name[key])
            h5_name[key] = np.concatenate(h5_name[key])
            
        if output_file is None:
            output_file = omicron_path / "glitch_info.h5"

        with h5py.File(output_file, "a") as g:
            
            g1 = g.create_group(ifo)
            
            for key in glitch_keys:
                g1.create_dataset(key, data=h5_name[key])

    return output_file



def omicron_bashes(
    ifos,
    start_time,
    end_time,
    project_dir: Path,
    # INI
    q_range,
    frequency_range,
    frame_type,
    channels,
    cluster_dt,
    sample_rate,
    chunk_duration,
    segment_duration,
    overlap_duration,
    mismatch_max,
    snr_threshold,
    # log_file: Path,
    verbose: bool = False,
    state_flag=None,
    mode="GW"
):
    # Modified from BBHNet and CCSNet. 

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file
    bash_files = []

    for i, ifo in enumerate(ifos):
        
        config = configparser.ConfigParser()
        section = mode
        config.add_section(section)

        config.set(section, "q-range", f"{q_range[0]} {q_range[1]}")
        config.set(section, "frequency-range", f"{frequency_range[0]} {frequency_range[1]}")
        config.set(section, "frametype", f"{ifo}_{frame_type[i]}")
        config.set(section, "channels", f"{ifo}:{channels[i]}")
        config.set(section, "cluster-dt", str(cluster_dt))
        config.set(section, "sample-frequency", str(sample_rate))
        config.set(section, "chunk-duration", str(chunk_duration))
        config.set(section, "segment-duration", str(segment_duration))
        config.set(section, "overlap-duration", str(overlap_duration))
        config.set(section, "mismatch-max", str(mismatch_max))
        config.set(section, "snr-threshold", str(snr_threshold))
        # in an online setting, can also pass state-vector,
        # and bits to check for science mode
        if state_flag != None:
            config.set(section, "state-flag", f"{ifo}:{state_flag}")

        config_file_path = project_dir / f"{ifo}/omicron_{ifo}.ini"
        bash_file_path = project_dir / f"{ifo}/run_omicron.sh"
        cache_file = project_dir / ifo / "data_file.lcf"
        output_dir = project_dir / f"{ifo}" / "trigger_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bash_files.append(bash_file_path)

        # write config file
        with open(config_file_path, "w") as config_file:
            config.write(config_file)
            
        omicron_args = [
            # f"omicron-process {section}", # Env pyomicron method
            f"python -m omicron.cli.process {section}", # Pyomicron submodule method
            f"--gps {start_time} {end_time}",
            f"--ifo {ifo}",
            f"--config-file {str(config_file_path.resolve())}",
            f"--output-dir {str(output_dir.resolve())}",
            f"--cache-file {cache_file.resolve()}",
            # f"--log-file {str(project_dir/ifo)}",
            "--verbose"
            # "request_disk=100M",
            # "--skip-gzip",
            # "--skip-rm",
        ]
        with open (bash_file_path, 'w') as rsh:
            for args in omicron_args:
                rsh.writelines(f"{args} \\\n")

    return bash_files



import os 
from pathlib import Path



class Pathfinder:

    def __init__(
            self,
            gwak_env: str,  
            suffix: str = None, 
            file_name: str = None, 
        ):

        self.file_name = file_name

        if suffix is not None:
            self.path = Path(os.getenv(gwak_env)) / f"{suffix}"

        else: 
            self.path = Path(os.getenv(gwak_env))


    def get_path(self):
            
        self.path.mkdir(parents=True, exist_ok=True)

        if self.file_name is not None:

            return self.path / self.file_name
        
        return self.path

#########################
### Strain data utils ###
#########################

class Sequence:

    def __init__(
        self,
        fname: Path,
        data_format: str, 
        shifts: list[float],
        # batch_size: int,
        ifos: list,
        kernel_size: int,
        sample_rate: int,
        # inference_sampling_rate: float,
        inj_type=None,
        precision: str="float32",
        device: str = "cuda"
        # state_shape: tuple,
    ):

        # self.fname = fname
        self.shifts = shifts
        # self.batch_size = batch_size
        self.ifos = ifos
        self.n_ifos = len(ifos)
        self.kernel_size = kernel_size

        self.state_shape = (self.n_ifos, kernel_size)

        self.inj_type = inj_type
        self.precision = precision

        self.sample_rate = sample_rate
        self.stride = int(sample_rate)
        self.step_size = self.stride * (kernel_size / sample_rate)
        
        self.strain_dict = {}
        self.fname = fname
        
        if data_format in ("h5", "hdf", "hdf5"):
            with h5py.File(self.fname, "r") as h:

                for ifo in self.ifos:
                    self.strain_dict[ifo] = torch.Tensor(h[ifo][:]).to(device)

        self.size = len(self.strain_dict[ifo])

    @property
    def remainder(self):
        # the number of remaining data points not filling a full batch
        return (self.size - max(self.shifts)) % self.step_size

    @property
    def num_pad(self):
        # the number of zeros we need to pad the last batch
        # to make it a full batch
        return int((self.step_size - self.remainder) % self.step_size)
    
    def __len__(self):

        return math.ceil((self.size - max(self.shifts)) / self.step_size)

    def __iter__(self):

        # Check if this line will hide potential implmetation error! 
        bh_state = torch.empty(self.state_shape) 
        inj_state = None

        for i in range(len(self)):

            last = i == len(self) - 1
            for ifo_idx, (ifo, shift) in enumerate(zip(self.ifos, self.shifts)): 

                start = int(shift + i * self.step_size)
                end = int(start + self.kernel_size)


                if last and self.remainder:
                    end = start + int(self.remainder)

                data = self.strain_dict[ifo][start:end]
                # if this is the last batch
                # possibly pad it to make it a full batch
                if last:

                    data = F.pad(data, (0, self.num_pad), "constant", 0)

                bh_state[ifo_idx, :] = data

                        
            yield bh_state, inj_state