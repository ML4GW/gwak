import csv
import io
import sys
import subprocess


def query_gpus():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    reader = csv.reader(io.StringIO(result.stdout), skipinitialspace=True)

    return [
        {
            "uuid": uuid, 
            "name": name,
            "cc": float(cc),
            "total_mem": int(total),
            "free_mem": int(free),
         }
        for uuid, name, cc, total, free in reader
    ]


def gpu_selector():

    gpu_infos = query_gpus()
    gpu_list = []
    for gpu_info in gpu_infos:
        if gpu_info["cc"] < 8.0:
            continue

        if gpu_info["total_mem"] < 16000:
            continue

        if gpu_info["free_mem"] < 10000:
            continue

        gpu_list.append(gpu_info)
        
    gpu_list = sorted(gpu_list, key=lambda g: g["cc"])
    if not gpu_list:
        sys.exit("No GPU cards found exit process!")
    return gpu_list