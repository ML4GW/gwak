import os
from copy import deepcopy
from pathlib import Path


GWAK_ROOT = Path(workflow.basedir).resolve()
DEFAULT_CONFIG = GWAK_ROOT / "setups" / "config.yaml"
LOCAL_SETUP_CONFIG_FILE = GWAK_ROOT / "setups" / "config.local.yaml"

include: GWAK_ROOT / "setups/converter.smk"
include: GWAK_ROOT / "setups/setup.smk"
include: GWAK_ROOT / "gwak/data/data.smk"
include: GWAK_ROOT / "gwak/train/train.smk"
include: GWAK_ROOT / "gwak/deploy/deploy.smk"
include: GWAK_ROOT / "gwak/postselection/postselection.smk"

# Working config
ana_ver_list = [
    "O4b_gwak_cat12"
]
data_ver_list = [
    "O4b_cat1-chunked",
    "O4b_cat12-katya"
]
ifos_list = ["HL"]
cl_config_list = [
    "ResNet_6d",
]
coh_mode_list = [
    "real",
    "real_imag",
    "abs"
]
fm_config_list = [
    "NF_from_file",
]
noise_run_list = ["one_month"]
foreground_run_list = [
    "bbc-short-0", "bbc-short-1"
]
run_name_list = noise_run_list + foreground_run_list

rule gwak_init:
    input: rules.bootstrap_complete.output

rule build_containers:
    input: rules.build_deploy_containers.output

rule pull_all:
    input:
        expand(rules.pull_data.output,
            segment_type=[
                'short-0.o4b-2', 
                'short-1.o4b-2', 
                'short-0.o4b-0', 
                'short-1.o4b-0'
            ],
            ifos=['hl', 'hv', 'lv', 'hlv']
        )

rule train_all:
    input:
        expand(
            rules.combine_models.output,
            data_ver=data_ver_list,
            ifo_mode=ifos_list,
            cl_config=cl_config_list,
            coh_mode=coh_mode_list,
            fm_config=fm_config_list,
        )

rule scan_all:
    input:
        expand(
            rules.condor_infer.output,
            ifo_mode=ifos_list,
            ana_ver=ana_ver_list,
            data_ver=data_ver_list,
            cl_config=cl_config_list,
            coh_mode=coh_mode_list,
            fm_config=fm_config_list,
            run_name=run_name_list
        )

rule benchmark:
    input:
        expand(
            rules.scan_outlier.output + rules.find_outlier_segs.output + rules.plot_bbc_benchmark.output,
            ifo_mode=ifos_list,
            ana_ver=ana_ver_list,
            data_ver=data_ver_list,
            cl_config=cl_config_list,
            coh_mode=coh_mode_list,
            fm_config=fm_config_list,
            noise_run=noise_run_list,
            run_name=run_name_list,
        )
