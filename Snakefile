import os
from copy import deepcopy
from pathlib import Path


GWAK_ROOT = Path(workflow.basedir).resolve()
DEFAULT_CONFIG = GWAK_ROOT / "setups" / "config.yaml"
LOCAL_SETUP_CONFIG_FILE = GWAK_ROOT / "setups" / "config.local.yaml"

include: GWAK_ROOT / "setups/setup.smk"
include: GWAK_ROOT / "gwak/data/data.smk"
include: GWAK_ROOT / "gwak/train/train.smk"
include: GWAK_ROOT / "gwak/deploy/deploy.smk"
include: GWAK_ROOT / "gwak/postselection/postselection.smk"

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

rule run_efficiency_plots_if:
    input:
        expand(
            OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/efficiency_vs_snr.png',
            cl_config='torch_rbw_zp_resnet_do6_dcs128_epoch25', 
            ifos='HL'
        )

rule scan_all:
    input: 
        expand(
            rules.scan_outlier.output,
            cl_config=[
                "torch_rbw_zp_resnet_do6_dcs128_epoch25",
            ], 
            fm_config=[
                "NF_from_file_conditioning",
            ], 
            ifo_mode=["HL"], 
            run_name=[
                "one_year", 
                # "bbc-short-0", 
                # "bbc-short-1", 
            ]
        )
