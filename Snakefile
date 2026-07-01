import os
from pathlib import Path

CONTAIN_OUTPUT_DIR = Path(os.environ["GWAK_CONTAIN_OUTPUT_DIR"])
IMAGE_DIR          = Path(os.environ["GWAK_IMAGE_DIR"])

GWAK_DIR           = Path(os.environ["GWAK_DIR"])
DATA_DIR           = Path(os.environ["GWAK_DATA_DIR"])
OUTPUT_DIR         = Path(os.environ["GWAK_OUTPUT_DIR"])
LOUVRE_DIR         = Path(os.environ["GWAK_LOUVRE_DIR"])

rule build_envs_containers:
    input:
        deploy_container = GWAK_DIR / "gwak/deploy/deploy.def"
    output:
        deploy_image =  IMAGE_DIR / "deploy.sif"
    params:
        processors = "--mksquashfs-args '-processors 128'"
    shell:
        "set -x; apptainer build {params.processors} \
        {output.deploy_image} {input.deploy_container}"

include: GWAK_DIR / "gwak/data/data.smk"
include: GWAK_DIR / "gwak/train/train.smk"
include: GWAK_DIR / "gwak/deploy/deploy.smk"


rule pull_all:
    input:
        expand(rules.pull_data.output,
            segment_type=['short-0.o4b-2', 'short-1.o4b-2', 'short-0.o4b-0', 'short-1.o4b-0'],
            ifos=['hl', 'hv', 'lv', 'hlv'])

rule run_efficiency_plots_if:
    input:
        expand(OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/efficiency_vs_snr.png',
            cl_config='torch_rbw_zp_resnet_do6_dcs128_epoch25', ifos='HL')

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