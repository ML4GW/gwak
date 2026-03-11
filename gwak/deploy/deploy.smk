ifo_modes = [
    'H', 'L', 'V', 'K',
    'HL', 'HV', 'HK', 'LV', 'LK', 'VK',
    'HLV', 'HLK', 'HVK', 'LVK', 
    'HLVK'
]

runs = [
    'background', 'bbc-short-0', 'bbc-short-1', 
    'one_day', 'one_month', 'one_year', 
    'one_decade','one_centure',
    'test_run'
]


wildcard_constraints:
    ifo_mode = '|'.join(x for x in ifo_modes),
    run_name = '|'.join(x for x in runs)


runs_TS_converter = {
    'background': 0, 'bbc-short-0': 0, 'bbc-short-1': 0, 
    'one_day': 86400, 'one_month': 2678400., 'one_year': 31557600, 
    'one_decade': 315576000, 'one_centure': 3155760000,
    'test_run': 1
}


# snakemake -c1 $GWAK_OUTPUT_DIR/export/{cl_config}_{fm_config}_{ifo_mode}/combination
rule export:
    input:
        arg = GWAK_DIR / "gwak/deploy/deploy/cli.py",
        config = GWAK_DIR / "gwak/deploy/deploy/config/export.yaml"
    # params:
        # gpu = "CUDA_VISIBLE_DEVICES=GPU-9be0d4df-e1db-fd6a-912b-a6a07ae3430f" {params.gpu} 
    output:
        artefact = directory(OUTPUT_DIR / "export/{cl_config}_{fm_config}_{ifo_mode}")
    shell:
        "set -x; cd deploy; uv run python \
        {input.arg} export \
        --config {input.config} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config}"


# python deploy/cli.py export --config deploy/config/export.yaml --project combination
rule production_export:
    input:
        arg = GWAK_DIR / "gwak/deploy/deploy/cli.py",
        image = IMAGE_DIR / "deploy.sif",
        config = GWAK_DIR / "gwak/deploy/deploy/config/export.yaml"
    params:
        bind_1 = f"{CONTAIN_OUTPUT_DIR}:/production",
        bind_2 = f"{OUTPUT_DIR}:/opt/gwak/gwak/output",
    shell: 
        "set -x; apptainer exec --nv \
        --bind {params.bind_1},{params.bind_2} \
        {input.image} \
        python {input.arg} export  \
        --config {input.config} \
        --project combination"

# snakemake -c1 $GWAK_OUTPUT_DIR/infer/{cl_config}_{fm_config}_{ifo_mode}/{run_name}
rule condor_infer:
    input:
        arg = GWAK_DIR / "gwak/deploy/deploy/cli.py",
        config = GWAK_DIR / "gwak/deploy/deploy/config/infer_condor.yaml",
        plan_model = rules.export.output
    params:
        # gpu = "CUDA_VISIBLE_DEVICES=GPU-9be0d4df-e1db-fd6a-912b-a6a07ae3430f", {params.gpu}
        timeslide = lambda wildcards: runs_TS_converter[wildcards.run_name]
    output:
        artefact = directory(OUTPUT_DIR / "infer/{cl_config}_{fm_config}_{ifo_mode}/{run_name}")
    shell:
        'mkdir -p tmp; '
        'set -x; cd deploy; uv run python \
        {input.arg} infer_condor \
        --config {input.config} \
        --run_name {wildcards.run_name} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config} \
        --Tb {params.timeslide}'


rule slurm_infer:
    input:
        config = 'deploy/deploy/config/infer_slurm.yaml',
    params:
        timeslide = lambda wildcards: runs_TS_converter[wildcards.run_name]
    output:
        artefact = directory(OUTPUT_DIR / "Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}")
    shell:
        "set -x; cd deploy; uv run python \
        deploy/cli.py deploy \
        --config ../{input.config} \
        --run_name {wildcards.run_name} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config} \
        --Tb {params.timeslide}"

rule scan_outlier:
    input:
        arg = GWAK_DIR / "gwak/deploy/deploy/cli.py",
        config = GWAK_DIR / "gwak/deploy/deploy/config/analysis.yaml",
        infer_result = rules.condor_infer.output
    output: 
        artefact = directory(LOUVRE_DIR / "{cl_config}_{fm_config}_{ifo_mode}/{run_name}/")
    shell:
        "set -x; cd deploy; uv run python \
        {input.arg} post_analyze --config {input.config} \
        --run_name {wildcards.run_name} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config}"

rule export_all:
    input:
        expand(
            rules.export.output,
            cl_config=["torch_rbw_zp_resnet_do6_dcs128_epoch25"],
            fm_config=["NF_from_file_conditioning"],
            ifo_mode=["HL"],
        )

# snakemake -c4 output/Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}/ -F 
rule slurm_infer_all:
    input: 
        expand(
            rules.slurm_infer.output,
            cl_config=[
                "ResNet_cat12",
                "ResNet_separate-glitch",
                "torch_rbw_zp_resnet_do6_dcs128_epoch25",
            ], 
            fm_config=[
                "NF_from_file_conditioning",
            ], 
            ifo_mode=["HL"], 
            run_name=["one_year"]
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

rule estimate_far:
    input:
        path_to_infer = '/home/hongyin.chen/anti_gravity/gwak/gwak/output/infer/combination/inference_result/'# expand(rules.infer.output, deploymodels='combination')
    output:
        'output/infer/far_metrics.npy'
    shell:
        'python deploy/save_far.py \
            {input.path_to_infer} \
            --dataset data \
            --duration 0.5 \
            --num_thresholds 100 \
            --outfile {output} \
            --direction negative '