ifo_modes = [
    'H', 'L', 'V', 'K',
    'HL', 'HV', 'HK', 'LV', 'LK', 'VK',
    'HLV', 'HLK', 'HVK', 'LVK', 
    'HLVK'
]

runs = [
    'background', 
    'one_day',
    'one_month',
    'one_year', 
    'one_decade',
    'one_centure',
    'test_run'
]

models = [
    'combination'
] 

wildcard_constraints:
    ifo_mode = '|'.join(x for x in ifo_modes),
    run_name = '|'.join(x for x in runs),
    deploymodels = '|'.join(models),


runs_TS_converter = {
    'background': 0,
    'one_day': 86400,
    'one_month': 2678400,
    'one_year': 31557600, 
    'one_decade': 315576000,
    'one_centure': 3155760000,
    'test_run': 1
}

rule export:
    input:
        config = 'deploy/deploy/config/export.yaml',
    params:
        cli = lambda wildcards: wildcards.deploymodels,
    output:
        artefact = 'tmp/export_{deploymodels}.log'
    shell:
        'mkdir -p tmp; '
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=0 poetry run python \
        deploy/cli.py export \
        --config ../{input.config} \
        --project {params.cli} | tee ../{output.artefact}'

rule infer:
    input:
        config = 'deploy/deploy/config/infer.yaml',
    params:
        cli = lambda wildcards: wildcards.deploymodels,
        output = 'output/infer/{deploymodels}'
    output:
        artefact = 'tmp/infer_{deploymodels}.log'
    shell:
        'mkdir -p tmp; '
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=0 poetry run python \
        deploy/cli.py infer \
        --config ../{input.config} \
        --project {params.cli} | tee ../{output.artefact}'
        # --result_dir {params.output} 

rule infer_condor:
    input:
        config = 'deploy/deploy/config/infer.yaml',
    params:
        cli = lambda wildcards: wildcards.deploymodels,
        output = 'output/infer/{deploymodels}'
    output:
        artefact = 'tmp/infer_condor_{deploymodels}.log'
    shell:
        'mkdir -p tmp; '
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=0 poetry run python \
        deploy/cli.py infer_condor \
        --config ../{input.config} \
        --project {params.cli} | tee ../{output.artefact}'
        # --result_dir {params.output} 

rule deploy:
    input:
        config = 'deploy/deploy/config/deploy.yaml',
    params:
        timeslide = lambda wildcards: runs_TS_converter[wildcards.run_name]
    output:
        artefact = directory('output/Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}/')
    shell:
        'set -x; cd deploy; poetry run python \
        deploy/cli.py deploy \
        --config ../{input.config} \
        --run_name {wildcards.run_name} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config} \
        --Tb {params.timeslide}'
        # --ifos {wildcards.ifo_mode}# Resolve HL to ["H1", "L1"]


rule export_all:
    input: expand(rules.export.output, deploymodels='combination')

rule infer_all:
    input: expand(rules.infer.output, deploymodels='combination')

rule infer_condor_all:
    input: expand(rules.infer_condor.output, deploymodels='combination')


# snakemake -c4 output/Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}/ -F 
rule deploy_all:
    input: 
        expand(
            rules.deploy.output,
            cl_config=[
                "torch_rbw_zp_resnet_do64_dcs128_epoch25", 
            ], 
            fm_config=[
                "NF_from_file_conditioning",
            ], 
            ifo_mode=["HL"], 
            run_name=["test_run"]
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