models = [
    'combination'
] 

wildcard_constraints:
    deploymodels = '|'.join(models)

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
    output:
        artefact = directory('output/Slurm_Job_{run_name}/')
    shell:
        'set -x; cd deploy; poetry run python \
        deploy/cli.py deploy \
        --config ../{input.config} \
        --run_name {wildcards.run_name}'


rule export_all:
    input: expand(rules.export.output, deploymodels='combination')

rule infer_all:
    input: expand(rules.infer.output, deploymodels='combination')

rule infer_condor_all:
    input: expand(rules.infer_condor.output, deploymodels='combination')

rule deploy_all:
    input: expand(rules.deploy.output, run_name=['first_run', 'second_run'])

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