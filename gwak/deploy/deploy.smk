models = ['white_noise_burst', 'gaussian', 'sine_gaussian', 'cusp', 'kink', 'kinkkink', 'bbh', 'S4_SimCLR_multiSignalAndBkg', 'combination'] 

wildcard_constraints:
    deploymodels = '|'.join(models)

rule export:
    input:
        config = '../deploy/deploy/config/export.yaml',
    params:
        cli = lambda wildcards: wildcards.deploymodels,
    output:
        artefact = 'tmp/export_{deploymodels}.log'
    shell:
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=GPU-3fbb2a42-ab69-aabf-c395-3f5c943dc939 poetry run python \
        ../deploy/deploy/cli_export.py \
        --config {input.config} \
        --project {params.cli} | tee {output.artefact}'

rule infer:
    input:
        config = '../deploy/deploy/config/infer.yaml',
    params:
        cli = lambda wildcards: wildcards.deploymodels,
        output = 'output/infer/{deploymodels}'
    output:
        artefact = 'tmp/infer_{deploymodels}.log'
    shell:
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=GPU-3fbb2a42-ab69-aabf-c395-3f5c943dc939 poetry run python \
        ../deploy/deploy/cli_infer.py \
        --config {input.config} \
        --project {params.cli} \
        --result_dir {params.output} | tee {output.artefact}'

rule export_all:
    input: expand(rules.export.output, deploymodels='combination')

rule infer_all:
    input: expand(rules.infer.output, deploymodels='combination')

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