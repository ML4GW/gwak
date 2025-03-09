models = ['white_noise_burst', 'gaussian', 'sine_gaussian', 'cusp', 'kink', 'kinkkink', 'bbh'] 

wildcard_constraints:
    deploymodels = '|'.join(models)

DEPLOY_CLI = {
    'white_noise_burst': 'white_noise_burst',
    'gaussian': 'gaussian',
    'bbh': 'bbh'
}

rule export: 
    input:
        config = 'deploy/deploy/config/export.yaml'
    params:
        cli = lambda wildcards: DEPLOY_CLI[wildcards.deploymodels]
    output:
        artefact = directory('output/export/{deploymodels}')
    shell:
        'set -x; cd deploy; poetry run python ../deploy/deploy/cli_export.py \
        --config ../{input.config} --project {params.cli}'

rule infer: 
    input:
        config = 'deploy/deploy/config/infer.yaml'
    params:
        cli = lambda wildcards: DEPLOY_CLI[wildcards.deploymodels]
    output:
        artefact = directory('output/infer/{deploymodels}')
    shell:
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=GPU-3fbb2a42-ab69-aabf-c395-3f5c943dc939 poetry run python \
        ../deploy/deploy/cli_infer.py --config ../{input.config} --project {params.cli}'

rule export_all:
    input: expand(rules.export.output, deploymodels='bbh')

rule infer_all:
    input: expand(rules.infer.output, deploymodels='white_noise_burst')
