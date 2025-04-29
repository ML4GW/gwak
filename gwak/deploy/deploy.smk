models = ['white_noise_burst', 'gaussian', 'sine_gaussian', 'cusp', 'kink', 'kinkkink', 'bbh', 'S4_SimCLR_multiSignalAndBkg', 'combination'] 

wildcard_constraints:
    deploymodels = '|'.join(models)

DEPLOY_CLI = {
    'white_noise_burst': 'white_noise_burst',
    'gaussian': 'gaussian',
    'bbh': 'bbh', 
    'S4_SimCLR_multiSignalAndBkg': 'S4_SimCLR_multiSignalAndBkg',
    'combination': 'combination'
}

rule export: 
    input:
        config = 'deploy/deploy/config/export.yaml'
    params:
        cli = lambda wildcards: DEPLOY_CLI[wildcards.deploymodels]
    output:
        artefact = directory('output/export/{deploymodels}')
    shell:
        'set -x; cd deploy; CUDA_VISIBLE_DEVICES=GPU-3fbb2a42-ab69-aabf-c395-3f5c943dc939 poetry run python ../deploy/deploy/cli_export.py \
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
    input: expand(rules.export.output, deploymodels='combination')

rule infer_all:
    input: expand(rules.infer.output, deploymodels='combination')


# models = ['white_noise_burst', 'gaussian', 'sine_gaussian', 'cusp', 'kink', 'kinkkink', 'bbh', 'S4_SimCLR_multiSignalAndBkg', 'combination'] 
# 
# wildcard_constraints:
#     deploymodels = '|'.join(models)
# 
# rule combine_models:
#     params:
#         embedding_model = expand(rules.train_cl.output.model,
#             cl_config='S4_SimCLR_multiSignalAndBkg',
#             ifos='HL'),
#         fm_model = expand(rules.train_fm.output.model,
#             fm_config='NF_onlyBkg',
#             cl_config='S4_SimCLR_multiSignalAndBkg',
#             ifos='HL'),
#     output:
#         '/home/katya.govorkova/gwak2/gwak/output/combination/model_JIT.pt'
#     shell:
#         'python deploy/deploy/combine_models.py \
#             {params.embedding_model} \
#             {params.fm_model} \
#             --batch_size 512 \
#             --kernel_length 0.5 \
#             --sample_rate 4096 \
#             --num_ifos 2 \
#             --outfile {output} '
# 
# rule export:
#     input:
#         config = 'deploy/deploy/config/export.yaml',
#         model = rules.combine_models.output
#     params:
#         cli = lambda wildcards: wildcards.deploymodels
#     output:
#         artefact = directory('/home/katya.govorkova/gwak2/gwak/output/export/{deploymodels}')
#     shell:
#         'set -x; cd deploy; CUDA_VISIBLE_DEVICES=0 poetry run python \
#         ../deploy/deploy/cli_export.py \
#         --config ../{input.config} \
#         --project {params.cli} \
#         --model_weights {input.model}'
# 
# rule infer:
#     input:
#         config = 'deploy/deploy/config/infer.yaml',
#         model_dir = expand(rules.export.output.artefact, deploymodels='{deploymodels}')
#     params:
#         cli = lambda wildcards: wildcards.deploymodels
#     output:
#         directory('/home/katya.govorkova/gwak2/gwak/output/infer/{deploymodels}')
#     shell:
#         'set -x; cd deploy; CUDA_VISIBLE_DEVICES=0 poetry run python \
#         ../deploy/deploy/cli_infer.py \
#         --config ../{input.config} \
#         --project {params.cli} \
#         --model_repo_dir {input.model_dir} \
#         --result_dir {output}'
# 
# rule export_all:
#     input: expand(rules.export.output, deploymodels='combination')
# 
# rule infer_all:
#     input: expand(rules.infer.output, deploymodels='combination')
# 
# rule estimate_far:
#     input:
#         path_to_infer = expand(rules.infer.output, deploymodels='combination')
#     output:
#         'output/infer/far_metrics.npy'
#     shell:
#         'python deploy/save_far.py \
#             {input.path_to_infer} \
#             --dataset data \
#             --duration 0.5 \
#             --num_thresholds 100 \
#             --outfile {output} \
#             --direction negative '