data_ver_path_converter = {
    "O4b_cat1": "O4_MDC_background", # Have omicron file
    "O4b_cat1-chunked": "O4_MDC_background-chunked",
    "O4b_cat12": "BBC_AnalysisReady_Cat12",
    "O4b_cat12-katya": "BBC_AnalysisReady_Cat12-katya"
}

cl_configs = [
    'Transformer_SimCLR_multiSignal_all',
    'Transformer_SimCLR_multiSignalAndBkg_noSG',
    'S4_SimCLR_multiSignalAndBkg',
    'Transformer_patch_SimCLR_multiSignalAndBkg',
    'Transformer_patch_noClass_SimCLR_multiSignalAndBkg',
    's4_kl1.0_bs512',
    'transformer_patch64_kl0.5_bs512',
    'resnet_kl1.0_bs512_noAnnealClassifier_noMultiSG',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG_fixedWNBGaus',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG_fixedWNBGaus_noFakeGlitch_lowDim',
    'resnet_kl1.0_bs512',
    'Astroconformer',
    'iTransformer',
    'ResNet',
    'ResNet_6d',
    'ResNet_cat12',
    'ResNet_separate-glitch',
    'ResNet_mid',
    'torch_rbw_zp_resnet_do6_dcs064_epoch25',
    'torch_rbw_zp_resnet_do6_dcs128_epoch25',
    ]
coh_modes = [
    "real", "real_imag", "abs",
]
fm_configs = [
    'NF_onlyBkg',
    'NF_from_file_conditioning_bs64',
    "NF_from_file_conditioning",
    "NF_from_file_conditioning_bs1024",
    'NF_from_file_6d',
    'FM_multiSignalAndBkg',
    ]
ifo_configs = [
    'HL',
    'HV',
    'LV',
    'HLV'
]

wildcard_constraints:
    data_ver = '|'.join([x for x in data_ver_path_converter.keys()]),
    cl_config = '|'.join([x for x in cl_configs]),
    coh_mode = '|'.join([x for x in coh_modes]),
    fm_config = '|'.join([x for x in fm_configs]),
    ifos = '|'.join([x for x in ifo_configs])

rule train_cl:
    input:
        arg = GWAK_ROOT / "gwak/train/train/cli_fm.py",
        config = GWAK_ROOT / 'gwak/train/configs/{cl_config}.yaml',
        data_dir = lambda wildcards: directory(
            DATA_DIR
            / data_ver_path_converter[wildcards.data_ver]
            / wildcards.ifos
        )
    output:
        model = Path(
            OUTPUT_DIR 
            / "models/{data_ver}/{ifos}/{cl_config}/model_JIT.pt"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/train/pyproject.toml",
        logger_dir = directory(
            OUTPUT_DIR / "models/{data_ver}/{ifos}/{cl_config}"
        ),
        # The omicron triggers can only generate on LDG cluster.
        omicron = DATA_DIR / "O4_MDC_background/omicron/",
    shell:
        'source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} \
            --config {input.config} \
            --trainer.logger.save_dir {params.logger_dir} \
            --data.init_args.data_dir {input.data_dir} \
            --data.ifos {wildcards.ifos} \
            --model.num_ifos {wildcards.ifos} \
            --data.init_args.glitch_root {params.omicron}'

rule precompute_embeddings:
    input:
        arg = GWAK_ROOT / "gwak/train/train/precompute_embeddings.py",
        config = GWAK_ROOT / "gwak/train/configs/{cl_config}.yaml",
        embedding_model = expand(
            rules.train_cl.output.model,
            data_ver="{data_ver}",
            ifos="{ifos}",
            cl_config="{cl_config}",
        ),
        data_dir = lambda wildcards: directory(
            DATA_DIR
            / data_ver_path_converter[wildcards.data_ver]
            / wildcards.ifos
        )
    output:
        precom_data_dir = directory(
            OUTPUT_DIR / "data/{data_ver}/{ifos}/{cl_config}_{coh_mode}"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/train/pyproject.toml",
    shell:
        'source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} \
            --data-dir {input.data_dir} \
            --ifos {wildcards.ifos} \
            --config {input.config} \
            --embedding-model {input.embedding_model} \
            --coh_mode {wildcards.coh_mode} \
            --means {output.precom_data_dir}/means.npy \
            --stds {output.precom_data_dir}/stds.npy \
            --embeddings {output.precom_data_dir}/embeddings.npy \
            --labels {output.precom_data_dir}/labels.npy \
            --correlations {output.precom_data_dir}/correlations.npy \
            --nevents 100000 '

rule train_fm:
    input:
        arg = GWAK_ROOT / "gwak/train/train/cli_fm.py",
        config = GWAK_ROOT / "gwak/train/configs/{fm_config}.yaml",
        precom_data_dir = rules.precompute_embeddings.output.precom_data_dir
    output:
        model = Path(
            OUTPUT_DIR / "models" 
            / "{data_ver}/{ifos}/{cl_config}_{coh_mode}_{fm_config}"
            / "model_JIT.pt"
        ),
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/train/pyproject.toml",
        logger_dir = directory(
            OUTPUT_DIR / "models" 
            / "{data_ver}/{ifos}/{cl_config}_{coh_mode}_{fm_config}"
        ),
    shell:
        'source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} fit \
            --config {input.config} \
            --trainer.logger.save_dir {params.logger_dir} \
            --model.coh_mode {wildcards.coh_mode} \
            --data.embedding_path {input.precom_data_dir}/embeddings.npy \
            --data.c_path {input.precom_data_dir}/correlations.npy'

rule combine_models:
    input:
        arg = GWAK_ROOT / "gwak/train/train/combine_models.py",
        config = GWAK_ROOT / 'gwak/train/configs/{cl_config}.yaml',
        embedding_model = rules.train_cl.output.model,
        fm_model = rules.train_fm.output.model
    output:
        model = Path(
            OUTPUT_DIR / "models" 
            / "{data_ver}/{ifos}/{cl_config}_{coh_mode}_{fm_config}"
            / "combination/model_JIT.pt"
        ),
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/train/pyproject.toml",
    shell:
        'source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} \
            {input.embedding_model} \
            {input.fm_model} \
            --coh_mode {wildcards.coh_mode} \
            --config {input.config} \
            --outfile {output.model} '

rule make_offline_dataset:
    params:
        ifos = 'HL',
        num_samples = 100_000,
        dataset = 'train',
    output:
    shell:
        'python train/make_offline_dataset.py {params.ifos} \
            {params.num_samples} \
            {params.dataset}'


# rule compare_embeddings:
#     input:
#         data_dir = DATA_DIR / "O4_MDC_background-chunked/HL/"
#     params:
#         config = GWAK_ROOT / 'gwak/train/configs/resnet_kl1.0_bs512.yaml',
#         models_to_compare = [OUTPUT_DIR / 'resnet_kl1.0_bs512_HL/model_JIT.pt', OUTPUT_DIR / 's4_kl1.0_bs256_HL/model_JIT.pt'],
#         plot_dir = OUTPUT_DIR / 'plots/compare_embeddings/'
#     shell:
#         'mkdir -p {params.plot_dir}; '
#         'cd gwak/train; uv run python train/compare_embeddings.py {params.models_to_compare} \
#             --config {params.config} \
#             --data-dir {input.data_dir} \
#             --output {params.plot_dir} \
#             --nevents 1024'


# rule precompute_wnb_embeddings_classifier:
#     params:
#         embedding_model = expand(rules.train_cl.output.model,
#             cl_config='ResNet',
#             ifos='HL'),
#         data_dir = DATA_DIR / 'O4_MDC_background-chunked/HL/',
#         config = GWAK_ROOT / 'gwak/train/configsResNet.yaml'
#     output:
#         means = OUTPUT_DIR / 'ResNet_wnb_HL/means.npy',
#         stds = OUTPUT_DIR / 'ResNet_wnb_HL/stds.npy',
#         embeddings = OUTPUT_DIR / 'ResNet_wnb_HL/embeddings.npy',
#         correlations = OUTPUT_DIR / 'ResNet_wnb_HL/correlations.npy',
#         labels = OUTPUT_DIR / 'ResNet_wnb_HL/labels.npy'
#     shell:
#         'cd gwak/train; uv run python train/precompute_embeddings.py \
#             --embedding-model {params.embedding_model} \
#             --data-dir {params.data_dir} \
#             --config {params.config} \
#             --ifos HL \
#             --embeddings {output.embeddings} \
#             --labels {output.labels} \
#             --correlations {output.correlations} \
#             --means {output.means} \
#             --stds {output.stds} \
#             --include-signals WNB \
#             --nevents 200000 '

# rule precompute_sg_embeddings_classifier:
#     params:
#         embedding_model = expand(rules.train_cl.output.model,
#             cl_config='ResNet',
#             ifos='HL'),
#         data_dir = DATA_DIR / 'O4_MDC_background-chunked/HL/',
#         config = GWAK_ROOT / 'gwak/train/configs/ResNet.yaml'
#     output:
#         means = OUTPUT_DIR / 'ResNet_sg_HL/means.npy',
#         stds = OUTPUT_DIR / 'ResNet_sg_HL/stds.npy',
#         embeddings = OUTPUT_DIR / 'ResNet_sg_HL/embeddings.npy',
#         correlations = OUTPUT_DIR / 'ResNet_sg_HL/correlations.npy',
#         labels = OUTPUT_DIR / 'ResNet_sg_HL/labels.npy'
#     shell:
#         'cd gwak/train; uv run python train/precompute_embeddings.py \
#             --embedding-model {params.embedding_model} \
#             --data-dir {params.data_dir} \
#             --config {params.config} \
#             --ifos HL \
#             --embeddings {output.embeddings} \
#             --labels {output.labels} \
#             --correlations {output.correlations} \
#             --means {output.means} \
#             --stds {output.stds} \
#             --include-signals SG \
#             --nevents 200000 '

# rule train_wnb_classifier:
#     params:
#         artefact = directory(OUTPUT_DIR / 'ResNet_HL_FM_multiSignalAndBkg/'),
#         embeddings = OUTPUT_DIR / 'ResNet_signals_HL/embeddings.npy',
#         data_dir = DATA_DIR / 'O4_MDC_background-chunked/HL/',
#         config = GWAK_ROOT / 'gwak/train/configsFM_multiSignalAndBkg.yaml',
#         means = OUTPUT_DIR / 'ResNet_signals_HL/means.npy',
#         stds = OUTPUT_DIR / 'ResNet_signals_HL/stds.npy',
#         labels = OUTPUT_DIR / 'ResNet_signals_HL/labels.npy'
#     shell:
#         'cd gwak/train; uv run python train/cli_fm.py fit --config {params.config} \
#             --trainer.logger.save_dir {params.artefact} \
#             --model.means {params.means} \
#             --model.stds {params.stds} \
#             --data.embedding_path {params.embeddings} \
#             --data.labels_path {params.labels} '

# rule train_sg_classifier:
#     params:
#         artefact = directory(OUTPUT_DIR / 'ResNet_HL_FM_multiSignalAndBkg/'),
#         embeddings = OUTPUT_DIR / 'ResNet_signals_HL/embeddings.npy',
#         data_dir = DATA_DIR / 'O4_MDC_background-chunked/HL/',
#         config = GWAK_ROOT / 'gwak/train/configsFM_multiSignalAndBkg.yaml',
#         means = OUTPUT_DIR / 'ResNet_signals_HL/means.npy',
#         stds = OUTPUT_DIR / 'ResNet_signals_HL/stds.npy',
#         labels = OUTPUT_DIR / 'ResNet_signals_HL/labels.npy'
#     shell:
#         'cd gwak/train; uv run python train/cli_fm.py fit --config {params.config} \
#             --trainer.logger.save_dir {params.artefact} \
#             --model.means {params.means} \
#             --model.stds {params.stds} \
#             --data.embedding_path {params.embeddings} \
#             --data.labels_path {params.labels} '


# # rule make_plots_i:
# #     input:
# #         embedding_model = expand(rules.train_cl.output.model,
# #             cl_config='{cl_config}',
# #             ifos='{ifos}'),
# #         fm_model = expand(rules.train_fm.output.model,
# #             fm_config='{fm_config}',
# #             cl_config='{cl_config}',
# #             ifos='{ifos}'),
# #     params:
# #         data_dir = 'output/BBC_AnalysisReady_Cat12/{ifos}/',
# #         config = 'train/configs/{cl_config}.yaml',
# #         conditioning = lambda wildcards: "True" if "conditioning" in wildcards.fm_config else "False"
# #     output:
# #         directory('output/plots/{cl_config}_{fm_config}_{ifos}/'),
# #     shell:
# #         'mkdir -p {output}; '
# #         'cd gwak/train; uv run python train/plots.py \
# #             --embedding-model {input.embedding_model} \
# #             --fm-model {input.fm_model} \
# #             --data-dir {params.data_dir} \
# #             --ifos {wildcards.ifos} \
# #             --config {params.config} \
# #             --output {output} \
# #             --conditioning {params.conditioning} \
# #             --nevents 15000 \
# #             --threshold-1yr 48 '

# # rule make_plots:
# #     input:
# #         expand(rules.make_plots_i.output,
# #             cl_config='ResNet',
# #             fm_config='NF_from_file_conditioning',
# #             ifos=['HL'])

# # rule run_evaluate_one_month:
# #     input:
# #         expand(OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/evaluation/scores.npy',
# #             cl_config='torch_rbw_zp_resnet_do6_dcs128_epoch25', fm_config='NF_from_file_6d', ifos='HL')


# # rule run_evaluate_one_month_if:
# #     input:
# #         expand(OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/scores.npy',
# #             cl_config='ResNet_6d', ifos='HL')




# # rule train_isolation_forest:
# #     input:
# #         embeddings   = OUTPUT_DIR / '{cl_config}_{ifos}/embeddings.npy',
# #         labels       = OUTPUT_DIR / '{cl_config}_{ifos}/labels.npy',
# #         correlations = OUTPUT_DIR / '{cl_config}_{ifos}/correlations.npy',
# #         means        = OUTPUT_DIR / '{cl_config}_{ifos}/means.npy',
# #         stds         = OUTPUT_DIR / '{cl_config}_{ifos}/stds.npy',
# #     output:
# #         OUTPUT_DIR / '{cl_config}_{ifos}/isolation_forest.joblib'
# #     shell:
# #         'cd gwak/train; uv run python train/train_if.py \
# #             --embeddings {input.embeddings} \
# #             --labels {input.labels} \
# #             --correlations {input.correlations} \
# #             --means {input.means} \
# #             --stds {input.stds} \
# #             --output {output}'

# # rule evaluate_one_month_if:
# #     input:
# #         embedding_model = OUTPUT_DIR / '{cl_config}_{ifos}/model_JIT.pt',
# #         if_model        = OUTPUT_DIR / '{cl_config}_{ifos}/isolation_forest.joblib',
# #         means           = OUTPUT_DIR / '{cl_config}_{ifos}/means.npy',
# #         stds            = OUTPUT_DIR / '{cl_config}_{ifos}/stds.npy',
# #     params:
# #         inference_dir = "/home/hongyin.chen/anti_gravity/gwak/gwak/output/infer/torch_rbw_zp_resnet_do6_dcs128_epoch25_NF_from_file_conditioning_HL/one_month/inference_result",
# #         output_dir    = lambda wildcards: str(OUTPUT_DIR / f'{wildcards.cl_config}_{wildcards.ifos}_IF/evaluation/'),
# #     output:
# #         scores = OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/scores.npy',
# #     shell:
# #         'cd gwak/train; uv run python ../evaluate_one_month.py \
# #             --model-path {input.embedding_model} \
# #             --if-model {input.if_model} \
# #             --means {input.means} \
# #             --stds {input.stds} \
# #             --inference-dir {params.inference_dir} \
# #             --output-dir {params.output_dir} \
# #             --smooth-window 1 \
# #             --veto-duration 10'

# # rule evaluate_one_month:
# #     input:
# #         model = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/combination/model_JIT.pt',
# #     params:
# #         inference_dir = "/home/hongyin.chen/anti_gravity/gwak/gwak/output/infer/torch_rbw_zp_resnet_do6_dcs128_epoch25_NF_from_file_conditioning_HL/one_month/inference_result",
# #         output_dir = lambda wildcards: str(OUTPUT_DIR / f'{wildcards.cl_config}_{wildcards.fm_config}_{wildcards.ifos}/evaluation/'),
# #     output:
# #         scores = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/evaluation/scores.npy',
# #     shell:
# #         'cd gwak/train; uv run python ../evaluate_one_month.py \
# #             --model-path {input.model} \
# #             --inference-dir {params.inference_dir} \
# #             --output-dir {params.output_dir} \
# #             --smooth-window 4 \
# #             --veto-duration 10'

# # rule efficiency_plots:
# #     input:
# #         model = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/combination/model_JIT.pt',
# #         scores = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/evaluation/scores.npy',
# #     params:
# #         signal_dataset = OUTPUT_DIR / 'dataset_train_HL_SR4096_kernel1.0_hrss.h5',
# #         output_dir = lambda wildcards: str(OUTPUT_DIR / f'{wildcards.cl_config}_{wildcards.fm_config}_{wildcards.ifos}/evaluation/'),
# #     output:
# #         snr_plot  = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/evaluation/efficiency_vs_snr.png',
# #         hrss_plot = OUTPUT_DIR / '{cl_config}_{fm_config}_{ifos}/evaluation/efficiency_vs_hrss.png',
# #     shell:
# #         'cd gwak/train; uv run python ../efficiency_plots.py \
# #             --model-path {input.model} \
# #             --background-scores {input.scores} \
# #             --signal-dataset {params.signal_dataset} \
# #             --output-dir {params.output_dir}'

# # rule efficiency_plots_if:
# #     input:
# #         embedding_model = OUTPUT_DIR / '{cl_config}_{ifos}/model_JIT.pt',
# #         if_model        = OUTPUT_DIR / '{cl_config}_{ifos}/isolation_forest.joblib',
# #         means           = OUTPUT_DIR / '{cl_config}_{ifos}/means.npy',
# #         stds            = OUTPUT_DIR / '{cl_config}_{ifos}/stds.npy',
# #         scores          = OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/scores.npy',
# #     params:
# #         signal_dataset = OUTPUT_DIR / 'dataset_train_HL_SR4096_kernel1.0_hrss.h5',
# #         output_dir     = lambda wildcards: str(OUTPUT_DIR / f'{wildcards.cl_config}_{wildcards.ifos}_IF/evaluation/'),
# #     output:
# #         snr_plot  = OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/efficiency_vs_snr.png',
# #         hrss_plot = OUTPUT_DIR / '{cl_config}_{ifos}_IF/evaluation/efficiency_vs_hrss.png',
# #     shell:
# #         'cd gwak/train; uv run python ../efficiency_plots.py \
# #             --model-path {input.embedding_model} \
# #             --if-model {input.if_model} \
# #             --means {input.means} \
# #             --stds {input.stds} \
# #             --background-scores {input.scores} \
# #             --signal-dataset {params.signal_dataset} \
# #             --output-dir {params.output_dir}'
