import os

cl_configs = [
    'Transformer_SimCLR_multiSignal_all',
    'Transformer_SimCLR_multiSignalAndBkg_noSG',
    'S4_SimCLR_multiSignalAndBkg',
    'Transformer_patch_SimCLR_multiSignalAndBkg',
    'Transformer_patch_noClass_SimCLR_multiSignalAndBkg',
    's4_kl0.5_bs512',
    's4_kl1.0_bs512',
    's4_kl2.0_bs512',
    's4_kl0.5_bs256',
    's4_kl1.0_bs256',
    's4_kl2.0_bs256',
    's4_kl1.0_bs256_noGlitch',
    's4_kl1.0_bs256_noClassifier',
    's4_kl1.0_bs256_noAnnealClassifier_noMultiSG',
    's4_kl1.0_bs256_noClassifier_noMultiSG_fixedWNBGaus',
    's4_kl1.0_bs256_noClassifier_noMultiSG_fixedWNBGaus_t0.01',
    's4_kl1.0_bs256_noClassifier_noMultiSG_fixedWNBGaus_t0.5',
    'transformer_patch64_kl0.5_bs512',
    'transformer_patch64_kl1.0_bs512',
    'transformer_patch64_kl2.0_bs512',
    'transformer_patch64_kl1.0_bs512_noClassifier',
    'resnet_kl1.0_bs512_noAnnealClassifier_noMultiSG',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG_fixedWNBGaus',
    'resnet_kl1.0_bs512_noClassifier_noMultiSG_fixedWNBGaus_noFakeGlitch_lowDim',
    'resnet_kl1.0_bs512',
    'Astroconformer',
    'iTransformer',
    'ResNet'
    ]
fm_configs = [
    'NF_onlyBkg',
    'NF_from_file_conditioning',
    'FM_multiSignalAndBkg',
    ]
ifo_configs = [
    'HL',
    'HV',
    'LV',
    'HLV'
]
wildcard_constraints:
    cl_config = '|'.join([x for x in cl_configs]),
    fm_config = '|'.join([x for x in fm_configs]),
    ifos = '|'.join([x for x in ifo_configs])

rule train_cl:
    input:
        config = 'train/configs/{cl_config}.yaml',
        data_dir = 'output/O4_MDC_background/{ifos}/'
    output:
        model = 'output/{cl_config}_{ifos}/model_JIT.pt'
    params:
        artefact = directory('output/{cl_config}_{ifos}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.init_args.data_dir {input.data_dir} \
            --data.ifos {wildcards.ifos} \
            --model.num_ifos {wildcards.ifos} \
            --data.glitch_root /n/netscratch/iaifi_lab/Lab/emoreno/O4_MDC_background/omicron/{wildcards.ifos}/'

rule compare_embeddings:
    input:
        data_dir = 'output/O4_MDC_background/HL/'
    params:
        config = 'train/configs/resnet_kl1.0_bs512.yaml',
        models_to_compare = ['output/resnet_kl1.0_bs512_HL/model_JIT.pt', 'output/s4_kl1.0_bs256_HL/model_JIT.pt'],
        plot_dir = 'output/plots/compare_embeddings/'
    shell:
        'mkdir -p {params.plot_dir}; '
        'python train/compare_embeddings.py {params.models_to_compare} \
            --config {params.config} \
            --data-dir {input.data_dir} \
            --output {params.plot_dir} \
            --nevents 1024'

rule precompute_embeddings:
    params:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='{cl_config}',
            ifos='{ifos}'),
        data_dir = 'output/O4_MDC_background/{ifos}/',
        config = 'train/configs/{cl_config}.yaml'
    output:
        means = 'output/{cl_config}_{ifos}/means.npy',
        stds = 'output/{cl_config}_{ifos}/stds.npy',
        embeddings = 'output/{cl_config}_{ifos}/embeddings.npy',
        labels = 'output/{cl_config}_{ifos}/labels.npy',
        correlations = 'output/{cl_config}_{ifos}/correlations.npy'
    shell:
        'python train/precompute_embeddings.py \
            --embedding-model {params.embedding_model} \
            --data-dir {params.data_dir} \
            --config {params.config} \
            --ifos {wildcards.ifos} \
            --embeddings {output.embeddings} \
            --labels {output.labels} \
            --correlations {output.correlations} \
            --means {output.means} \
            --stds {output.stds} \
            --nevents 100000 '

rule train_fm:
    params:
        artefact = directory('output/{cl_config}_{fm_config}_{ifos}/'),
        config = 'train/configs/{fm_config}.yaml',
        means = 'output/{cl_config}_{ifos}/means.npy',
        stds = 'output/{cl_config}_{ifos}/stds.npy',
        embeddings = 'output/{cl_config}_{ifos}/embeddings.npy',
        correlations = 'output/{cl_config}_{ifos}/correlations.npy',
        conditioning = lambda wildcards: "True" if "conditioning" in wildcards.fm_config else "False"
    output:
        model = 'output/{cl_config}_{fm_config}_{ifos}/model_JIT.pt'
    shell:
        'python train/cli_fm.py fit --config {params.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.embedding_path {params.embeddings} \
            --data.c_path {params.correlations} \
            --model.means {params.means} \
            --model.stds {params.stds} \
            --model.conditioning {params.conditioning} '

rule combine_models:
    input:
        config = 'train/configs/{cl_config}.yaml',
    params:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='{cl_config}',
            ifos='{ifos}'),
        fm_model = expand(rules.train_fm.output.model,
            fm_config='{fm_config}',
            cl_config='{cl_config}',
            ifos='{ifos}'),
    output:
        'output/{cl_config}_{fm_config}_{ifos}/combination/model_JIT.pt'
    shell:
        'python train/combine_models.py \
            {params.embedding_model} \
            {params.fm_model} \
            --config {input.config} \
            --outfile {output} '

rule make_plots_i:
    params:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='{cl_config}',
            ifos='{ifos}'),
        fm_model = expand(rules.train_fm.output.model,
            fm_config='{fm_config}',
            cl_config='{cl_config}',
            ifos='{ifos}'),
        data_dir = 'output/O4_MDC_background/{ifos}/',
        config = 'train/configs/{cl_config}.yaml',
        conditioning = lambda wildcards: "True" if "conditioning" in wildcards.fm_config else "False"
    output:
        directory('output/plots/{cl_config}_{fm_config}_{ifos}/'),
    shell:
        'mkdir -p {output}; '
        'python train/plots.py \
            --embedding-model {params.embedding_model} \
            --fm-model {params.fm_model} \
            --data-dir {params.data_dir} \
            --ifos {wildcards.ifos} \
            --config {params.config} \
            --output {output} \
            --conditioning {params.conditioning} \
            --nevents 100000 \
            --threshold-1yr 15 '

rule make_plots:
    input:
        expand(rules.make_plots_i.output,
            cl_config='resnet_kl1.0_bs512',
            fm_config='NF_from_file_conditioning',
            ifos=['HL'])