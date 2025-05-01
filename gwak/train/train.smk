import os

cl_configs = [
    'Transformer_SimCLR_multiSignal_all',
    'Transformer_SimCLR_multiSignalAndBkg_noSG',
    'S4_SimCLR_multiSignalAndBkg'
    ]
fm_configs = [
    'NF_onlyBkg',
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
        data_dir = '/n/netscratch/iaifi_lab/Lab/emoreno/O4_MDC_background/{ifos}/'
        #data_dir = '/n/netscratch/iaifi_lab/Lab/sambt/LIGO/O4_MDC_background/{ifos}/'
        #data_dir = '/n/netscratch/iaifi_lab/Lab/emoreno/O4_MDC_background/train/'
    output:
        model = 'output/{cl_config}_{ifos}/model_JIT.pt'
    params:
        artefact = directory('output/{cl_config}_{ifos}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.init_args.data_dir {input.data_dir} \
            --data.init_args.ifos {wildcards.ifos}'

rule train_fm:
    input:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='{cl_config}',
            ifos='{ifos}'),
        config = 'train/configs/{fm_config}.yaml',
        data_dir = '/home/eric.moreno/gwak2_temp/gwak/gwak/output/O4_MDC_background/{ifos}/'
    output:
        model = 'output/{cl_config}_{fm_config}_{ifos}/model_JIT.pt'
    params:
        artefact = directory('output/{cl_config}_{fm_config}_{ifos}/')
    shell:
        'python train/cli_fm.py fit --config {input.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.init_args.data_dir {input.data_dir} \
            --data.ifos {wildcards.ifos} \
            --model.embedding_model {input.embedding_model}'

rule combine_models:
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
            --batch_size 512 \
            --kernel_length 0.5 \
            --sample_rate 4096 \
            --num_ifos 2 \
            --outfile {output} '

rule make_plots:
    input:
        combined_model = expand(rules.combine_models.output,
            cl_config='S4_SimCLR_multiSignalAndBkg',
            fm_config='NF_onlyBkg',
            ifos='HV'),
    params:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='S4_SimCLR_multiSignalAndBkg',
            ifos='HV'),
        fm_model = expand(rules.train_fm.output.model,
            fm_config='NF_onlyBkg',
            cl_config='S4_SimCLR_multiSignalAndBkg',
            ifos='HV'),
        data_dir = 'output/O4_MDC_background/HV/',
        config = 'train/configs/S4_SimCLR_multiSignalAndBkg.yaml'
    output:
        directory('output/plots/')
    shell:
        'mkdir {output}; '
        'python train/plots.py \
            --combined-model {input.combined_model} \
            --embedding-model {params.embedding_model} \
            --fm-model {params.fm_model} \
            --data-dir {params.data_dir} \
            --config {params.config} \
            --output {output} '
            # '--use-freq-correlation '
