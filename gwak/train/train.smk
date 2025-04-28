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
    output:
        model = 'output/{cl_config}_{ifos}/model_JIT.pt'
    params:
        artefact = directory('output/{cl_config}_{ifos}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.init_args.data_dir {input.data_dir} \
            --data.ifos {wildcards.ifos}'

rule train_fm:
    input:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='{cl_config}',
            ifos='{ifos}'),
        config = 'train/configs/{fm_config}.yaml',
        data_dir = '/home/katya.govorkova/gwak2/gwak/output/O4_MDC_background/{ifos}/'
    output:
        model = 'output/{cl_config}_{fm_config}_{ifos}/model_JIT.pt'
    params:
        artefact = directory('output/{cl_config}_{fm_config}_{ifos}/')
    shell:
        'python train/cli_fm.py fit --config {input.config} \
            --trainer.logger.save_dir {params.artefact} \
            --data.init_args.data_dir {input.data_dir} \
            --data.ifos {wildcards.ifos}'

rule make_plots:
    input:
        embedding_model = expand(rules.train_cl.output.model,
            cl_config='S4_SimCLR_multiSignalAndBkg',
            ifos='HL'),
        fm_model = expand(rules.train_fm.output.model,
            fm_config='NF_onlyBkg',
            cl_config='S4_SimCLR_multiSignalAndBkg',
            ifos='HL'),
        data_dir = '/home/katya.govorkova/gwak2/gwak/output/O4_MDC_background/{ifos}/',
        config = 'train/configs/S4_SimCLR_multiSignalAndBkg.yaml'
    output:
        directory('output/plots/')
    shell:
        'mkdir {output}; '
        'python train/plots.py \
            --fm-model {input.fm_model} \
            --data-dir {input.data_dir} \
            --config {input.config} \
            --output {output} '
            # '--use-freq-correlation '
