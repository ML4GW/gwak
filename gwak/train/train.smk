import os

train_configs = [
    'NF_onlyBkg',
    'FM_multiSignalAndBkg',
    'Transformer_SimCLR_multiSignal_all',
    'Transformer_SimCLR_multiSignalAndBkg_noSG',
    'S4_SimCLR_multiSignalAndBkg'
    ]
wildcard_constraints:
    train_config = '|'.join([x for x in train_configs])


rule train_cl:
    input:
        config = 'train/configs/{train_config}.yaml'
    log:
        artefact = directory('output/{train_config}/')
    shell:
        'python train/cli.py fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train:
    input:
        expand('output/{train_config}', train_config=['NF_onlyBkg'])
