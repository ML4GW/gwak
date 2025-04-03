import os

train_configs = ['Transformer_SimCLR_multiSignal_all', 'Transformer_SimCLR_multiSignalAndBkg_noSG', 'S4_SimCLR_multiSignalAndBkg']
wildcard_constraints:
    train_config = '|'.join([x for x in train_configs])


rule train_cl:
    input:
        config = 'train/configs/{train_config}.yaml'
    params:
        cli = 'train/cli.py'
    log:
        artefact = directory('output/{train_config}/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule S4_SimCLR_multiSignalAndBkg:
    input:
        expand('output/{train_config}', train_config=['S4_SimCLR_multiSignalAndBkg'])

rule train_linear_metric:
    input:
        config = 'train/configs/linear_metric.yaml'
    params:
        cli = 'train/cli.py'
    log:
        artefact = directory('output/linear_metric/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'
