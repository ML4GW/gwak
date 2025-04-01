import os

signalclasses = ['bbh', 'sine_gaussian', 'sine_gaussian_lf', 'sine_gaussian_hf', 'kink', 'kinkkink', 'white_noise_burst', 'gaussian', 'cusp']
backgroundclasses = ['background', 'glitches']
multiSignals = ['SimCLR_multiSignal_all']
dataclasses = signalclasses+backgroundclasses+multiSignals

wildcard_constraints:
    datatype = '|'.join([x for x in dataclasses])

CLI = {
    'background': 'train/cli_base.py',
    'glitches': 'train/cli_base.py',
    'sine_gaussian_lf': 'train/cli_signal_gwak1.py',
    'sine_gaussian_hf': 'train/cli_signal_gwak1.py',
    'bbh': 'train/cli_base.py',
    'sine_gaussian': 'train/cli_signal.py',
    'kink': 'train/cli_signal.py',
    'kinkkink': 'train/cli_signal.py',
    'white_noise_burst': 'train/cli_signal.py',
    'gaussian': 'train/cli_signal.py',
    'cusp': 'train/cli_signal.py',
    'SimCLR_multiSignal_all': 'train/cli_multiSignal.py'
    }

rule train_gwak1:
    input:
        config = 'train/configs/gwak1/{datatype}.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.datatype]
    log:
        artefact = directory('output/gwak1/{datatype}/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train_gwak1_all:
    input:
        expand(rules.train_gwak1.log, datatype='background')

rule train:
    input:
        config = 'train/configs/{datatype}.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.datatype]
    log:
        artefact = directory('output/{datatype}/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train_all:
    input:
        expand(rules.train.log, datatype=['white_noise_burst', 'kinkkink', 'gaussian', 'cusp'])

rule SimCLR_multiSignal_all:
    input:
        expand('output/{datatype}',datatype=['SimCLR_multiSignal_all'])

rule train_linear_metric:
    input:
        config = 'train/configs/linear_metric.yaml'
    params:
        cli = lambda wildcards: CLI[wildcards.datatype]
    log:
        artefact = directory('output/linear_metric/{datatype}/')
    shell:
        'python {params.cli} fit --config {input.config} \
            --trainer.logger.save_dir {log.artefact}'

rule train_linear_all:
    input:
        expand(rules.train_linear_metric.log, datatype=['SimCLR_multiSignal_all'])

