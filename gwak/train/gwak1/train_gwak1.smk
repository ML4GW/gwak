import os

signalclasses = ['bbh', 'sine_gaussian', 'sine_gaussian_lf', 'sine_gaussian_hf', 'kink', 'kinkkink', 'white_noise_burst', 'gaussian', 'cusp']
backgroundclasses = ['background', 'glitches']
dataclasses = signalclasses + backgroundclasses

wildcard_constraints:
    datatype = '|'.join([x for x in dataclasses])

CLI = {
    'background': 'train/gwak1/cli_base.py',
    'glitches': 'train/gwak1/cli_base.py',
    'sine_gaussian_lf': 'train/gwak1/cli_signal_gwak1.py',
    'sine_gaussian_hf': 'train/gwak1/cli_signal_gwak1.py',
    'bbh': 'train/gwak1/cli_base.py',
    'sine_gaussian': 'train/gwak1/cli_signal.py',
    'kink': 'train/gwak1/cli_signal.py',
    'kinkkink': 'train/gwak1/cli_signal.py',
    'white_noise_burst': 'train/gwak1/cli_signal.py',
    'gaussian': 'train/gwak1/cli_signal.py',
    'cusp': 'train/gwak1/cli_signal.py',
    }

rule train_gwak1:
    input:
        config = 'train/gwak1/configs/gwak1/{datatype}.yaml'
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

rule train_linear_metric:
    input:
        config = 'train/gwak1/configs/linear_metric.yaml'
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

