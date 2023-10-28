from config import VERSION

signalclasses = ['bbh', 'sglf', 'sghf']
backgroundclasses = ['background', 'glitches']
modelclasses = signalclasses + backgroundclasses
fm_training_classes = [
    'bbh_fm_optimization',
    'sghf_fm_optimization',
    'sglf_fm_optimization',
    'supernova_fm_optimization',
    'wnbhf_fm_optimization',
    'wnblf_fm_optimization'
    ]
dataclasses = fm_training_classes+[
    'wnblf',
    'wnbhf',
    'supernova',
    'timeslides',
    'bbh_varying_snr',
    'sghf_varying_snr',
    'sglf_varying_snr',
    'wnbhf_varying_snr',
    'wnblf_varying_snr',
    'supernova_varying_snr']

wildcard_constraints:
    modelclass = '|'.join([x for x in modelclasses]),
    dataclass = '|'.join([x for x in dataclasses + modelclasses])


rule find_valid_segments:
    input:
        hanford_path = 'data/O3a_Hanford_segments.json',
        livingston_path = 'data/O3a_Livingston_segments.json'
    params:
        save_path = 'output/O3a_intersections.npy'
    script:
        'scripts/segments_intersection.py'

rule run_omicron:
    input:
        intersections = rules.find_valid_segments.params.save_path
    params:
        user_name = 'katya.govorkova',
        folder = 'output/omicron/'
    shell:
        'mkdir -p {params.folder}; '
        'ligo-proxy-init {params.user_name}; '
        'python3 scripts/run_omicron.py {input.intersections} {params.folder}'

rule fetch_site_data:
    input:
        omicron = rules.run_omicron.params.folder,
        intersections = rules.find_valid_segments.params.save_path
    output:
        'tmp/dummy_{site}.txt'
    shell:
        'touch {output}; '
        'python3 scripts/fetch_data.py {input.omicron} {input.intersections}\
            --site {wildcards.site}'

rule generate_data:
    input:
        omicron = 'output/omicron/',
        intersections = rules.find_valid_segments.params.save_path,
    params:
        dependencies = expand(rules.fetch_site_data.output,
                              site=['L1', 'H1'])
    output:
        file = 'output/data/{dataclass}.npz'
    shell:
        'python3 scripts/generate.py {input.omicron} {output.file} \
            --stype {wildcards.dataclass} \
            --intersections {input.intersections}'

rule upload_data:
    input:
        expand(rules.generate_data.output.file,
               dataclass='{dataclass}')
    params:
        '/home/katya.govorkova/gwak/{version}/data/{dataclass}.npz'
    shell:
        'mkdir -p /home/katya.govorkova/gwak/{wildcards.version}/data/; '
        'cp {input} {params}; '

rule validate_data:
    input:
        expand(rules.upload_data.params,
               dataclass=modelclasses+dataclasses,
               version=VERSION)
    shell:
        'mkdir -p data/{VERSION}/; '
        'python3 scripts/validate_data.py {input}'

rule train_quak:
    input:
        data = expand(rules.upload_data.params,
                      dataclass='{dataclass}',
                      version=VERSION)
    output:
        savedir = directory('output/trained/{dataclass}'),
        model_file = 'output/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.model_file} {output.savedir} '

rule upload_models:
    input:
        expand(rules.train_quak.output.model_file,
               dataclass='{dataclass}')
    output:
        '/home/katya.govorkova/gwak/{version}/models/{dataclass}.pt'
    shell:
        'mkdir -p /home/katya.govorkova/gwak/{wildcards.version}/models/; '
        'cp {input} {output}; '

rule generate_timeslides_for_far:
    params:
        data_path = expand(rules.upload_data.params,
            dataclass='timeslides',
            version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        shorten_timeslides = False,
        save_path = directory('output/timeslides_{id}/')
    output:
        save_evals_path = directory('output/timeslides_{id}/evals/'),
        save_normalizations_path = directory('output/timeslides_{id}/normalization/')
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/evaluate_timeslides.py {params.save_path} {params.model_path} \
            --data-path {params.data_path} \
            --save-evals-path {output.save_evals_path} \
            --save-normalizations-path {output.save_normalizations_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.id}'

rule evaluate_signals:
    params:
        source_file = expand(rules.upload_data.params,
                             dataclass='{signal_dataclass}',
                             version=VERSION),
        model_path = expand(rules.train_quak.output.model_file,
                            dataclass=modelclasses,
                            version=VERSION)
    output:
        save_file = 'output/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {params.source_file} {output.save_file} {params.model_path}'

rule generate_timeslides_for_fm:
    input:
        data_path = expand(rules.upload_data.params,
            dataclass='timeslides',
            version=VERSION),
    params:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        shorten_timeslides = True,
        save_path = directory('output/timeslides/')
    output:
        save_evals_path = directory('output/timeslides/evals/'),
        save_normalizations_path = directory('output/timeslides/normalization/')
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/evaluate_timeslides.py {params.save_path} {params.model_path} \
            --data-path {input.data_path} \
            --save-evals-path {output.save_evals_path} \
            --save-normalizations-path {output.save_normalizations_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} '

rule train_final_metric:
    input:
        signals = expand(rules.evaluate_signals.output.save_file,
                         signal_dataclass=fm_training_classes),
        dependencies = rules.generate_timeslides_for_fm.output,
        timeslides = 'output/timeslides/evals/',
        normfactors = 'output/timeslides/normalization/'
    params:
        params_file = '/home/katya.govorkova/gw-anomaly/output/trained/final_metric_params.npy',
        norm_factor_file = '/home/katya.govorkova/gw-anomaly/output/trained/norm_factor_params.npy',
        fm_model_path = '/home/katya.govorkova/gw-anomaly/output/trained/fm_model.pt'
    shell:
        'python3 scripts/final_metric_optimization.py {params.params_file} \
            {params.fm_model_path} {params.norm_factor_file} \
            --timeslide-path {input.timeslides} \
            --signal-path {input.signals} \
            --norm-factor-path {input.normfactors}'

rule recreation_and_quak_plots:
    input:
        fm_model_path = rules.train_final_metric.params.fm_model_path
    params:
        models = expand(rules.train_quak.output.model_file,
                        dataclass=modelclasses,
                        version=VERSION),
        test_path = expand(rules.upload_data.params,
                           dataclass='bbh',
                           version=VERSION),
        savedir = directory('output/paper/')
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/rec_and_quak_plots.py {params.test_path} {params.models} \
            {input.fm_model_path} {params.savedir}'

rule compute_far:
    input:
        metric_coefs_path = rules.train_final_metric.params.params_file,
        norm_factors_path = rules.train_final_metric.params.norm_factor_file,
        fm_model_path = rules.train_final_metric.params.fm_model_path
    params:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        data_path = expand(rules.generate_timeslides_for_far.output.save_evals_path,
            id='{far_id}'),
        shorten_timeslides = False,
    output:
        save_path = 'output/far_bins_{far_id}.npy'
    shell:
        'python3 scripts/evaluate_timeslides.py {output.save_path} {params.model_path} \
            --data-path {params.data_path} \
            --fm-model-path {input.fm_model_path} \
            --metric-coefs-path {input.metric_coefs_path} \
            --norm-factor-path {input.norm_factors_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.far_id}'

rule merge_far_hist:
    input:
        expand(rules.compute_far.output.save_path, far_id=[0,1,2,3])
    params:
        save_path = 'output/far_bins.npy'
    script:
        'scripts/merge_far_hist.py'

rule quak_plotting_prediction_and_recreation:
    input:
        test_data = expand(rules.upload_data.params,
                           dataclass='{dataclass}',
                           version=VERSION)
    params:
        model_path = expand(rules.train_quak.output.model_file,
                            dataclass=modelclasses,
                            version=VERSION),
        reduce_loss = False,
        save_file = 'output/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {params.save_file} {params.reduce_loss} \
            --model-path {params.model_path} '

rule plot_results:
    input:
        dependencies = [rules.merge_far_hist.params.save_path,
            expand(rules.evaluate_signals.output, signal_dataclass=fm_training_classes)],
        fm_model_path = rules.train_final_metric.params.fm_model_path
    params:
        evaluation_dir = 'output/',
        save_path = directory('output/paper/')
    shell:
        'mkdir -p {params.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {params.save_path} \
            {input.fm_model_path}'

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'