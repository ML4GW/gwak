ifo_modes = [
    'H', 'L', 'V', 'K',
    'HL', 'HV', 'HK', 'LV', 'LK', 'VK',
    'HLV', 'HLK', 'HVK', 'LVK', 
    'HLVK'
]

ana_ver_path_converter = {
    "O4b_gwak_cat1": "O4_MDC_background", # Have omicron file
    "O4b_gwak_cat12": "BBC_AnalysisReady_Cat12",
}

noise_runs = [
    'background', 'test_run',
    'one_day', 'one_month', 'one_year', 
    'one_decade','one_centure',
]

foreground_runs = [
    'bbc-short-0', 'bbc-short-1', 
    'injections'
]

runs = noise_runs + foreground_runs

benchmark_models = [
    "EM_NF_HL",
    "EM_IF_HL",
]

wildcard_constraints:
    ifo_mode = '|'.join(x for x in ifo_modes),
    ana_ver = '|'.join([x for x in ana_ver_path_converter.keys()]),
    data_ver = '|'.join([x for x in data_ver_path_converter.keys()]),
    noise_run = '|'.join(x for x in noise_runs),
    foreground_run = '|'.join(x for x in foreground_runs),
    run_name = '|'.join(x for x in runs),
    benchmark_model = '|'.join(x for x in benchmark_models)


runs_TS_converter = {
    'background': 0, 'test_run': 1, 'one_day': 86400, 
    'one_month': 2678400, 'one_year': 31557600, 
    'one_decade': 315576000, 'one_centure': 3155760000,
    'bbc-short-0': 0, 'bbc-short-1': 0, 
    'injections': 0
}

ts_pair_for_run = {
    f"{run}_{ts_run}": ts_run
    for run in foreground_runs
    for ts_run in noise_runs
}

ts_pair_for_run.update(
    {f"{run}_{run}": run for run in noise_runs}
)

bm_model_threshold_converter = {
    "EM_NF_HL": -10,
    "EM_IF_HL": -0.7
}

# snakemake -c1 $GWAK_OUTPUT_DIR/export/{cl_config}_{fm_config}_{ifo_mode}/combination
rule export:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/export.yaml",
    output:
        directory(
            OUTPUT_DIR / "export"
            / "{ifo_mode}/{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} export \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} "


# python deploy/cli.py export --config deploy/config/export.yaml --project combination
# Consider adding a swap key for the I/O searching in the Pathfinder
rule production_export:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        image = IMAGE_DIR / "deploy.sif",
        config = GWAK_ROOT / "gwak/deploy/configs/export.yaml"
    output:
        directory(
            CONTAINER_OUTPUT_DIR / "export"
            / "{ifo_mode}/{data_ver}/{cl_config}_{coh_mode}_{fm_config}"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        bind_1 = f"{CONTAINER_OUTPUT_DIR}:/production",
        bind_2 = f"{OUTPUT_DIR}:/opt/gwak/gwak/output",
    shell: 
        "source {params.gwak_env}; set -x; apptainer exec --nv \
        --bind {params.bind_1},{params.bind_2} \
        {input.image} \
        python {input.arg} export  \
        --config {input.config} \
        --project combination"

# snakemake -c1 $GWAK_OUTPUT_DIR/infer/{cl_config}_{fm_config}_{ifo_mode}/{run_name}
rule condor_infer:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/infer_condor.yaml",
        plan_model = rules.export.output
    output:
        directory(
            OUTPUT_DIR / "infer"
            / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{run_name}/inference_result"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
        ana_data = lambda wildcards: ana_ver_path_converter[wildcards.ana_ver],
        timeslide = lambda wildcards: runs_TS_converter[wildcards.run_name],
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} infer_condor \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --ana_data {params.ana_data} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --run_name {wildcards.run_name} \
            --Tb {params.timeslide}"
            # --fname.init_args.suffix test \


rule slurm_infer:
    input:
        config = 'deploy/configs/infer_slurm.yaml',
    params:
        timeslide = lambda wildcards: runs_TS_converter[wildcards.run_name]
    output:
        artefact = directory(
            OUTPUT_DIR / "Slurm_Jobs"
            / "{cl_config}_{fm_config}_{ifo_mode}/{run_name}"
        )
    shell:
        "set -x; cd gwak/deploy; uv run python \
        deploy/cli.py deploy \
        --config ../{input.config} \
        --cl_config {wildcards.cl_config} \
        --fm_config {wildcards.fm_config} \
        --ifo_mode {wildcards.ifo_mode} \
        --run_name {wildcards.run_name} \
        --Tb {params.timeslide}"


##--- Summary ---##
rule export_all:
    input:
        expand(
            rules.export.output,
            ifo_mode=["HL"],
            data_ver=["O4b_cat12-katya"],
            cl_config=["ResNet_6d"],
            coh_mode=["real"],
            fm_config=["NF_from_file_conditioning"],
        )

# snakemake -c4 output/Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}/ -F
rule slurm_infer_all:
    input:
        expand(
            rules.slurm_infer.output,
            cl_config=[
                "ResNet_cat12",
                "ResNet_separate-glitch",
                "torch_rbw_zp_resnet_do6_dcs128_epoch25",
            ], 
            fm_config=[
                "NF_from_file_conditioning",
            ], 
            ifo_mode=["HL"], 
            run_name=["one_year"]
        )

# #####################
# ### Post-analysis ###
# #####################
rule threshold_lock:
    input: 
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/threshold.yaml",
        # infer_result = rules.condor_infer.output
    output:
        Path(
            LOG_DIR / "infer/{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{run_name}"
            / "threshold_lock.log"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} threshold_lock \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --run_name {wildcards.run_name}"

rule scan_outlier:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/scan_outlier.yaml",
        log = Path(
            LOG_DIR / "infer/{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{noise_run}"
            / "threshold_lock.log"
        )
    output:
        Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{run_name}_{noise_run}"
            / "scan_outlier.log"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} scan_outlier \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --run_name {wildcards.run_name} \
            --threshold_setting {wildcards.noise_run}"

rule bbc_benchmark:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/benchmark.yaml",
        task_1 = Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/bbc-short-0_{noise_run}"
            / "scan_outlier.log"
        ),
        task_2 = Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/bbc-short-1_{noise_run}"
            / "scan_outlier.log"
        )
    output:
        Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}" 
            / "{cl_config}_{coh_mode}_{fm_config}"
            / "{foreground_run}_{noise_run}/benchmark.log"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} resolve_O4_bbc \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --foreground {wildcards.foreground_run} \
            --threshold_setting {wildcards.noise_run}"

rule find_outlier_segs:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/plot_segs.yaml",
        task = Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{noise_run}_{noise_run}"
            / "scan_outlier.log"
        ),
    output:
        Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{noise_run}"
            / "find_outlier_segs.log"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} plot_segs \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --threshold_setting {wildcards.noise_run}"

rule plot_bbc_benchmark:
    input:
        arg = GWAK_ROOT / "gwak/deploy/deploy/cli.py",
        config = GWAK_ROOT / "gwak/deploy/configs/plot_bbc.yaml",
        task_1 = Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/bbc-short-0_{noise_run}"
            / "benchmark.log"
        ),
        task_2 = Path(
            LOG_DIR / "{ifo_mode}/{ana_ver}/{data_ver}" 
            / "{cl_config}_{coh_mode}_{fm_config}/bbc-short-1_{noise_run}"
            / "benchmark.log"
        )
    output:
        # artefact = LOUVRE_DIR / "{cl_config}_{coh_mode}_{fm_config}_{ifo_mode}/{noise_run}/trigger-rate.png"
        Path(
            LOUVRE_DIR / "{ifo_mode}/{ana_ver}/{data_ver}"
            / "{cl_config}_{coh_mode}_{fm_config}/{noise_run}"
            / "trigger-rate.png"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/deploy/pyproject.toml",
    shell:
        "source {params.gwak_env}; set -x; uv run \
            --project {params.pyproject} python {input.arg} plot_bbc \
            --config {input.config} \
            --ifo_mode {wildcards.ifo_mode} \
            --ana_ver {wildcards.ana_ver} \
            --data_ver {wildcards.data_ver} \
            --cl_config {wildcards.cl_config} \
            --coh_mode {wildcards.coh_mode} \
            --fm_config {wildcards.fm_config} \
            --threshold_setting {wildcards.noise_run}"


rule estimate_far:
    input:
        path_to_infer = '/home/hongyin.chen/anti_gravity/gwak/gwak/output/infer/combination/inference_result/'# expand(rules.infer.output, deploymodels='combination')
    output:
        'output/infer/far_metrics.npy'
    shell:
        'python deploy/save_far.py \
            {input.path_to_infer} \
            --dataset data \
            --duration 0.5 \
            --num_thresholds 100 \
            --outfile {output} \
            --direction negative '