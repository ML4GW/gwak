from pathlib import Path


benchmark_models = [
    'EM_IF_HL',
    'EM_NF_HL'
]

bbc_datasets = [
    'short-0',
    'short-1',
]
wildcard_constraints:
    benchmark_model = '|'.join(x for x in benchmark_models),
    bbc_dataset = '|'.join(x for x in bbc_datasets)

rule cuts:
    input:
        cuts_script = GWAK_ROOT / "gwak/postselection/make_correlation_cuts.py",
        error_config = BENCHMAKR_DIR / "{benchmark_model}/{bbc_dataset}/outlier_config-greedy.h5",
        # kernels = GWAK_ROOT / 'gwak/corrcuts_kernels_105t02wk.h5',
        data_dir = GWAK_ROOT / "gwak/output/BBC_AnalysisReady_Cat12/HL",
        thresholds = GWAK_ROOT / "gwak/postselection/postselection_thresholds_example.json",
    output:
        output_csv = BENCHMAKR_DIR / "{benchmark_model}/{bbc_dataset}_correlation_cuts-thresholds_example.csv",
    shell:
        'cd {GWAK_ROOT}/gwak/deploy/; \
        source .venv/bin/activate; \
        python {input.cuts_script} \
            --error-config {input.error_config} \
            --data-dir {input.data_dir} \
            --output {output.output_csv} \
            --thresholds {input.thresholds}'
            # --kernels {params.kernels} \


rule cuts_all:
    input: 
        expand(
            rules.cuts.output,
            model=models,
            bbc_dataset=bbc_datasets
        )