from pathlib import Path

rule cuts:
    params:
        cuts_script = GWAK_ROOT / 'gwak/postselection/make_correlation_cuts.py',
        error_config = OUTPUT_DIR / 'gwak-internal-benchmark/EM_IF_HL/short-0/outlier_config-greedy.h5',
        kernels = GWAK_ROOT / 'gwak/corrcuts_kernels_105t02wk.h5',
        data_dir = GWAK_ROOT / 'gwak/output/BBC_AnalysisReady_Cat12/HL',
    output:
        output_csv = OUTPUT_DIR / 'correlation_cuts.csv',
    shell:
        'cd {GWAK_ROOT}/gwak/train/; \
        source .venv/bin/activate; \
        python {params.cuts_script} \
            --error-config {params.error_config} \
            --data-dir {params.data_dir} \
            --kernels {params.kernels} \
            --output {output.output_csv}'
            