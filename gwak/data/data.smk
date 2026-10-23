ifo_configs = [
    'HL',
    'HV',
    'LV',
    'HLV'
]
segment_types = [
    "o4.strain",
    'o4b.bbc-background-0',
    'o4b.bbc-background-2',
    'o4b.short-0-0',
    'o4b.short-1-0',
    'o4b.short-0-2',
    'o4b.short-1-2',
]
wildcard_constraints:
    ifos = '|'.join([x for x in ifo_configs]),
    segment_type = '|'.join([x for x in segment_types])

rule pull_O3a_data:
    input:
        config = GWAK_ROOT / 'gwak/data/configs/O3a.yaml',
        segments = OUTPUT_DIR / 'data/segments.O3a.npy'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '

rule pull_O3b_data:
    input:
        config = GWAK_ROOT / 'gwak/data/configs/O3b.yaml',
        segments = OUTPUT_DIR / 'data/segments.O3b.npy'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '


# Step 1: resolve the segments of a config 
# into an (N, 2) [start, end] array. 
rule get_segment_list:
    input:
        arg = GWAK_ROOT / "gwak/data/data/cli.py",
        config = GWAK_ROOT / 'gwak/data/configs/{segment_type}-{ifos}.yaml',
    output:
        segments = OUTPUT_DIR / 'data/segments.{segment_type}-{ifos}.npy'
    log:
        LOG_DIR / 'data/segments.{segment_type}-{ifos}.log'
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/data/pyproject.toml",
    shell:
        "source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} \
            --config {input.config} make_seg_list \
            --segment_type {wildcards.segment_type} \
            --resolved_segments {output.segments} \
            --logger {log}"

# Step 2: download the strain of every segment in the list.
rule pull_data_from_segments:
    input:
        arg = GWAK_ROOT / "gwak/data/data/cli.py",
        config = GWAK_ROOT / 'gwak/data/configs/{segment_type}-{ifos}.yaml',
        segments = rules.get_segment_list.output.segments,
    output:
        logger = LOG_DIR / 'data/{segment_type}-{ifos}-0.log'
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/data/pyproject.toml",
    shell:
        "source {params.gwak_env}; uv run \
            --project {params.pyproject} python {input.arg} \
            --config {input.config} get_strain \
            --segments {input.segments} \
            --logger {output.logger}"
