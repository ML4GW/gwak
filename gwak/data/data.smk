ifo_configs = [
    'hl',
    'hv',
    'lv',
    'hls'
]
segment_types = [
    'original.o4b-2',
    'short-0',
    'short-1',
]
wildcard_constraints:
    ifos = '|'.join([x for x in ifo_configs]),
    segment_type = '|'.join([x for x in segment_types])


rule pull_O3a_data:
    input:
        config = 'data/configs/O3a.yaml',
        segments = 'output/data/segments.O3a.npy'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '

rule pull_O3b_data:
    input:
        config = 'data/configs/O3b.yaml',
        segments = 'output/data/segments.O3b.npy'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '

rule find_valid_segments:
    output:
        save_path = 'output/data/segments.{segment_type}-{ifos}.npy'
    script:
        'segments_intersection.py'

rule pull_data:
    input:
        config = 'data/configs/{segment_type}-{ifos}.yaml',
        segments = 'output/data/segments.{segment_type}-{ifos}.npy'
    output:
        'tmp/{segment_type}-{ifos}.log'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} \
            | tee {output}'

rule pull_hl:
    input:
        expand(rules.pull_data.output,
            segment_type='short-1',
            ifos='hl')

rule pull_hv:
    input:
        expand(rules.pull_data.output,
            segment_type='short-0',
            ifos='hv')

rule pull_lv:
    input:
        expand(rules.pull_data.output,
            segment_type='short-0',
            ifos='lv')

rule pull_hlv:
    input:
        expand(rules.pull_data.output,
            segment_type='short-0',
            ifos='hlv')