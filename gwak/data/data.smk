ifo_configs = [
    'hl',
    'hv',
    'lv',
    'hlv'
]
wildcard_constraints:
    ifos = '|'.join([x for x in ifo_configs])

rule find_valid_segments:
    output:
        save_path = 'output/data/segments.original.o4b-2-{ifos}.npy'
    script:
        'segments_intersection.py'

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

rule pull_data:
    input:
        config = 'data/configs/O4b-2-{ifos}.yaml',
        segments = expand(rules.find_valid_segments.output, ifos='{ifos}')
    output:
        'tmp/{ifos}.log'
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} \
            | tee {output}'

rule pull_hl:
    input:
        expand(rules.pull_data.output, ifos='hl')

rule pull_hv:
    input:
        expand(rules.pull_data.output, ifos='hv')

rule pull_lv:
    input:
        expand(rules.pull_data.output, ifos='lv')

rule pull_hlv:
    input:
        expand(rules.pull_data.output, ifos='hlv')