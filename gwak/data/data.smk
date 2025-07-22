ifo_configs = [
    'hl',
    'hv',
    'lv',
    'hlv'
]
segment_types = [
    'original.o4b-2',
    'short-0.o4b-2',
    'short-1.o4b-2',
    'original.o4b-0',
    'short-0.o4b-0',
    'short-1.o4b-0',
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
    params:
        segments = 'data/segments/'
    output:
        save_path = 'output/data/segments.{segment_type}-{ifos}.npy'
    shell:
        'python data/segments_intersection.py \
            --folder-segments {params.segments} \
            --segment-type {wildcards.segment_type} \
            --ifos {wildcards.ifos} \
            --save-path {output.save_path}'

rule get_token:
    output: "tmp/token_ready.txt"
    shell:
        """
        echo " "
        echo " "
        echo "Get scitoken..."
        echo " "
        echo "    Check if any window pops up automatically."
        echo " "
        echo " "
        htgettoken -a vault.ligo.org -i igwn 
        echo "Token obtained at $(date)" > {output}
        """

rule pull_data:
    input:
        "tmp/token_ready.txt",
        config = 'data/configs/{segment_type}-{ifos}.yaml',
        segments = 'output/data/segments.{segment_type}-{ifos}.npy'
    output:
        'tmp/{segment_type}-{ifos}.log'
    shell:
        'python -u data/cli.py --config {input.config} \
            --segments {input.segments} \
            | tee {output}'

rule pull_all:
    input:
        expand(rules.pull_data.output,
            segment_type=['short-0.o4b-2', 'short-1.o4b-2', 'short-0.o4b-0', 'short-1.o4b-0'],
            ifos=['hl', 'hv', 'lv', 'hlv'])