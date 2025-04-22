rule find_valid_segments:
    output:
        save_path = 'output/data/segments.o4b-2.npy'
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
        config = 'data/configs/O4b-2.yaml',
        segments = rules.find_valid_segments.output
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '