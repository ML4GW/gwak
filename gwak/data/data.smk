rule find_valid_segments:
    output:
        save_path = 'output/data/segments.o4b-2.npy'
    script:
        'segments_intersection.py'

rule pull_data:
    input:
        config = 'data/configs/config.yaml',
        segments = rules.find_valid_segments.output
    shell:
        'python data/cli.py --config {input.config} \
            --segments {input.segments} '