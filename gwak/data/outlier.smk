import json
import re
import yaml

def get_outlier_config_json(wildcards):
    name = wildcards.outlier_name
    path = omicron_check_config["outlier_configs"][name]
    return json.dumps({name: path})


OMICRON_CHECK_CONFIG_FILE = (
    GWAK_ROOT / "gwak/data/configs/outlier_omicron_triggers.yaml"
)
with OMICRON_CHECK_CONFIG_FILE.open(encoding="utf-8") as config_file:
    outlier_workflow_config = yaml.safe_load(config_file)

omicron_check_config = outlier_workflow_config["run_omicron_check"]
qtransform_config = outlier_workflow_config["run_outlier_qtransform"]
outlier_names = tuple(omicron_check_config["outlier_configs"])
outlier_ifos = tuple(omicron_check_config.get("ifos", ["H1", "L1"]))
qtransform_names = set(qtransform_config["outlier_configs"])
# if set(outlier_names) != qtransform_names:
#     raise ValueError(
#         "run_omicron_check and run_outlier_qtransform must configure "
#         "the same outlier names"
#     )

OMICRON_CHECK_OUTPUT_DIR = (
    Path(omicron_check_config.get("out_dir", OUTPUT_DIR))
    / "omicron_check"
    / omicron_check_config["project_name"]
)
QTRANSFORM_OUTPUT_DIR = (
    Path(qtransform_config.get("out_dir", OUTPUT_DIR))
    / "omicron_check"
    / qtransform_config["project_name"]
)

wildcard_constraints:
    outlier_name = '|'.join(re.escape(x) for x in outlier_names),
    ifo = '|'.join(re.escape(x) for x in outlier_ifos)



# Check omicron triggers with outlier configs
rule outlier_omicron_check:
    input:
        arg = GWAK_ROOT / "gwak/data/data/cli.py",
        config = OMICRON_CHECK_CONFIG_FILE
    output:
        trigger = (
            OMICRON_CHECK_OUTPUT_DIR / "{outlier_name}/{ifo}_triggers.h5"
        )
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/data/pyproject.toml",
        outlier_config_json=lambda wildcards: json.dumps(
            {wildcards.outlier_name: omicron_check_config["outlier_configs"][wildcards.outlier_name]}
        ),
        ifos_json = lambda wildcards: json.dumps( [wildcards.ifo] )
    shell:
        "source {params.gwak_env}; \
        uv run \
            --project {params.pyproject} python {input.arg} \
            --config {input.config} run_omicron_check \
            --outlier_configs {params.outlier_config_json:q} \
            --ifos {params.ifos_json:q}"


rule outlier_omicron_check_all:
    input:
        expand(
            str(OMICRON_CHECK_OUTPUT_DIR / "{outlier_name}/{ifo}_triggers.h5"),
            outlier_name=outlier_names,
            ifo=outlier_ifos,
        )


# Create Q-transform spectrograms for the triggers selected by Omicron check
rule outlier_qtransform:
    input:
        arg = GWAK_ROOT / "gwak/data/data/cli.py",
        config = OMICRON_CHECK_CONFIG_FILE,
        triggers = lambda wildcards: expand(
            str(OMICRON_CHECK_OUTPUT_DIR / "{outlier_name}/{ifo}_triggers.h5"),
            outlier_name=wildcards.outlier_name,
            ifo=outlier_ifos,
        )
    output:
        QTRANSFORM_OUTPUT_DIR / "{outlier_name}/qtransform.h5"
    params:
        gwak_env = GWAK_ROOT / ".gwak/env.sh",
        pyproject = GWAK_ROOT / "gwak/data/pyproject.toml",
        outlier_config_json=lambda wildcards: json.dumps(
            {wildcards.outlier_name: qtransform_config["outlier_configs"][wildcards.outlier_name]}
        ),
    shell:
        "source {params.gwak_env}; \
        uv run \
            --project {params.pyproject} python {input.arg} \
            --config {input.config} run_outlier_qtransform \
            --outlier_configs {params.outlier_config_json:q}"


rule outlier_qtransform_all:
    input:
        expand(
            str(QTRANSFORM_OUTPUT_DIR / "{outlier_name}/qtransform.h5"),
            outlier_name=outlier_names,
        )
