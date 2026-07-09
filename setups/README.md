# Setup Configuration

`setups/setup.smk` now reads setup values from two config layers:

1. `setups/config.yaml`
   This is the tracked default configuration for the repository.
2. `setups/config.local.yaml`
   This is an optional local override for machine-specific changes. It is gitignored.

If both files exist, local values override the defaults. Command-line Snakemake config still wins over both.

## Files

- `setups/config.yaml`: shared defaults for the repo
- `setups/config.local.yaml.example`: example local override
- `setups/config.local.yaml`: your private local override after copying the example

Create the local file with:

```bash
cp setups/config.local.yaml.example setups/config.local.yaml
```

## Path Settings

Edit the `paths` section to change where GWAK writes data, logs, plots, and container artifacts.

Example:

```yaml
paths:
  gwak_output_dir: /tmp/gwak-output
  image_dir: ~/scratch/containers/gwak
  contain_output_dir: ~/scratch/containers/gwak
```

These values are written into:

- `.gwak/paths.json`
- `.gwak/env.sh`

If you want your interactive shell to use the same exported variables, run:

```bash
source .gwak/env.sh
```

## PATH Prepend

If you want `gwak_init` to generate a shell file that prepends entries to your shell `PATH`, set `shell_path_prepend`.

Example:

```yaml
shell_path_prepend:
  - ~/miniconda3/bin
  - ~/bin
```

When present, `.gwak/env.sh` will include a matching `export PATH=...:$PATH`.

## Conda Bootstrap Toggle

`gwak_init` can now decide whether to run `bootstrap_conda_data_env`.

Use this in either config file:

```yaml
run_bootstrap_conda_data_env: true
```

or:

```yaml
run_bootstrap_conda_data_env: false
```

Behavior:

- `true`: `snakemake -c1 gwak_init` also creates or updates the `gwak-data` conda environment
- `false`: `snakemake -c1 gwak_init` skips the conda environment bootstrap and still writes the GWAK path state

You can also override from the command line:

```bash
snakemake -c1 gwak_init --config run_bootstrap_conda_data_env=false
```

## Shared vs Local Changes

Use `setups/config.yaml` for defaults the team should share.

Use `setups/config.local.yaml` for:

- personal filesystem locations
- machine-specific container directories
- local `PATH` adjustments
- skipping conda bootstrap on a machine where the env is already managed another way
