# HTCondor Training on LDAS

This profile is used to run GWAK training jobs with Snakemake and HTCondor on LDAS.

The `train_cl` rule requests:

- 1 GPU
- 4 CPU threads
- 32 GB of memory
- 32 GB of disk space

The profile uses `$USER` for the `LigoSearchUser` attribute. The `LigoSearchTag` and `SGNL_GPU` settings are specific to LDAS.

## Setup

Before submitting a job, initialize GWAK and make sure the paths in `setups/config.local.yaml` are correct.

Run the setup if it has not been completed:

```bash
snakemake -c1 gwak_init
```

Load the GWAK environment:

```bash
source .gwak/env.sh
```

## Submit a Training Job

The following example uses HL data and the SG/WNB training config.

First, set Weights & Biases to offline mode:

```bash
export WANDB_MODE=offline
export WANDB_SILENT=true
```

Set the model output path:

```bash
TARGET="$GWAK_OUTPUT_DIR/models/HL/O4b_cat12-katya/ResNet_6d_prior/model_JIT.pt"
```

Run a Snakemake dry-run to check the workflow:

```bash
snakemake --profile profiles/htcondor-ldas -n -p "$TARGET"
```

If the dry-run passes, remove `-n` to submit the training job to HTCondor:

```bash
snakemake --profile profiles/htcondor-ldas -p "$TARGET"
```

The training uses the number of epochs set in `ResNet_6d_prior.yaml`, which is currently 25.

## Notes

- This profile includes LDAS-specific HTCondor settings.
- GPU node requirements may need to be adjusted depending on the cluster.
- A one-epoch SG/WNB smoke test completed successfully using a local HTCondor profile (Cluster `71390923`).
- The exported TorchScript model was successfully loaded with `torch.jit.load()`.
- The shared profile has passed the Snakemake dry-run but has not yet been tested with a real GPU submission.