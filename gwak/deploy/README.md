# Deploy your trained NN model #
The deploy projcet provides three major utilities:
- Exporting trained NN model to excutibles
- Spins up the Triton server and handle the job via HTCondor or Slurm. 
- Search outlier and run injction tests.

Follow the following command to install or update the deploy enviroment.

# Enviroment #
We use [uv](https://docs.astral.sh/uv/) to manage our enviroment. One benefit is that the enviroment will be install when you run the job via uv for the first time. So there is no need to run poetry, conda or whatever install before the run. <br>
Run the following command to interact with your project enviroment and uv. <br>(These command should run under `$GWAK_DIR/gwak/deploy`)
```
$ uv run python --version            # Check the python version
$ uv add <pacakge_name>          # Install new package 
$ uv sync                                      # Resolve the depedcies version and update the uv.lock 
$ uv venv 
$ source .venv/bin/activate          # Activate the virtual enviroment 
```


# Use the following command to analysis your trained model #
The following should under `$GWAK_DIR/gwak/` directory. 

The allowed snakemake key words includes `cl_config`, `fm_config`, `ifo_mode`, and `run_name`.

- `cl_config`: Embedding model. (ResNet, S4, ...)
- `fm_config`: Metric model. (NF, NPLM, ...)
- `ifo_mode`: Combination of different interferometers. (HL, LV, ...)
- `run_name`: Name of the specific run. (test_run, one_year, ...)


- Export a model for Triton inference.
```
$ snakemake -c1 $GWAK_OUTPUT_DIR/export/{cl_config}_{fm_config}_{ifo_mode}
```
- Spin up the Triton server and run background, timeslide, or injection analysis.
```
$ snakemake -c1 $GWAK_OUTPUT_DIR/infer/{cl_config}_{fm_config}_{ifo_mode}/{run_name}
```
- Scan outlier and produce plots. 
```
$ snakemake -c1 $GWAK_LOUVRE_DIR/{cl_config}_{fm_config}_{ifo_mode}/{run_name}
```
- Use the following command to run Export + Inference + Plotting in one command
```
$ snakemake -c1 scan_all -F
```

# Help 
Use the following commad if VScode did not recognize the jupyter kernel correctly. 
```
$ uv venv                                     # Create a .venv/ file under you project 
$ source .venv/bin/activate          # Activate the virtual enviroment 
$ which python                            # Copy paste the python path and use it as jupyter kernel
```