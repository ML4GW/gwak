# Deploy your trained NN model #
The deploy application provides two major utilities:
- Exporting trained NN model to excutibles
- Producing inference results on sequential data

# Use the following command to deploy your NN model #
### Via ```Snakemake``` ###

- Export
```
$ cd gwak/gwak
$ snakemake -c1 export_all -F
```
- Inference
```
$ cd gwak/gwak
$ snakemake -c1 infer_all -F
```

- Deploy: (Export + Inference using Slurm)
```
$ cd gwak/gwak
$ snakemake -c1 deploy_all -F
```
#### Running Timeslides for Specific Configurations

You can explicitly assign the following parameters `cl_config`, `fm_config`, `ifo_mode`, and `run_name` to run timeslides for a particular configuration.
```
$ snakemake -c4 output/Slurm_Jobs/{cl_config}_{fm_config}_{ifo_mode}/{run_name}/ -F
```
- `cl_config`: Cluster configuration
- `fm_config`: Frame or filter model configuration
- `ifo_mode`: Combination of different interferometers (HL, LV, ...)
- `run_name`: Name of the specific run (test_run, one_year, ...)

# Help 
Use the following commad if VScode did not recognize the jupyter kernel correctly. 
```
poetry run python -m ipykernel install --user --name deploy-py3.9 --display-name "Python (deploy-py3.9)"
```