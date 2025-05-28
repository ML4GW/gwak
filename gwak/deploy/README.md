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
$ snakemake -c1 deploy -F
```
 