# Deploy your trained NN model #
The deploy application provides two major utilities:
- Exporting trained NN model to excutibles
- Producing inference results on sequential data

# Use the following command to deploy your NN model #
### Via ```Snakemake``` ###

- Export
```
$ cd gwak/gwak
$ snakemake -c1 export_all 
```
- Inference
```
$ cd gwak/gwak
$ snakemake -c1 infer_all 
```

### Via ```Poetry``` ###

- Export white_noise_burst
```
$ cd gwak/gwak/deploy
$ poetry run python deploy/cli_export.py --config deploy/config/export.yaml --project white_noise_burst
``` 

- Infernce on white_noise_burst

```
$ cd gwak/gwak/deploy
$ poetry run python deploy/cli_infer.py --config deploy/config/infer.yaml --project white_noise_burst
``` 

