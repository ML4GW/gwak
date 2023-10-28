# Introduction
This repo contains a complimentary code for the GWAK LIGO paper.

<img width="593" alt="3d_gwak" src="https://github.com/ML4GW/gwak/assets/4249113/8b00eced-9bad-4bee-b5cb-dad50e51d422">



# Set up
To start, install [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html)
(if you don't already have it) to be able to create a virtual environment.

Once conda is there, setup a virtual environment using the file from the repo
```
conda env create --file environment.yaml
conda activate gwak
```

Install additional package inside the create virtual environment
```
pip install snakemake
pip install matplotlib-label-lines
```

To make pytorch available, do 
```
pip install torch==2.0.1
```

Good thing to do before running (only has to be done once) is to create
an `output/` directory that is a symbolic link to your work space (
wherever you have a lot of available space), to be able to store all the data.
```
ln -s {your-work-path-to-output} output/
```
Otherwise `Snakemake` will create `output/` directory in the same location this repo is.

Now every time you want to run the code, first activate the virtual environment
```
conda activate gwak
```


# Snakemake
The code is organized using [`Snakemake`](https://snakemake.readthedocs.io/en/stable/).
The Snakemake workflow management system is a tool to create reproducible and scalable data analyses.

To run snakemake do
```
snakemake -c1 {rule_name}
```
where `-c1` specifies number of cores provided (one in this case).
It became required to specify it in the latest versions of snakemake,
so to make life easier you can add
`alias snakemake="snakemake -c1"` to your `bash/zsch/whatever` profile
and afterwards simply run `snakemake {rule_name}`.

If you want to run a rule, but Snakemake tells you `Nothing to be done`, use `-f`
to force it. Use `-F` to also force all the upstream rules to be re-run.

# Running with Condor

In order to be able to submit jobs to `HTCondor`, install [snakemake-condor-profile](https://github.com/msto/snakemake-condor-profile).

Sending Snakemake process to `HTCondor`:
 
    $ snakemake --profile HTCondor

# Analysis Pipeline
Each step of the analysis is represented as a corresponding rule in the `Snakefile`.
There are several steps starting with downloading and generating data samples,
pre-processing them and training the GWAK space using these samples.
Followed by the Pearson correlation calculation and the evaluation step.
The details of all the available rules are given below.
## Configuration parameters
All the parameters of generation and training are specified in `config.py`, that's
the only place they are specified and the only place through which they can be changed.

## Running Omicron
The analysis pipeline starts with fetching valid 03a background data and running Omicron software on it to select loud and quite segments of data taking.
To find the valid data segments, we have in the repo under the `data/` folder valid segments for H1 and L1 downloaded from the LIGO website.
```
snakemake -c1 run_omicron
```

## Downloading LIGO data
Next, the LIGO data is fetched and saved with
```
snakemake -c1 fetch_data
```

## Data Generation and pre-processing
To generate all the necessary samples for the analysis from downloaded data use `generate_dataset` rule.
These rules are written using `Wildcards`, such that one rule can be used for
each of four datasets, eg BBH, SG, glitches and background.

## Plotting
Finally, all the plots from the paper can be reproduced with the plotting rule
```
snakemake -c1 plot_results
```
