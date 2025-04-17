# GWAK 2.0 🥑🦾

This repo is dedicated to the updated version of the algorithm presented in the [MLST](https://iopscience.iop.org/article/10.1088/2632-2153/ad3a31). 

![DALL·E 2024-06-07 15 21 58 - A futuristic and artistic version of an avocado  The avocado is designed with sleek, metallic textures and glowing neon pink and yellow accents  The s](https://github.com/ML4GW/gwak2/assets/4249113/f396688b-125e-48f1-bbd5-48f3c9854e8e&key_width=80)

The current projects include
- [`data`](./gwak/data/README.md) - Scripts for generating training and testing data
- [`train`](./gwak/train/README.md) - Pytorch (lightning) code for training neural-networks
- [`deploy`](./gwak/deploy/README.md) - Triton wapper (hermes) code to deploy trained neural-networks

The repo includes implementation of both `gwak1` and `gwak2`, where the configs for `gwak1` live in the corresponding folder and in the `Snakefile` the corresponding rules have `_gwak1`.

The project uses `poetry`, `Conda` and `Snakemake` to run the code. Follow installation instructions below to prepare your environment.


## Installation ⚙️


### Optional step (only if you don't have `Miniconda3`)
If you **do not have** `Miniconda` installed on your machine, follow first those steps
- use the [`quickstart`](https://github.com/ML4GW/quickstart) repo to setup `Miniconda` and install `poetry`
```
$ git clone git@github.com:ml4gw/quickstart.git
$ cd quickstart
$ make
```

If you see this error, it is already known in [issue#7](https://github.com/ML4GW/quickstart/issues/7)
```
Verifying checksum... Done.
Preparing to install helm into /you/path/miniconda3-tmp/bin/
helm installed into /you/path/miniconda3-tmp/bin//helm
helm not found. Is /you/path/miniconda3-tmp/bin/ on your $PATH?
Failed to install helm
    For support, go to https://github.com/helm/helm.
make: *** [Makefile:65: install-helm] Error 1
```
do the following commands:
```
$ source ~/.bashrc
$ make install-poetry install-kubectl install-s3cmd
```

If everything was installed successfully, continue to the steps below.


### Main installation

If you **do have** `Miniconda` already installed on your machine, follow those steps
- checkout this repo and clone submodules (such as `ml4gw`)
```
$ git clone git@github.com:ML4GW/gwak.git
$ cd gwak
$ git submodule update --init --recursive
```
- create a new `Conda` environment
```
$ conda env create -n gwak --file environment.yaml
$ conda activate gwak
```
- install `gwak` project in the editing mode
```
$ pip install -e .
```

Now you are ready to *gwak*!
As a first step, you can run the training by doing
```
$ cd gwak
$ snakemake -c1 train_all
```

- if you want to modify any of the submodules, first do the changes localy and then re-install `gwak` to pick up the changes:
```
$ pip install -e .
```

- Installation for deployment

Run the following command at base enviroment. 
```
(base)$ pip install snakemake==7.32.4 pulp==2.6.0
(base)$ cd gwak/gwak/deploy
(base)$ poetry install
```
Use ```poetry add <package>``` to add new packages to the deploy application. 
