# Quantized_Image_Captioning
This holds our work for the final project in CSE 493G. Our written paper can be found [here](https://github.com/briggzi31/Quantized_Image_Captioning/blob/main/Quantizing_Image_Captioning.pdf).

## Conda Environment Versioning

### Setup

To create the conda environment:

```commandline
conda env create -f environment.yml
```

### Updating the Conda environment
If you pull code from the repo and the environment.yml file has changed, 
update your environment by running the
following (after activating the environment):

```commandline
conda env update -f environment.yml --prune
```

### Installing new packages
If you need any new packages, install them with either

1. conda install PACKAGE_NAME. 
2. pip install PACKAGE_NAME.

Then, before committing, run:

```commandline
conda env export --no-builds > environment.yml
```

Then delete the prefix: line (last line) in the environment.yml file.
