#!/bin/bash

## Job Name and User Config
#SBATCH --job-name=experiment_loading
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briggs3@uw.edu


## Parition and Allocation
#SBATCH -p gpu-2080ti  # specify partition
#SBATCH -A stf  # specify account group

## Resources
#SBATCH --nodes=1
#SBATCH --time=00:20:00  # Max runtime in DD-HH:MM:SS format.
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1  

# Directory
#SBATCH --chdir=./
#SBATCH --output=./logs/log.out # where STDOUT goes
#SBATCH --error=./logs/log.err # where STDERR goes


## Command(s) to run:
./scripts/experiment_loading.sh
