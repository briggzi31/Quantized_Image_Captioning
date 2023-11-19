#!/bin/bash

## Job Name and User Config
#SBATCH --job-name=blip2_finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briggs3@uw.edu


## Parition and Allocation 
#SBATCH -p compute  # specify partition
#SBATCH -A stf  # specify account group

## Resources
#SBATCH --nodes=1
#SBATCH --time=00:20:00  # Max runtime in DD-HH:MM:SS format.
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32G


# Directory
#SBATCH --chdir=./
#SBATCH --output=./outputs/log.out # where STDOUT goes
#SBATCH --error=./outputs/log.err # where STDERR goes


## Command(s) to run:
./scripts/blip2_finetune.sh