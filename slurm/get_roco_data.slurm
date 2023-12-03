#!/bin/bash

## Job Name and User Config
#SBATCH --job-name=get_roco_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briggs3@uw.edu


## Parition and Allocation 
##SBATCH -p compute  # specify partition
#SBATCH -p compute-hugemem
#SBATCH -A stf  # specify account group

## Resources
#SBATCH --nodes=10
#SBATCH --time=02:00:00  # Max runtime in DD-HH:MM:SS format.
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=10
#SBATCH --mem=150G


# Directory
#SBATCH --chdir=./
#SBATCH --output=logs/get_dataset/roco_log.out # where STDOUT goes
#SBATCH --error=logs/get_dataset/roco_log.err # where STDERR goes


## Command(s) to run:
./scripts/get_medicat_data/get_roco_data.sh
