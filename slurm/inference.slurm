#!/bin/bash

## Job Name and User Config
#SBATCH --job-name=val_no_quant_blip2_inference
#SBATCH --mail-type=ALL
##SBATCH --mail-user=irisz1@uw.edu
#SBATCH --mail-user=briggs3@uw.edu


## Parition and Allocation
##SBATCH -p gpu-2080ti  # specify partition
#SBATCH -p ckpt
#SBATCH -A stf  # specify account group

## Resources
#SBATCH --nodes=1
#SBATCH --time=02:00:00  # Max runtime in DD-HH:MM:SS format.
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1

# Directory
#SBATCH --chdir=./
#SBATCH --output=logs/inference_no-quantization_val/log.out # where STDOUT goes
#SBATCH --error=logs/inference_no-quantization_val/log.err # where STDERR goes


## Command(s) to run:
./scripts/inference.sh
