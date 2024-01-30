#!/bin/sh

# activate conda env
# source /mmfs1/home/irisz1/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2


# make sure conda env was activated
echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

echo "Running cider.sh..."

python src/cider.py \
    -i outputs/generated_captions/generated_captions_no-quantization_val.csv \
    -o outputs/scores/no-quantization_val_scores.txt
