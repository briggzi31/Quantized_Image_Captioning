#!/bin/sh

# activate conda env
source /gscratch/scrubbed/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/miniconda3/envs/qlora_blip2

echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

# run python script

echo "running python script"
python src/blip2_finetune.py \
    --log-file logs/datasets/log.log \
    --model-id "Salesforce/blip2-opt-2.7b" \
    --cache-dir "/gscratch/scrubbed/briggs3/.cache/"
