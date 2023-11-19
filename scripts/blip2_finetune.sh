#!/bin/sh

# activate conda env
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/qlora_blip2


# run python script

echo "running python script"
python src/blip2_finetune.py \
    --log-file outputs/log.log
