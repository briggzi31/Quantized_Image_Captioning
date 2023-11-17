#!/bin/sh

# activate conda env
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/qlora_blip2


# run python script
python src/experiment_loading.py
