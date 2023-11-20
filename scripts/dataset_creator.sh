#!/bin/sh

# activate conda env
source /gscratch/scrubbed/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/miniconda3/envs/qlora_blip2

# make sure conda env was activated 
echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

echo "Running dataset_creator.sh..."

# run python script
python src/dataset_creator.py \
    --log-file logs/datasets/log.log \
    -i /gscratch/scrubbed/briggs3/data/flickr8k/images \
    -c /gscratch/scrubbed/briggs3/data/flickr8k/captions.txt \
    -o /gscratch/scrubbed/briggs3/data/flickr8k/datasets/data.pkl \
    --cache-dir "/gscratch/scrubbed/briggs3/.cache" \
    --overwrite_file

echo "Finished dataset_creator.sh!"