#!/bin/sh

# activate conda env
# source /mmfs1/home/irisz1/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2


# make sure conda env was activated
echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

echo "Running dataset_creator.sh..."

# run python script
python src/dataset_creator.py \
    --log-file logs/datasets/log.log \
    -i /gscratch/scrubbed/briggs3/data/roco \
    -c /gscratch/scrubbed/briggs3/data/roco/test/radiology/captions.txt \
    -o /gscratch/scrubbed/briggs3/data/roco/datasets \
    --cache-dir "/gscratch/scrubbed/briggs3/.cache" \
    --overwrite_file

# python src/dataset_creator.py \
#     --log-file logs/datasets/log.log \
#     -i /gscratch/scrubbed/briggs3/data/roco \
#     -c /gscratch/scrubbed/briggs3/data/roco/test/radiology/captions.txt \
#     -o /gscratch/scrubbed/irisz1/data/roco/datasets \
#     --cache-dir "/gscratch/scrubbed/irisz1/.cache" \
#     --overwrite_file


echo "Finished dataset_creator.sh!"
