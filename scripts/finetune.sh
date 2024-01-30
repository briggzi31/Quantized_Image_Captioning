#!/bin/sh

# activate conda env
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2
# source /mmfs1/home/irisz1/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2

echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

# run python script

# finetune the model for radiology data
echo "running python script"
python src/finetune_model.py \
    --log-file logs/finetune/log.log \
    --model-id Salesforce/blip2-opt-2.7b \
    --cache-dir /gscratch/scrubbed/briggs3/.cache/ \
    --data_path /mmfs1/gscratch/scrubbed/briggs3/data/roco/datasets/radiology_data.pkl \
    --checkpoint_dir /gscratch/scrubbed/briggs3/model_checkpoints/radiology \
    --hyper_param_config hyper_param_config/finetuning_config.yaml


# finetune the model for non-radiology data
# echo "running python script"
# python src/finetune_model.py \
#     --log-file logs/finetune2/log.log \
#     --model-id Salesforce/blip2-opt-2.7b \
#     --cache-dir /gscratch/scrubbed/briggs3/.cache/ \
#     --data_path /mmfs1/gscratch/scrubbed/briggs3/data/roco/datasets/non-radiology_data.pkl \
#     --checkpoint_dir /gscratch/scrubbed/briggs3/model_checkpoints/non-radiology \
#     --hyper_param_config hyper_param_config/finetuning_config.yaml

