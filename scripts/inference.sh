#!/bin/sh

# activate conda env
# source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2
source /mmfs1/home/irisz1/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2

echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

# run python script

echo "running python script"
python src/inference.py \
    --log-file logs/inference/log.log \
    --model-id Salesforce/blip2-opt-2.7b \
    --cache-dir /gscratch/scrubbed/irisz1/.cache/ \
    --data_path /mmfs1/gscratch/scrubbed/briggs3/data/roco/datasets/radiology_data.pkl \
    --checkpoint_dir /gscratch/scrubbed/briggs3/model_checkpoints/radiology \
    --batch_size 10 \
    --split "test" \
    --output_file /outputs/generated_captions.csv
    # --hyper_param_config hyper_param_config/finetuning_config.yaml
