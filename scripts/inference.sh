#!/bin/sh

# activate conda env
source /mmfs1/home/briggs3/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2
# source /mmfs1/home/irisz1/miniconda3/bin/activate /gscratch/scrubbed/briggs3/conda_envs/qlora_blip2

echo "current conda environment: "
echo $CONDA_PREFIX
echo ""

split='val'  # 'test', 'val', 'train'
model_type='no-quantization'  # 'pre-trained', 'finetuned'

echo "running python script for inference"
python src/inference.py \
    --log-file logs/inference_${model_type}_${split}/log.log \
    --model-id Salesforce/blip2-opt-2.7b \
    --cache-dir /gscratch/scrubbed/briggs3/.cache/ \
    --data_path /mmfs1/gscratch/scrubbed/briggs3/data/roco/datasets/radiology_data.pkl \
    --checkpoint_dir /gscratch/scrubbed/briggs3/model_checkpoints/radiology \
    --checkpoint_file outputs/checkpoint_${model_type}_${split}.txt \
    --batch_size 10 \
    --split ${split} \
    --output_file outputs/generated_captions_${model_type}_${split}.csv \
    --no_quantization
    # --use_finetuned_model

echo "finished inference"
