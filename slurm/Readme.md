

To submit a batch job:

First make sure that you are in the /mmfs1/home/briggs3/Quantized_Image_Captioning/ directory

```
sbatch slurm/finetune_gpu.slurm
```

This submits the finetune_gpu.slurm script which calls scripts/finetune.sh which calls finetune_model.py
The slurm script can be changed to change the partition ```#SBATCH -p [PARTITION]```. Possible partitions
are ```gpu-2080ti``` and ```ckpt```.

To view the queue:
```
squeue -u briggs3
```

Logs go to 
```
logs/finetune
```


The python scripts is based off of [colab](https://colab.research.google.com/drive/16XbIysCzgpAld7Kd9-xz-23VPWmqdWmW?usp=sharing#scrollTo=6cCVhsmJxxjH). I am confused why they pass input_ids=input_ids in this method when training the model:

```15 outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
```

I am having similar issues as this [issue](https://github.com/huggingface/peft/issues/376)

