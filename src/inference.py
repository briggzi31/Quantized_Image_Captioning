import sys
import os

import argparse
import logging
import pickle
import json
import pandas as pd

from finetune_model import gpu_config, quantize_model, logging_config, resume_training
import torch

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor
)

from datasets import DatasetDict, Dataset


def load_unquantized_unfinetuned_model(
    args: argparse.Namespace,
) -> tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    """
    This loads in the specified unquantized and unfinetuned base model from HuggingFace

    :param args: The arguments passed in from the console
    :return: The pre-trained double quanitized model, and the corresponding pre-trained tokenizer
    """
    processor = Blip2Processor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        device_map={"": 0},
        torch_dtype=torch.float16
    )

    return model, processor


def load_finetuned_model(
    args: argparse.Namespace,
    model: Blip2ForConditionalGeneration,
) -> Blip2ForConditionalGeneration:
    """
    Loads in finetuned model from a checkpoint directory

    :param args: The arguments passed in from the console
    :param model: The pre-trained model

    :return: Returns the finetuned model
    """
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

    while len(args.current_checkpoints) > 0:
        current_checkpoint = os.path.join(args.checkpoint_dir, "model_" + str(args.current_checkpoints.pop()) + ".tar")
        logging.info(f"Trying to load in model from {current_checkpoint}")

        try:
            checkpoint = torch.load(current_checkpoint)
            logging.info(f"Successfully loaded model in from {current_checkpoint}")
            break
        except RuntimeError:
            logging.warning(f"Error: Failed to load checkpoint {current_checkpoint}")

    model.load_state_dict(checkpoint['model_state_dict'])

    loss = checkpoint['loss']
    logging.info(f"Current Loss: {loss}")

    model.eval()

    return model


def load_data(args: argparse.Namespace) -> DatasetDict:
    """
    This loads in the specified data from a pickle file

    :param args: The command line argument kwargs
    :return: The loaded data
    """
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    return data


def predict_captions(
    args: argparse.Namespace,
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    data: Dataset
    ) -> None:
    """
    This will perform inference on the given model

    :param args: The arguments passed in from the console
    :param model: The pre-trained model
    :param processor: The pre-trained processor containing a tokenizer and vision encoder
    :param data: The data to be inferred on
    """
    # decides whether we continue inference where we left off, or start anew
    if os.path.exists(args.checkpoint_file):
        with open(args.checkpoint_file, 'r') as f:
            checkpoints = f.readlines()
        start = max(list(map(lambda x: int(x), checkpoints)))
    else:
        start = 0
    logging.info(f"starting batch num: {start}")

    for i in range(start, len(data), args.batch_size):
        print(f"cur iter {i}/{len(data)}")
        logging.info(f"starting batch {i}/{((len(data) + 1) // args.batch_size)}")

        captions_df = pd.DataFrame()

        stop = i + args.batch_size
        if i + args.batch_size > len(data):
            stop = len(data)

        batch = data[i:stop]

        batch_roco_ids = batch["ROCO_ID"]
        batch_images = batch["image"]
        target_captions = batch["text"]

        inputs = processor(images=batch_images, return_tensors="pt").to(args.device)
        pixel_values = inputs.pixel_values
        print('batch_image:', batch_images[0])
        print('tar_caption:', target_captions[0])

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print('gen_caption:\t', generated_captions[0])

        for j in range(len(generated_captions)):
            roco_id = batch_roco_ids[j].strip()
            gen_caption = generated_captions[j].strip()
            tar_caption = target_captions[j].strip()

            out_row = {
                'ROCO_ID': [roco_id],
                'generated_caption': [gen_caption],
                'target_caption': [tar_caption]
            }

            cur_caption_df = pd.DataFrame(out_row)
            print(cur_caption_df)
            captions_df = pd.concat([captions_df, cur_caption_df], axis=0, ignore_index=True)
            print(captions_df[-5:])

        captions_df.to_csv(args.output_file, mode='a', index=False, header=False)

        with open(args.checkpoint_file, 'a') as out:
            logging.info(f"writing batch_num {i} to {args.checkpoint_file}")
            out.write(str(i) + "\n")


def get_args() -> argparse.Namespace:
    """
    This reads all command line arguments

    :return: The command line argument kwargs
    """
    parser = argparse.ArgumentParser(
        prog="Blip2_Inference",
        description="This will perform inference on MedICaT using a quantized model",
    )

    parser.add_argument('-l', '--log-file', type=str, default="logs/inference/log.log", required=False)
    parser.add_argument('-m', '--model-id', type=str, default="Salesforce/blip2-opt-2.7b", required=False)
    parser.add_argument('-c', '--cache-dir', type=str, default="/gscratch/scrubbed/briggs3/.cache/", required=False)
    parser.add_argument('-cpkt', '--checkpoint_dir', type=str, default="/gscratch/scrubbed/briggs3/checkpoints", required=False)
    parser.add_argument('-d', '--data_path', type=str,
                        default="/gscratch/scrubbed/briggs3/data/flickr8k/datasets/data.pkl", required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=10, required=False)
    parser.add_argument('-s', '--split', type=str, default="test", required=False)  # 'train', 'val', 'test'
    parser.add_argument('-o', '--output_file', type=str, default="/outputs/generated_captions.csv", required=False)
    parser.add_argument('-cf', '--checkpoint_file', type=str, default='outputs/checkpoint.txt', required=False)
    parser.add_argument('-uf', '--use_finetuned_model', action='store_true', required=False)
    parser.add_argument('-nq', '--no_quantization', action='store_true', required=False)

    return parser.parse_args()


def main():
    # configure module
    args = get_args()
    logging_config(args)
    args.device = gpu_config(args)

    logging.info(f"Current Configuration args: {args}")

    # load in quantized pre-trained model
    if args.no_quantization:
        logging.info("Loading unquantized, unfinetuned model...")
        model, processor = load_unquantized_unfinetuned_model(args)
        logging.info("Successfuly loaded unquantized, unfinetuned model!")
    else:   
        logging.info("Quantizing Pre-trained model...")
        model, processor = quantize_model(args)
        logging.info("Successfully quantized Pre-trained model!")
    print('model:', model)
    print('processor:', processor)

    if args.use_finetuned_model:
    # load in local finetuned model
        logging.info(f"Trying to load in fine-tuned model for inference...")
        _ , args.current_checkpoints = resume_training(args)
        model = load_finetuned_model(args, model)
        logging.info("Successfully loaded finetuned model!")
        print('model fine:', model)
        print('processor fine:', processor)
    else:
        logging.info("skipping loading in fine-tuned model. Using pre-trained model only!")

    # load teh data
    logging.info(f"Loading data from {args.data_path}...")
    data = load_data(args)
    data = data[args.split]
    logging.info(f"data length: {len(data)}")
    logging.info("Sucessfully loaded data!")

    # generate captions
    logging.info("Predicting captions...")
    predict_captions(args, model, processor, data)
    logging.info("Sucessfully predicted captions...")


if __name__ == '__main__':
    main()