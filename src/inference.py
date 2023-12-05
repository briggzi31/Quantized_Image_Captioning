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
    for i in range(0, len(data), args.batch_size):
        captions_df = pd.DataFrame()

        stop = i + args.batch_size
        if i + args.batch_size > len(data):
            stop = len(data)

        batch = data[i:stop]
        batch_roco_ids = batch["ROCO_ID"]
        batch_file_names = batch["file_name"]
        batch_images = batch["images"]
        target_captions = batch["text"]

        inputs = processor(images=batch_images, return_tensors="pt").to(args.device)
        pixel_values = inputs.pixel_values
        print('batch_image:', batch_image[0])
        print('tar_caption:', target_captions[0])

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print('gen_caption:\t', generated_captions[0])

        for j in len(generated_captions):
            roco_id = batch_roco_ids[j].strip()
            image_file_name = batch_file_names[j].strip()
            gen_caption = generated_captions[j].strip()
            tar_caption = target_captions[j].strip()

            out_row = {
                'ROCO_ID': roco_id,
                'file_name': image_file_name,
                'generated_caption': gen_caption,
                'target_caption': tar_caption
            }

            captions_df.append(out_row, ignore_index=True)

        captions_df.to_csv(args.output_file, mode='a', index=False, header=False)

        break



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
    parser.add_argument('-s', '--split', type=str, default="test", required=False)
    # parser.add_argument('-hp', '--hyper_param_config', type=str, default="hyper_param_config/finetuning_config.yaml", required=True)
    # parser.add_argument('-e', '--num_epochs', type=int, default=200, required=False)

    return parser.parse_args()


def main():
    args = get_args()
    logging_config(args)
    args.device = gpu_config(args)

    logging.info(f"Current Configuration args: {args}")

    logging.info("Quantizing Pre-trained model...")
    model, processor = quantize_model(args)
    logging.info("Successfully quantized Pre-trained model!")
    print('model quant:', model)
    print('processor quant:', processor)

    logging.info(f"Trying to load in model from for inference")
    _ , args.current_checkpoints = resume_training(args)
    model = load_finetuned_model(args, model)
    logging.info("Successfully loaded finetuned model!")
    print('model fine:', model)
    print('processor fine:', processor)

    logging.info(f"Loading data from {args.data_path}...")
    data = load_data(args)
    data = data[args.split]
    logging.info("Sucessfully loaded data!")

    logging.info("Predicting captions...")
    predict_captions(args, model, processor, data)
    logging.info("Sucessfully predicted captions...")


if __name__ == '__main__':
    main()