import sys
import os

import argparse
import logging
import pickle

from finetune_model import gpu_config, quantize_model, logging_config
import torch

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor
)

from datasets import DatasetDict, Dataset


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

    for i, x in enumerate(data):
        # print('x for pred_captions:', x)

        image = x["image"]
        inputs = processor(images=image, return_tensors="pt").to(args.device, torch.float16)
        pixel_values = inputs.pixel_values

        # print('x:', x)
        # print('x[image]:', image)
        # print('inputs:', inputs)
        print('x[text]:\t\t', x["text"])

        # BLIP2_peft version
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('peft caption:\t', generated_caption)

        # QLORA version
        # outputs = model.generate(**inputs, max_new_tokens=20)
        # decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print('tim\'s caption:', decoded_outputs)
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
    # parser.add_argument('-cpkt', '--checkpoint_dir', type=str, default="/gscratch/scrubbed/briggs3/checkpoints", required=False)
    parser.add_argument('-d', '--data_path', type=str,
                        default="/gscratch/scrubbed/briggs3/data/flickr8k/datasets/data.pkl", required=False)
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
    print('model:', model)
    print('processor:', processor)

    logging.info(f"Loading data from {args.data_path}...")
    data = load_data(args)
    logging.info("Sucessfully loaded data!")

    test_data = data["test"]

    logging.info("Predicting captions...")
    predict_captions(args, model, processor, test_data)
    logging.info("Sucessfully predicted captions...")


if __name__ == '__main__':
    main()