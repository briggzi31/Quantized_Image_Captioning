import sys
import os

import argparse
import logging

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, AutoTokenizer, AutoProcessor


def quantize_model(
    args: argparse.Namespace
) -> tuple[Blip2ForConditionalGeneration, AutoTokenizer]:
    """
    This loads in the pretrained weights and double quantizes the weights for the 
        model specified in args.

    :param args: The arguments passed in from the console
    :return: The pre-trained double quanitized model, and the corresponding pre-trained tokenizer
    """
    # configuing 4bit quantization
    logging.info("Configuring 4bit double quantization")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_double_quant = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, 
        quantization_config=nf4_config, 
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir
    )

    return model_double_quant, tokenizer



def logging_config(args: argparse.Namespace) -> None:
    """
    This sets up the logger for log files
    
    :param args: The command line argument kwargs
    :return: None
    """
    fmt = '%(asctime)s | %(levelname)s | "%(filename)s::line-%(lineno)d | %(message)s'
    logging.basicConfig(
        filename=args.log_file, 
        filemode='w',
        level=logging.DEBUG,
        format=fmt,
    )


def get_args() -> argparse.Namespace:
    """
    This reads all command line arguments

    :return: The command line argument kwargs
    """
    parser = argparse.ArgumentParser(
        prog="Blip2_FineTuning",
        description="This will fine tune Quantized (on QLora) Blip2 on MedICaT",
    )
    
    parser.add_argument('-l', '--log-file', type=str, default="logs/finetune/log.log", required=False)
    parser.add_argument('-m', '--model-id', type=str, default="Salesforce/blip2-opt-2.7b", required=False)
    parser.add_argument('-c', '--cache-dir', type=str, default="/gscratch/scrubbed/briggs3/.cache/", required=False)

    return parser.parse_args()


def main():
    args = get_args()
    logging_config(args)
    logging.info(f"Current Configuration args: {args}")

    logging.info("Quantizing Pre-trained model...")
    model, tokenizer = quantize_model(args)

    print("tokenizer", tokenizer)
    print("model", model)

    logging.info("Finished!!")


if __name__ == '__main__':
    main()
