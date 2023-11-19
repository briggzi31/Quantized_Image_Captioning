import sys
import os

import argparse
import logging

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM


def quantize_blip2(args):

# configuing 4bit quantization
logging.info("Configuring 4bit double quantization")
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)



def logging_config(args: argparse.Namespace) -> None:
    """
    This sets up the logger for log files
    
    :param args: The command line argument kwargs
    :return: None
    """
    fmt = '%(asctime)s | %(levelname)s | "%(filename)s::%(lineno)d | %(message)s'
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
    
    parser.add_argument('--log-file', type=str, default="outputs/log.log", required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging_config(args)

    quantize_

    

    logging.info("Finished!!")
