import sys
import os

import argparse
import logging
import pickle

import torch
from peft import (
    PeftModel, 
    LoraConfig,
    get_peft_model, 
    prepare_model_for_kbit_training
)

from transformers import (
    BitsAndBytesConfig, 
    Blip2ForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    Blip2Processor,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from transformers.trainer_utils import get_last_checkpoint

from datasets import DatasetDict, Dataset

from typing import Union
    

def train_model(
    args: argparse.Namespace, 
    model: Blip2ForConditionalGeneration, 
    processor: Blip2Processor, 
    data: DatasetDict
) -> None:
    """
    This will train the model

    :param args: The arguments passed in from the console
    :param model: The pre-trained model 
    :param processor: The pre-trained processor containing a tokenizer and vision encoder
    :param data: The data in train, val, test split
    """
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
        logging_dir="logs/finetune",
        logging_strategy="epoch",
        output_dir=args.checkpoint_dir,
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=10
    )

    logging.info("Configuring trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(processor.tokenizer, mlm=False),

    ### We need data['train']['labels'] = decoder_input_ids
    ### We need data['train']['pixel_values'] = processor(images=item["image"], padding="max_length", return_tensors="pt")

    )
    logging.info("Successfully configured Trainer!")

    if args.resume_training:
        logging.info(f"Resumeing training from {args.current_checkpoint_dir}")
        trainer.train(
            resume_from_checkpoint=args.current_checkpoint_dir
        )
    else:
        logging.info("Starting training...")
        trainer.train()
    logging.info("Training finished")


def quantize_model(
    args: argparse.Namespace
) -> tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    """
    This loads in the pretrained weights and double quantizes the weights for the 
        model specified in args.

    :param args: The arguments passed in from the console
    :return: The pre-trained double quanitized model, and the corresponding pre-trained tokenizer
    """
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir
    )

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

    model_double_quant.gradient_checkpointing_enable()
    model_double_quant = prepare_model_for_kbit_training(model_double_quant)

    print("model_double_quant\n", model_double_quant)

    qlora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj"],   # only change last language model queries and keys
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model_double_quant, qlora_config)

    model.to(args.device)

    logging.info(f"{get_trainable_parameters(model)}")
 
    return model, processor


def get_trainable_parameters(model: PeftModel) -> str:
    """
    Returns the trainable paramaters vs. all parameters in the model as a string
    """
    trainable_params, all_param = model.get_nb_trainable_parameters()

    return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"


def load_data(args: argparse.Namespace, processor: Blip2Processor) -> DatasetDict:
    """
    This loads in the specified data from a pickle file
    
    :param args: The command line argument kwargs
    :return: The loaded data
    """
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    
    print("pre-loaded data", data)

    data = data.map(lambda samples: processor.tokenizer(samples["text"]), batched=True)

    return data


def gpu_config(args: argparse.Namespace) -> None:
    """
    This configures the gpu usage
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Device Type: {device}")
    return device


def logging_config(args: argparse.Namespace) -> None:
    """
    This sets up the logger for log files
    
    :param args: The command line argument kwargs
    :return: None
    """
    head, tail = os.path.split(args.log_file)

    if not os.path.isdir(head):
        os.makedirs(head)

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
    parser.add_argument('-cpkt', '--checkpoint_dir', type=str, default="/gscratch/scrubbed/briggs3/checkpoints", required=False)
    parser.add_argument('-d', '--data_path', type=str, 
                        default="/gscratch/scrubbed/briggs3/data/flickr8k/datasets/data.pkl", required=False)

    return parser.parse_args()


def main():
    args = get_args()
    logging_config(args)
    args.device = gpu_config(args)

    logging.info(f"Current Configuration args: {args}")

    logging.info("Quantizing Pre-trained model...")
    model, processor = quantize_model(args)
    logging.info("Successfully quantized Pre-trained model!")

    logging.info(f"Loading data from {args.data_path}...")
    data = load_data(args, processor)
    logging.info("Sucessfully loaded data!")

    print("data", data)
    train_data = data['train']
    print("train_data", train_data)
    print("train_data[0]", train_data[0])

    args.current_checkpoint_dir = get_last_checkpoint(args.checkpoint_dir)

    args.resume_training = True
    if args.current_checkpoint_dir is None:
        args.resume_training = False

    train_model(args, model, processor, data)

    print("processor", processor)
    print("model", model)

    logging.info("Finished!!")


if __name__ == '__main__':
    main()