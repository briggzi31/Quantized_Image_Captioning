import sys
import os

import argparse
import logging
import pickle
import yaml

import torch

from torch.optim import (
    AdamW,
    Adam
)

from torch.utils.data import (
    DataLoader
)

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

from typing import Union, Any, Optional

from image_dataset import ImageCaptioningDataset


def train_model(
    args: argparse.Namespace,
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    data: ImageCaptioningDataset,
    hyper_params: dict[Any]
) -> None:
    """
    This will train the model.
    This checkpoints the model to the following directory structure:

        args.checkpoint_dir/
            checkpoint_20
            checkpoint_40
            checkpoint_60
            checkpoint_80
            checkpoint_100

    :param args: The arguments passed in from the console
    :param model: The pre-trained model
    :param processor: The pre-trained processor containing a tokenizer and vision encoder
    :param data: The data in train, val, test split
    """
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    train_dataloader = DataLoader(data, shuffle=True, batch_size=hyper_params['batch_size'], collate_fn=data.collate_fn)
    optimizer = AdamW(model.parameters(), lr=hyper_params['alpha'])

    if args.resume_training:
        
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
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        tot_steps = checkpoint['tot_steps']
        loss = checkpoint['loss']
        logging.info(f"Current epoch: {start_epoch}")
        logging.info(f"Total steps: {tot_steps}")
        logging.info(f"Current Loss: {loss}")

    else:
        start_epoch = 0
        tot_steps = 0
        loss = None

    model.train()

    for i in range(start_epoch, hyper_params['num_epochs']):
        epoch = i + 1

        print(f"Epoch: {epoch}/{hyper_params['num_epochs']}")
        for idx, batch in enumerate(train_dataloader):

            # bos_token = batch.pop('bos').to(args.device)
            captions = batch.pop('labels').to(args.device)
            pixel_values = batch.pop('pixel_values').to(args.device, torch.float16)

            # why do we pass captions to both input_ids and labels
            logging.debug("Performing forward pass...")
            outputs = model(
                input_ids=captions,
                pixel_values=pixel_values,
                labels=captions
            )
            logging.debug("Forward pass done!")

            loss = outputs.loss

            logging.debug("Performing backward pass...")
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            logging.debug("Backward pass done!")

            tot_steps += 1

            print("Loss:", loss.item())
            logging.info(f"Epoch: {i + 1}, idx: {idx}, tot_steps: {tot_steps}")
            logging.info(f"Loss: {loss.item()}")

            # checkpoint the model every args.save_steps
            if tot_steps % hyper_params['save_steps'] == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, "model_" + str(tot_steps) + ".tar")
                logging.info(f"Step: {tot_steps} | Saving model to {checkpoint_path}...")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'tot_steps': tot_steps
                }, checkpoint_path)
                args.current_checkpoints.append(tot_steps)

                # remove previous model checkpoints and keep only the 10 most recent
                oldest_checkpoint_idx = 0
                while len(os.listdir(args.checkpoint_dir)) > 10:
                    oldest_checkpoint_num = args.current_checkpoints.pop(oldest_checkpoint_idx)
                    oldest_checkpoint = os.path.join(args.checkpoint_dir, "model_" + str(oldest_checkpoint_num) + ".tar")
                    logging.info(f"Removing oldest checkpoint: {oldest_checkpoint}")
                    os.remove(oldest_checkpoint)
                    oldest_checkpoint_idx += 1

                logging.info(f"Model Saved!")


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
    
    device_properties = torch.cuda.get_device_properties('cuda:0')
    logging.info(f"Cuda GPU properties: {device_properties}")

    max_gpu_model_mem = device_properties


    model_double_quant = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=nf4_config,
        cache_dir=args.cache_dir,
        device_map='auto',
        max_memory={0: "16GIB"}
    )

    model_double_quant.gradient_checkpointing_enable()
    model_double_quant = prepare_model_for_kbit_training(model_double_quant)

    qlora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],   # only change last language model queries and keys
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model_double_quant, qlora_config)

    model.to(args.device)

    return model, processor


def get_trainable_parameters(model: PeftModel) -> str:
    """
    Returns the trainable paramaters vs. all parameters in the model as a string
    """
    trainable_params, all_param = model.get_nb_trainable_parameters()

    return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"


def resume_training(args: argparse.Namespace) -> tuple[bool, Optional[str]]:
    """
    Returns whether to resume training, as well as the last checkpoint_dir if
        resume_training is True
    :param args: The command line argument kwargs
    :return: whether we should resume training, and if so the the current checkpoint directory to load the model from
    """
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    current_checkpoint_files = os.listdir(args.checkpoint_dir)

    if len(current_checkpoint_files) < 1:
        resume_training = False
        current_checkpoint = None
        sorted_step_num = []
    else:
        resume_training = True
        sorted_step_num = list(sorted(map(lambda x: int(x.split("_")[1].split(".")[0]), current_checkpoint_files)))

    return resume_training, sorted_step_num


def load_data(args: argparse.Namespace, processor: Blip2Processor) -> dict[str, ImageCaptioningDataset]:
    """
    This loads in the specified data from a pickle file

    :param args: The command line argument kwargs
    :return: The loaded data
    """
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    train_data = ImageCaptioningDataset(data['train'], processor)
    val_data = ImageCaptioningDataset(data['val'], processor)
    test_data = ImageCaptioningDataset(data['test'], processor)

    return {'train': train_data, 'val': val_data, 'test': test_data}


def gpu_config(args: argparse.Namespace) -> None:
    """
    This configures the gpu usage
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # to handle memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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
        filemode='a',
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
    parser.add_argument('-hp', '--hyper_param_config', type=str, default="hyper_param_config/finetuning_config.yaml", required=True)

    return parser.parse_args()


def main():
    args = get_args()
    logging_config(args)
    args.device = gpu_config(args)

    logging.info(f"Current Configuration args: {args}")

    logging.info("Quantizing Pre-trained model...")
    model, processor = quantize_model(args)
    logging.info("Successfully quantized Pre-trained model!")

    print("peft model", model)
    print("peft processor", processor)
    print(f"{get_trainable_parameters(model)}")

    logging.info(f"{get_trainable_parameters(model)}")

    logging.info(f"Loading data from {args.data_path}...")
    data = load_data(args, processor)
    logging.info("Sucessfully loaded data!")

    train_data = data['train']

    args.resume_training, args.current_checkpoints = resume_training(args)

    with open(args.hyper_param_config) as f:
            hyper_params = yaml.safe_load(f)

    train_model(args, model, processor, train_data, hyper_params)

    logging.info("Finished!!")


if __name__ == '__main__':
    main()
