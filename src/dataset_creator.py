"""
dataset_creator

This module will load in a directory full of images and corresponding captions into a HuggingFace
    image datasets.DatasetDict object
"""

import argparse
import logging
import os
import json
import pickle

import pandas as pd

from datasets import (
    load_dataset, 
    Image, 
    Dataset, 
    DatasetDict,
)


def save_dataset(args: argparse.Namespace, data: DatasetDict) -> None:
    """
    This pickles the dataset and saves it to the specified path in args for downstream use

    :param args: Arguments passed in from the console. 
    :return: None
    """

    output_dir, _ = os.path.split(args.output_file) 
    if not os.path.isdir(output_dir):
        logging.info(f"Making directory {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Pickling to {args.output_file}")
    with open(args.output_file, 'wb') as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


def perform_train_val_test_split(
    args: argparse.Namespace,
    data: Dataset
) -> DatasetDict:
    """
    This performs train test split on the dataset, putting it into a datasets.DatasetDict object.
    This splits based on the specified train-test-split in args, and then further splits the test
    set into val and test sets.

    :param args: Arguments passed in from the console. 
    :data data: The data which train-val-test-split is to be performed on
    :return: The partitioned data as a datasets.Datadict object with train, val, and test splits
    """

    logging.info("Performing train-test-val split!")
    # split into train and test/val
    train_testval = data.train_test_split(test_size=args.train_test_split)
    # split test and val in half
    test_val = train_testval['test'].train_test_split(test_size=0.5)

    ds = DatasetDict(
        {
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test'],
        }
    )

    return ds


def read_in_data(args: argparse.Namespace) -> Dataset:
    """
    This loads in a dataset from the specified directory, and returns a 
        HuggingFace datasets object.

    :param args: Arguments passed in from the console. 
    :return: The loaded in HuggingFace Dataset object
    """
    meta_data_path = os.path.join(args.image_input_dir, "metadata.jsonl")

    logging.info("Loading data into HuggingFace Datasets object!")
    image_data = load_dataset(
        "imagefolder", 
        data_dir=args.image_input_dir, 
        split='train'
    )

    return image_data


def zip_captions(args) -> None:
    """
    This creates a metadata.jsonl file and saves the metadata.jsonl file within args.image_input_dir/ '
        for downstream dataset loading into a HuggingFace Dataset object

    :param args: Arguments passed in from the console. 
    :return: None
    """

    captions = pd.read_csv(args.caption_input_file)
    captions.rename(columns={"image": "file_name", "caption": "text"}, inplace=True)
    
    # create jsone
    json_captions = captions.to_json(orient='records')
    json_captions = json.loads(json_captions)

    # save metadata.jsonl to images folder
    meta_data_path = os.path.join(args.image_input_dir, "metadata.jsonl")
    logging.info(f"Writing to metadata.jsonl file to {meta_data_path}!")
    with open(meta_data_path, 'w') as out:
        for image_caption in json_captions:
            cur_image_caption = json.dumps(image_caption)
            out.write(str(cur_image_caption) + "\n")


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
    
    parser.add_argument('-l', '--log-file', type=str, default="logs/datasets/log.log", required=False,
        help='Path to output the log file.')
    parser.add_argument('-i', '--image_input_dir', type=str, 
        default="/gscratch/scrubbed/briggs3/data/flickr8k/images", required=False,
        help='Directory where all images are kept.')
    parser.add_argument('-c', '--caption_input_file', type=str, default="/gscratch/scrubbed/briggs3/data/flickr8k/captions.txt",
        help='Path to the image file name to captions .txt file')
    parser.add_argument('-o', '--output_file', type=str, default="gscratch/scrubbed/briggs3/datasets", required=False,
        help='Path to output the datasets object for downstream finetuning')
    parser.add_argument('-cache', '--cache-dir', type=str, default="/gscratch/scrubbed/briggs3/.cache", required=False,
        help='Directory where to store cached pre-trained models')
    parser.add_argument('-split', '--train_test_split', type=float, default=0.2, required=False,
        help='How much to partition the train from the test and val datasets')
    parser.add_argument('--overwrite_file', action='store_true', default=False, required=False,
        help="Whether to overwrite the output file if it already exists. If false and the output file exists, raises an error.")
    parser.add_argument('-rs', '--random_seed', type=int, default=10, required=False,
        help='Random_seed for run stability')

    return parser.parse_args()
    

def main():
    args = get_args()
    logging_config(args)
    logging.info(f"Current Configuration args: {args}")

    if (not args.overwrite_file) and (os.path.exists(args.output_file)):
        raise FileExistsError(f"Output file '{args.output_file}' already exists. " 
                              "Either specify a new output_file, or use the flag --overwrite_file!")

    # create metadata.jsonl file
    zip_captions(args)

    # read in the dataset
    data = read_in_data(args)

    # train_val_test split
    data = perform_train_val_test_split(args, data)

    # pickle and save the data set 
    save_dataset(args, data)


if __name__ == '__main__':
    main()
