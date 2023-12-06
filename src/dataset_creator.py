"""
dataset_creator

This module will load in a directory full of images and corresponding captions into a HuggingFace
    image datasets.DatasetDict object

usage:
    Run from Quantized_Image_Captioning directory:

        ./scripts/dataset_creator.sh

        OR

        sbatch slurm/dataset_creator_gpu.slurm
"""

import argparse
import logging
import os
import json
import pickle

import PIL
import PIL.Image

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
    if (not args.overwrite_file) and (os.path.exists(args.output_file)):
        raise FileExistsError(f"Output file '{args.output_file}' already exists. " 
                              "Either specify a new output_file, or use the flag --overwrite_file!")

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
    captions = pd.read_csv(args.caption_file, sep='\t', header=None, names=['ROCO_ID', 'text'])
    licenses = pd.read_csv(args.licenses_file)

    captions = pd.merge(captions, licenses, on="ROCO_ID", how="inner")
    captions.rename(columns={"PMC_ID": "file_name"}, inplace=True)

    # Identify and/or delete non-existent or corrupted images
    for image_name in os.listdir(args.image_input_dir):
        image_path = os.path.join(args.image_input_dir, image_name)

        # image doesn't exist
        if not os.path.exists(image_path):
            captions.drop(captions.loc[captions['file_name']==image_name].index, inplace=True)
        else: 
            try:
                image = PIL.Image.open(image_path)
            except PIL.UnidentifiedImageError as e:  #image does exist but is corrupted
                print(f"Error in file {filename}: {e}")
                os.remove(os.path.join(folder_path, filename))
                captions.drop(captions.loc[captions['file_name']==image_name].index, inplace=True)
                print(f"Removed file {filename}")
    
    # create json
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
        prog="Dataset Creator",
        description="This will create a HuggingFace Image Captioning Datasets object for MedICaT",
    )
    
    parser.add_argument('-l', '--log-file', type=str, default="logs/datasets/log.log", required=False,
        help='Path to output the log file.')
    parser.add_argument('-i', '--data_dir', type=str, 
        default="/gscratch/scrubbed/briggs3/data/all_data", required=False,
        help='Directory where all roco data is kept.')
    parser.add_argument('-o', '--output_dir', type=str, default="gscratch/scrubbed/briggs3/all_data/datasets", required=False,
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

    all_data = {
        'radiology': {},
        'non-radiology': {}
    }
    
    for split in ['train', 'test', 'validation']:
        cur_split = split if split != 'validation' else 'val'

        for image_type in ['radiology', 'non-radiology']:
            cur_directory = os.path.join(args.data_dir, split, image_type)

            args.image_input_dir = os.path.join(cur_directory, "images")
            args.caption_file = os.path.join(cur_directory, "captions.txt")
            args.licenses_file = os.path.join(cur_directory, "licences.txt")

            # create metadata.jsonl file
            zip_captions(args)

            # read in the dataset to a HuggingFace Dataset object
            data: Dataset = read_in_data(args)

            all_data[image_type][cur_split] = data

    radiology_data = DatasetDict(all_data['radiology'])
    non_radiology_data = DatasetDict(all_data['non-radiology'])


    # pickle and save the data set 
    args.output_file = os.path.join(args.output_dir, "radiology_data.pkl")
    save_dataset(args, radiology_data)

    args.output_file = os.path.join(args.output_dir, "non-radiology_data.pkl")
    save_dataset(args, non_radiology_data)


if __name__ == '__main__':
    main()
