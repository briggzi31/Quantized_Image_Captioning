import argparse
import logging
import os

from datasets import load_dataset, Dataset, Image


def read_in_data(args: argparse.Namespace) -> Dataset:
    """
    This loads in a dataset from the specified directory, and returns a 
        HuggingFace datasets object.
    """
    image_dir = os.path.join(args.input_dir, "images")

    print(os.listdir(image_dir))

    # /gscratch/scrubbed/briggs3/data/flickr8k/images

    image_data = load_dataset("imagefolder", data_dir=image_dir, split='train')

    return image_data

    


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
    
    parser.add_argument('-l', '--log-file', type=str, default="logs/datasets/log.log", required=False)
    parser.add_argument('-i', '--input_dir', type=str, default="/gscratch/scrubbed/briggs3/data", required=False)
    parser.add_argument('-o', '--output_dir', type=str, default="gscratch/scrubbed/briggs3/datasets", required=False)
    parser.add_argument('-c', '--cache-dir', type=str, default="/gscratch/scrubbed/briggs3/.cache", required=False)

    return parser.parse_args()
    

def main():
    args = get_args()
    logging_config(args)
    logging.info(f"Current Configuration args: {args}")
    print("args.input", args.input_dir)

    data = read_in_data(args)
    print(data)


if __name__ == '__main__':
    main()

