import argparse
import pandas as pd


def create_eval_dicts(captions_df: pd.DataFrame) -> tuple(dict[str, dict[str, str]], dict[str, dict[str, str]]):
    """
    Create two dictionaries for CIDEr evaluation: generated captions and target captions

    :params captions_df: The DataFrame of generated and target captions with associated metadata
    :returns: tuple of generated caption and target caption dictionaries formatted for evaluation
    """
    gen_dict = {}
    tar_dict = {}
    return gen_dict, targ_dict


def load_captions_file(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load in generated captions file

    :params args: The arguments passed in from the console
    :returns: The generated captions DataFrame
    """
    captions_df = pd.read_csv(args.input_file, names=['ROCO_ID', 'file_name', 'generated_captions', 'target_captions'])
    return captions_df


def get_args() -> argparse.Namespace:
    """
    This reads all command line arguments

    :return: The command line argument kwargs
    """
    parser = argparse.ArgumentParser(
        prog="CIDEr",
        description="This will get the CIDEr evaluation scores",
    )
    parser.add_argument('-i', '--input_file', type=str, default="/outputs/generated_captions.csv", required=False)

    return parser.parse_args()


def main():
    args = get_args()

    # Load in generated captions csv
    captions_df = load_captions_file(args)

    # Turn generated and target captions into 2 dictionaries of specified format


    # Pass into CIDEr



if __name__ == '__main__':
    main()