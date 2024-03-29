import argparse
import pandas as pd

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from typing import Any


def compute_scores(generated: dict[Any], target: dict[Any]) -> float:
    """
    This computes the CIDEr score for the given dictionaries of captions

    :param gen_dict: Dictionary containing generated captions
    :param tar_dict: Dictionar containing target captions
    :return: The overall CIDEr score between the given generated and target captions
    """
    # tokenizer = PTBTokenizer()

    # target = tokenizer.tokenize(tar_dict)
    # generated = tokenizer.tokenize(gen_dict)

    scores_dict = {}

    cider = Cider()
    score, _ = cider.compute_score(generated, target)
    scores_dict["CIDEr"] = round(score * 10, 3)

    meteor = Meteor()
    score, _ = meteor.compute_score(generated, target)
    scores_dict["METEOR"] = score


    rouge = Rouge()
    score, _ = rouge.compute_score(generated, target)
    scores_dict["ROUGE"] = score


    bleu = Bleu()
    score, _ = bleu.compute_score(generated, target)
    scores_dict["BLEU"] = sum(score) / len(score)

    return scores_dict


def create_eval_dicts(captions_df: pd.DataFrame) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    Create two dictionaries for CIDEr evaluation: generated captions and target captions. Dictionary format is
        as follows:

        {"roco_id": [{"caption": "caption_text"}]}

    :params captions_df: The DataFrame of generated and target captions with associated metadata
    :returns: tuple of generated caption and target caption dictionaries formatted for evaluation
    """
    gen_dict = {}
    tar_dict = {}

    roco_id = list(captions_df["ROCO_ID"])
    gen_captions = list(captions_df["generated_captions"])
    target_captions = list(captions_df["target_captions"])

    for id, gen, target in zip(roco_id, gen_captions, target_captions):
        if id in gen_dict:
            raise ValueError(f"gen_dict already contains {id}")

        if id in tar_dict:
            raise ValueError(f"tar_dict already contains {id}")

        gen_dict[id] = [{"caption": gen}]
        tar_dict[id] = [{"caption": target}]

    return gen_dict, tar_dict


def load_captions_file(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load in generated captions file

    :params args: The arguments passed in from the console
    :returns: The generated captions DataFrame
    """
    captions_df = pd.read_csv(args.input_file, names=['ROCO_ID', 'generated_captions', 'target_captions'])

    captions_df.drop_duplicates(inplace=True)

    for col in captions_df.columns:
        captions_df[col] = captions_df[col].astype(str)

    assert len(captions_df) == captions_df['ROCO_ID'].nunique()

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
    parser.add_argument('-i', '--input_file', type=str, default="outputs/generated_captions.csv", required=False)
    parser.add_argument('-o', '--output_file', type=str, default="outputs/cider_score.txt", required=False)

    return parser.parse_args()


def main():
    args = get_args()

    # Load in generated captions csv
    captions_df = load_captions_file(args)

    # Turn generated and target captions into 2 dictionaries of specified format
    gen_dict, tar_dict = create_eval_dicts(captions_df)

    tokenizer = PTBTokenizer()

    target = tokenizer.tokenize(tar_dict)
    generated = tokenizer.tokenize(gen_dict)

    # Pass into CIDEr
    scores_dict = compute_scores(generated, target)

    print(scores_dict)

    out_str = ""
    for metric, score in scores_dict.items():
        out_str += metric + ": " + str(score) + "\n"
    
    with open(args.output_file, 'w') as out:
        out.write(out_str)


if __name__ == '__main__':
    main()