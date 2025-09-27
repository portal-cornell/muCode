"""
This script creates a Hugging Face dataset of single-turn unpaired preference data to paired preference data.

The dataset is expected to be in unpaired preference format, where each datapoint consists of a prompt, completion, and label.
"""
import argparse

from pandas.core import base
from datasets import Dataset, concatenate_datasets

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the unpaired preference dataset.")
    args.add_argument("--new_dataset_path", type=str, required=True, help="The path to the unpaired iteration dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the paired preference dataset.")
    args = args.parse_args()

    base_dataset = Dataset.load_from_disk(args.dataset_path)
    new_dataset = Dataset.load_from_disk(args.new_dataset_path)

    cols_to_remove = new_dataset.column_names
    for cols in base_dataset.column_names:
        cols_to_remove.remove(cols)

    new_dataset = new_dataset.remove_columns(cols_to_remove)

    dataset = concatenate_datasets([base_dataset, new_dataset])

    dataset.save_to_disk(args.output_path)
