"""
This script expands a multi-turn Huggingface dataset by converting entire rollouts into subtrajectories.

The dataset is expected to be in unpaired preference format, where each datapoint consists of a prompt, completion, and label.
"""
import argparse
from datasets import Dataset
from tqdm import tqdm
from copy import deepcopy

from collections import defaultdict

LABEL_KEY = "label"
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"

def convert_multi_to_intermediate_subtrajectories(dataset_path):
    """
    This function takes a multi-turn dataset and converts it into intermediate subtrajectories for each step.

    For some trajectory (s0, a0, s1, a1, s2, ..., sn-1, an-1), we convert it into the following subtrajectories:
    (s0, a0), (s0, a0, s1, a1), (s0, a0, s1, a1, s2, a2), ..., (s0, a0, s1, a1, s2, ..., sn-1, an-1)

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        dataset (Dataset): The truncated dataset.
    """
    dataset = Dataset.load_from_disk(dataset_path)
    new_dataset = defaultdict(list)
    for data in dataset:
        # Add all subtrajectories
        for i, prompt in enumerate(data[PROMPT_KEY]):
            if prompt["role"] == "assistant":
                new_prompt = data[PROMPT_KEY][:i]
                new_completion = [data[PROMPT_KEY][i]]
                new_data = deepcopy(data)
                new_data[PROMPT_KEY] = new_prompt
                new_data[COMPLETION_KEY] = new_completion
                new_data[LABEL_KEY] = False
                for key, value in new_data.items():
                    new_dataset[key].append(value)
        # Add the original data point
        for key, value in data.items():
            new_dataset[key].append(value)
    subtrajectory_dataset = Dataset.from_dict(new_dataset)
    return subtrajectory_dataset

def convert_multi_to_relabeled_subtrajectories(dataset_path):
    """
    This function takes a multi-turn dataset and relabels based on the last action.

    For some trajectory (s0, a0, s1, a1, s2, ..., sn-1, an-1), we convert it into the following subtrajectories:
    (s0, an-1), (s0, a0, s1, an-1), (s0, a0, s1, a1, s2, an-1), ..., (s0, a0, s1, a1, s2, ..., sn-1, an-1)

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        dataset (Dataset): The truncated dataset.
    """
    dataset = Dataset.load_from_disk(dataset_path)
    new_dataset = defaultdict(list)
    for data in dataset:
        # Add all positive subtrajectories
        for i, prompt in enumerate(data[PROMPT_KEY]):
            if prompt["role"] == "assistant":
                new_prompt = data[PROMPT_KEY][:i]
                new_data = deepcopy(data) # Positive completion and true label remain the same
                new_data[PROMPT_KEY] = new_prompt
                for key, value in new_data.items():
                    new_dataset[key].append(value)
        # Add the original data point
        for key, value in data.items():
            new_dataset[key].append(value)
    subtrajectory_dataset = Dataset.from_dict(new_dataset)
    return subtrajectory_dataset

def convert_multi_to_positive_subtrajectories(dataset_path):
    """
    This function takes a multi-turn dataset and converts it into positive subtrajectories.

    For some positive trajectory (s0, a0, s1, a1, s2, ..., sn-1, an-1), we convert it into the following subtrajectories:
    (s0, an-1), (s0, a0, s1, an-1), (s0, a0, s1, a1, s2, an-1), ..., (s0, a0, s1, a1, s2, ..., sn-1, an-1)

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        dataset (Dataset): The truncated dataset.
    """
    dataset = Dataset.load_from_disk(dataset_path)
    dataset = dataset.filter(lambda x: x[LABEL_KEY] == 1)
    new_dataset = defaultdict(list)
    for data in dataset:
        # Add all positive subtrajectories
        for i, prompt in enumerate(data[PROMPT_KEY]):
            if prompt["role"] == "assistant":
                new_prompt = data[PROMPT_KEY][:i]
                new_data = deepcopy(data) # Positive completion and true label remain the same
                new_data[PROMPT_KEY] = new_prompt
                for key, value in new_data.items():
                    new_dataset[key].append(value)
        # Add the original data point
        for key, value in data.items():
            new_dataset[key].append(value)
    subtrajectory_dataset = Dataset.from_dict(new_dataset)
    return subtrajectory_dataset

# def convert_multi_to_last_action_subtrajectories(dataset_path):
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the multi-turn dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the single-turn dataset.")
    args.add_argument("--positive_only", default=False, action="store_true", help="Whether to convert trajectory into positive subtrajectorys.")
    args.add_argument("--relabel_only", default=False, action="store_true", help="Whether to convert trajectory into relabeled subtrajectory.")
    args = args.parse_args()

    if args.positive_only:
        dataset = convert_multi_to_positive_subtrajectories(args.dataset_path)
    elif args.relabel_only:
        dataset = convert_multi_to_relabeled_subtrajectories(args.dataset_path)
    else:
        dataset = convert_multi_to_intermediate_subtrajectories(args.dataset_path)
    dataset.save_to_disk(args.output_path)