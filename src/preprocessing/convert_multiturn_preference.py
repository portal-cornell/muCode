import argparse
from datasets import Dataset
from tqdm import tqdm
import random
from copy import deepcopy

from collections import defaultdict

LABEL_KEY = "label"
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


def convert_multiturn_to_paired_data(dataset_path, num_samples=32):
    """
    Loads a dataset from disk and converts it from unpaired preference data to paired preference data.

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        dataset (Dataset): The converted dataset.
    """    
    dataset = Dataset.load_from_disk(dataset_path)

    new_dataset = defaultdict(list)

    for data in dataset:
        # Add all subtrajectories
        for i, prompt in enumerate(data[PROMPT_KEY]):
            if prompt["role"] == "assistant":
                new_prompt = data[PROMPT_KEY][0:1]
                new_completion = [data[PROMPT_KEY][i]]
                new_data = deepcopy(data)
                new_data[PROMPT_KEY] = new_prompt
                new_data[COMPLETION_KEY] = new_completion
                new_data[LABEL_KEY] = False
                for key, value in new_data.items():
                    new_dataset[key].append(value)

        # Add the original data point
        new_data = deepcopy(data)
        new_data[PROMPT_KEY] = data[PROMPT_KEY][0:1]
        new_data[COMPLETION_KEY] = data[COMPLETION_KEY]
        new_data[LABEL_KEY] = data[LABEL_KEY]
        for key, value in new_data.items():
            new_dataset[key].append(value)

    dataset = Dataset.from_dict(new_dataset)

    # Split dataset into positive and negative based on the label
    positive_data = dataset.filter(lambda x: x[LABEL_KEY] == 1)
    negative_data = dataset.filter(lambda x: x[LABEL_KEY] == 0)

    # Get unique prompts by filtering for unique
    user_messages = [x[PROMPT_KEY][0]['content'] for x in dataset]
    unique_prompts = set(user_messages)

    paired_data = {"prompt": [], "chosen": [], "rejected": []}
    no_pairs = 0
    no_pos, no_neg = 0, 0
    for prompt in tqdm(unique_prompts):
        positive_rows = positive_data.filter(lambda x: x[PROMPT_KEY][0]['content'] == prompt)
        negative_rows = negative_data.filter(lambda x: x[PROMPT_KEY][0]['content'] == prompt)

        if len(positive_rows) == 0:
            no_pos += 1

        if len(negative_rows) == 0:
            no_neg += 1

        if len(positive_rows) == 0 or len(negative_rows) == 0:
            no_pairs += 1
            continue

        main_prompt = positive_rows[0][PROMPT_KEY]

        for _ in range(num_samples):
            paired_data["prompt"].append(
                main_prompt
            )
            paired_data["chosen"].append(
                random.choice(positive_rows)[COMPLETION_KEY]
            )
            paired_data["rejected"].append(
                random.choice(negative_rows)[COMPLETION_KEY]
            )

    paired_dataset = Dataset.from_dict(paired_data)

    print ("Prompts with no data: ", no_pairs, no_pos, no_neg, len(unique_prompts))
    return paired_dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the unpaired preference dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the paired preference dataset.")
    args.add_argument("--num_samples", type=int, default=32, required=False, help="Number of samples.")
    args = args.parse_args()

    dataset = convert_multiturn_to_paired_data(args.dataset_path, num_samples=args.num_samples)
    dataset.save_to_disk(args.output_path)