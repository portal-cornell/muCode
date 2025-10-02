"""
This script filters a Hugging Face dataset of unpaired preference data to positive completions for SFT.

The dataset is expected to be in unpaired preference format, where each datapoint consists of a prompt, completion, and label.
"""
import argparse
from datasets import Dataset
import random
from tqdm import tqdm


LABEL_KEY = "label"
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


def convert_unpaired_to_sft_data(dataset_path, num_samples=100):
    """
    Loads a dataset from disk and filters it to positive completions for SFT.

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        dataset (Dataset): The converted dataset.
    """
    dataset = Dataset.load_from_disk(dataset_path)
    dataset = dataset.filter(lambda x: x[LABEL_KEY] == 1)

    print (" Upsample: ", args.upsample)
    if args.upsample:
        # Get unique prompts by filtering for unique
        user_messages = [x[PROMPT_KEY][0]['content'] for x in dataset]
        unique_prompts = set(user_messages)

        new_data = {"data_id": [], "prompt": [], "completion": [], "label": []}

        no_pos = 0

        for prompt in tqdm(unique_prompts):
            positive_rows = dataset.filter(lambda x: x[PROMPT_KEY][0]['content'] == prompt)

            main_prompt = positive_rows[0][PROMPT_KEY]
            data_id = positive_rows[0]["data_id"]

            if len(positive_rows) == 0:
                no_pos += 1
                continue

            for _ in range(num_samples):
                new_data["data_id"].append(
                    data_id
                )
                new_data[PROMPT_KEY].append(
                    main_prompt
                )
                new_data[COMPLETION_KEY].append(
                    random.choice(positive_rows)[COMPLETION_KEY]
                )
                new_data[LABEL_KEY].append(
                    positive_rows[0][LABEL_KEY]
                )
        print (" Data with no positives: ", no_pos)
        dataset = Dataset.from_dict(new_data)

    print (dataset)
    return dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the multi-step dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the single-step dataset.")
    args.add_argument("--upsample", default=False, action="store_true", help="Whether to upsample the positive rows.")
    args = args.parse_args()

    dataset = convert_unpaired_to_sft_data(args.dataset_path)
    dataset.save_to_disk(args.output_path)
