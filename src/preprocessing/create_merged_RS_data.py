"""
This script creates a Hugging Face dataset of single-turn unpaired preference data to paired preference data.

The dataset is expected to be in unpaired preference format, where each datapoint consists of a prompt, completion, and label.
"""
import argparse
from datasets import Dataset
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import ast 

LABEL_KEY = "label"
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


def create_multiturn_RS_data_topK(dataset_path, K=1, num_to_relabel=3, merge_positive_data=False):
    """
    Loads a dataset from disk and converts it from unpaired preference data to paired preference data.

    [RFT w/ OV] Only include data with label = 1
    [RFT w/ LV] Include up to 'K' samples with the highest reward
    [MuCode] Include up to 'K' samples relabeled with 'num_to_relabel' samples all w/ highest reward

    Args:
        dataset_path (str): The path to the dataset.
        K (int): The number of top trajectories to select
        num_to_relabel (int): The number of trajectories to relabel (if any)
        merge_positive_data (bool): Whether to merge positive data with the relabeled data.

    Returns:
        dataset (Dataset): The converted dataset.
    """
    dataset = pd.read_csv(dataset_path)

    dataset['reward'] = dataset['reward'] + 100 * dataset['label']

    # Select top K trajectories based on reward of final code completion
    df = dataset.copy()[['data_id', COMPLETION_KEY, 'reward']]
    topK_df = (
        df.sort_values(by=['data_id', 'reward'], ascending=[True, False])  # Sort by prompt descendin score
        .groupby('data_id')                                                # Group by prompt
        .head(K)                                                          # Select top 5 for each group
    ) # Select top K for each group (in our case 1)
    topK_df = topK_df.drop(columns=['reward'])

    # Select top 'num_to_relabel' trajectories, prioritizing negative samples, for relabeling
    relabel_df = dataset.copy().drop(columns=[COMPLETION_KEY])
    relabel_df = (
        relabel_df.sort_values(by=['data_id', 'label', 'reward'], ascending=[True, True, False])  # Sort by prompt descendin score
        .groupby('data_id')                                                # Group by prompt
        .head(num_to_relabel)                                              # Select top 5 for each group
    )

    # Merge relabel_df with topK_df to fill in missing relabel_df COMPLETION_KEY values with the top K completions
    df = pd.merge(relabel_df, topK_df, how='left', on='data_id') # Relabels relabel_df using topK_df

    if merge_positive_data:
        df = df[df['label'] == 0] # Ensure we only relabel negative samples
        # Merge positive data with relabel_df
        pos_data = dataset[dataset['label'] == 1]
        df = pd.concat([df, pos_data])

    df[PROMPT_KEY] = df[PROMPT_KEY].apply(lambda x: ast.literal_eval(x))
    df[COMPLETION_KEY] = df[COMPLETION_KEY].apply(lambda x: ast.literal_eval(x))
    df[LABEL_KEY] = df[LABEL_KEY].apply(lambda x: bool(x))

    df_final = df[['data_id', PROMPT_KEY, COMPLETION_KEY, LABEL_KEY]]

    final_data = Dataset.from_pandas(df_final)
    print (final_data)
    return final_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the unpaired preference dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the paired preference dataset.")
    args.add_argument("--K", type=int, required=True, help="Number of top samples for Rejection Sampling.")
    args.add_argument("--num_to_relabel", type=int, help="Number of trajectories to relabel.")
    args.add_argument("--merge_positive_data", action='store_true', help="Whether to merge positive data with relabeled data.")

    args = args.parse_args()
    dataset = create_multiturn_RS_data_topK(args.dataset_path, K=args.K, num_to_relabel=args.num_to_relabel, merge_positive_data=args.merge_positive_data)
    dataset.save_to_disk(args.output_path)