import argparse
import pandas
from datasets import Dataset


def filter_topK_prompt(data_path, K=1, num_rollout_experiments=None, ground_truth=False, public_truth=False, public_and_reward_truth=False, select_all=False, random=False):
    """
    Calculate the top K rollouts for each prompt based on the reward.

    For convenience, the top K of multiple experiments can be calculated by passing an ordered
    dataset with the rollouts for each experiment in order. For example, for 3 experiments with
    4 rollouts each, the dataset should be ordered as follows:

    [ 
      e_1, e_2, e_3,
      e_1, e_2, e_3,
      e_1, e_2, e_3,
      e_1, e_2, e_3,
    ]

    such that idx % num_rollout_experiments gives the experiment number.

    Args:
        data_path (str): The path to the rollouts.
        K (int): The number of top rollouts to keep.
        num_rollout_experiments (int): The number of rollout experiments in the dataset.
        ground_truth (bool): Whether to select based on ground truth.
        public_truth (bool): Whether to select based on public truth.
        public_and_reward_truth (bool): Whether to select based on public and reward truth.
        select_all (bool): Whether to select all rollouts.

    Returns:
        Dataset: A dataset with the filtered top K rollouts.
    """

    df = pandas.read_csv(data_path)
    if ground_truth:
        selection_field = 'label'
    elif public_truth:
        df['public_label'] = df['feedbacks'].apply(eval).apply(lambda x: x[0]['public'] == "Code passed all tests")
        selection_field = 'public_label'
    elif public_and_reward_truth:
        df['public_and_reward_label'] = df['feedbacks'].apply(eval).apply(lambda x: (x[0]['public'] == "Code passed all tests") * 100)
        df['public_and_reward_label'] += df['reward']
        selection_field = 'public_and_reward_label'
    else:
        selection_field = 'reward' 

    if num_rollout_experiments is not None:
        df = (
            df.sort_values("ID", ascending=True)                                                        # Sort by original order of data
            .groupby('data_id', group_keys=False)                                                       # Group by data_id
            .apply(lambda group: group.assign(
                experiment_group=group.reset_index(drop=True).index % num_rollout_experiments)          # Separate into contiguous groups
            )
            .groupby(['data_id', 'experiment_group'], group_keys=False)                                 # Group by data_id and contiguous rollout groups
            .apply(lambda group: group.nlargest(K if not select_all else len(group), selection_field) if not random else group.sample(K)) # Select top K for each group
        )
        df = df.drop(columns='experiment_group')
    else:
        df = (
            df.sort_values(by=['data_id', selection_field], ascending=[True, False])     # Sort by prompt and reward/label
            .groupby('data_id')                                                          # Group by prompt
            .head(K)                                                                     # Select top K for each group
        )

    # Update fields
    df['prompt'] = df['prompt'].apply(eval)
    df['completion'] = df['completion'].apply(eval)
    df['feedbacks'] = df['feedbacks'].apply(eval)
    df['label'] = df['label'].apply(bool)
    dataset = Dataset.from_pandas(df)

    return dataset

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="The path to the unpaired preference dataset.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the paired preference dataset.")
    args.add_argument("--K", type=int, required=True, help="Number of top responses to keep.")
    args.add_argument("--num_rollout_experiments", type=int, default=None, help="Number of rollout experiments in the dataset.")
    args.add_argument("--ground_truth", action="store_true", help="Whether to select based on ground truth.")
    args.add_argument("--public_truth", action="store_true", help="Whether to select based on public truth.")
    args.add_argument("--public_and_reward_truth", action="store_true", help="Whether to select based on public and reward truth.")
    args.add_argument("--select_all", action="store_true", help="Whether to select all rollouts.")
    args.add_argument("--random", action="store_true", help="Whether to select randomly.")

    args = args.parse_args()
    assert sum([args.ground_truth, args.public_truth, args.public_and_reward_truth, args.random]) <= 1, "Only one of ground_truth, public_truth, and public_and_reward_truth, and select_all can be selected."
    dataset = filter_topK_prompt(
        args.dataset_path, 
        K=args.K, 
        num_rollout_experiments=args.num_rollout_experiments, 
        ground_truth=args.ground_truth, 
        public_truth=args.public_truth,
        public_and_reward_truth=args.public_and_reward_truth,
        select_all=args.select_all,
        random=args.random
    )
    dataset.save_to_disk(args.output_path)