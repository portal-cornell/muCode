"""
This script creates a directory with visualizations for a given data path.

Example Usage:
    python -m src.common.create_viz_script --data_path data/humaneval/results/training_results.csv --output_dir training_results
"""
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import math
import random

from itertools import chain
from datasets import load_from_disk

from src.common.metrics import (
    group_by_first_user_message,
    calculate_best_k_of_n_accuracy,
    calculate_best_k_of_n_precision,
    calculate_best_of_n,
    calculate_pass_at_k
)

PROMPT_KEY = 'prompt'
LABEL_KEY = 'label'
REWARD_KEY = 'reward'

def seed_all(seed):
    """
    Seeds all random number generators.

    Args:
        seed (int): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

def plot_label_distribution(df, output_file):
    """
    This function creates a label distribution plot for the provided dataframe.

    Args:
        df (pd.DataFrame): The rollouts dataframe
        output_file (str): The output file path
    
    Side Effects:
        - Saves a label distribution plot to the output file
    """
    groups = group_by_first_user_message(df)
    groups[LABEL_KEY].sum().hist()
    plt.title('Distribution of labels')
    plt.xlabel('Number of labels')
    plt.ylabel('Frequency')
    plt.savefig(output_file)
    plt.close()

def plot_line_graph(x, y, output_file, title="Title", x_label="X", y_label="Y", linestyle="-", color='blue', marker='o'):
    """
    This function plots a line graph.

    Parameters:
        x (list): List of x values.
        y (list): List of y values.
        output_file (str): The output file path.
        title (str): The title of the plot.
        x_label (str): The x-axis label.
        y_label (str): The y-axis label.
        linestyle (str): The line style.
        color (str): The color of the line.
        marker (str): The marker style.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, marker=marker, linestyle=linestyle, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_accuracy_and_precision(df, output_dir):
    """
    This function creates accuracy and precision plots for the provided dataframe.

    Args:
        df (pd.DataFrame): The rollouts dataframe
        output_dir (str): The output directory
    
    Side Effects:
        - Saves accuracy and precision plots to the output directory
    """
    ### Calculate Accuracy and Precision
    groups = group_by_first_user_message(df)
    N = len(groups.first())
    K_range = range(1, 5)
    accuracy_list = [calculate_best_k_of_n_accuracy(df, K, N) for K in K_range]    
    precision_list = [calculate_best_k_of_n_precision(df, K, N) for K in K_range]

    ### Plot
    plot_line_graph(
        K_range, accuracy_list, f"{output_dir}/best_k_of_n_accuracy.png", 
        title="Best K of N Accuracy", x_label="K", y_label="Accuracy", color='blue')
    plot_line_graph(
        K_range, precision_list, f"{output_dir}/best_k_of_n_precision.png",
        title="Best K of N Precision", x_label="K", y_label="Precision", color='green')

def plot_reward_distribution(df, output_file):
    """
    This function creates a reward distribution plot for the provided dataframe.

    Args:
        df (pd.DataFrame): The evaluated rollouts dataframe
        output_file (str): The output file path
    
    Side Effects:
        - Saves a reward distribution plot to the output file
    """
    ### Calculate distribution
    zero_label = df[df[LABEL_KEY] == 0]
    one_label = df[df[LABEL_KEY] == 1]

    ### Plot
    plt.figure(figsize=(10, 5))
    plt.hist(zero_label[REWARD_KEY], bins=50, alpha=0.5, label='Fail', color='blue')
    plt.hist(one_label[REWARD_KEY], bins=50, alpha=0.5, label='Pass', color='green')
    plt.legend()
    plt.title('Reward Distribution vs Ground Truth')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig(output_file)
    plt.close()

def plot_best_of_n(df, output_file, num_sample_calculations=25, oracle=False):
    """
    This function creates a best of N plot for the provided dataframe.

    The provided dataframe must have a REWARD_KEY column to choose the best of N.

    Args:
        df (pd.DataFrame): The evaluated rollouts dataframe
        output_file (str): The output file path
        num_sample_calculations (int): The number of times to calculate the best of N
        oracle (bool): Whether to use the ground-truth label as the 'reward'

    Side Effects:
        - Saves a best of N plot to the output file
        - Prints best of N=1 and best of N=number of completions
    
    Raises:
        AssertionError: If the number of completions is not the same for each prompt
    """
    ### Calculate Best of N
    groups = group_by_first_user_message(df)
    num_completions = groups.size().iloc[0]
    # assert all(len(group) == num_completions for _, group in groups)

    N = 1 # Best of N=1 == Pass @ K=1
    best_of_n = {} # N -> pass rate
    while N <= 2 ** (math.ceil(math.log(num_completions, 2))): # For every N = 1, 2, 4, 8, 16, ...
        N = min(N, num_completions)
        best_of_n[N] = [calculate_best_of_n(df, N, oracle) for _ in range(num_sample_calculations)]
        N *= 2
    average_best_of_n = {k: sum(v) / num_sample_calculations for k, v in best_of_n.items()} # Average over all calculations
    n_values = list(best_of_n.keys())

    ### Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(
        list(chain(*[[k] * num_sample_calculations for k in best_of_n.keys()])),
        list(chain(best_of_n.values())),
        alpha=0.6,
        label="Data Points",
        s=10
    )
    plt.plot(
        n_values,
        list(average_best_of_n.values()),
        color='blue',
        label="Average Solve Rate"
    )

    # Logarithmic Scale for x-axis
    plt.xscale('log')
    plt.xticks(n_values, labels=[str(xi) for xi in n_values])  # Evenly spaced labels

    # Axis Labels
    plt.xlabel("Number of rollouts (N)")
    plt.ylabel("Solve Rate (%)")
    plt.title("Best of N")

    # Style and Show
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    with open(output_file.replace('.png', '.txt'), 'w') as f:
        f.write(f"Best of N=1: {average_best_of_n[1]}\n")
        f.write(f"Best of N={num_completions}: {average_best_of_n[num_completions]}\n")
        print (f"Best of N=1: {average_best_of_n[1]}\n")
        print (f"Best of N={num_completions}: {average_best_of_n[num_completions]}\n")

def plot_pass_at_k(df, output_file, label="pass@k", color="blue", ax=None, save_fig=True, log=True, close=True):
    """
    This function creates a pass@k plot for the provided dataframe.

    Args:
        df (pd.DataFrame): The rollouts dataframe
        output_file (str): The output file path
        label (str): The label for the plot
        color (str): The color for the plot
        ax (plt.Axes): The axes to plot on
        save_fig (bool): Whether to save the plot
        log (bool): Whether to log the pass@k in a text file
        close (bool): Whether to close the plot after saving
    
    Side Effects:
        - Saves a pass@k plot to the output file
        - Prints pass@1 and pass@10
    """
    ### Calculate Pass@K
    groups = group_by_first_user_message(df)
    num_completions = groups.size().iloc[0]
    # assert all(len(group) == num_completions for _, group in groups)
    pass_at_k = {k: np.mean(calculate_pass_at_k(df, num_completions, k)) for k in range(1, num_completions + 1)}

    ### Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    ax.plot(
        list(pass_at_k.keys()),
        list(pass_at_k.values()),
        color=color,
        label=label
    )

    # Axis Labels
    ax.set_xlabel("Number of rollouts (k)")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Pass @ k")

    # Style and Show
    ax.legend()
    fig.tight_layout()
    if save_fig:
        fig.savefig(output_file)
    if log:
        with open(output_file.replace('.png', '.txt'), 'w') as f:
            for i in range(1, num_completions + 1):
                f.write(f"Pass @ {i}: {pass_at_k[i]}\n")
                print (f"Pass @ {i}: {pass_at_k[i]}\n")
    if close:
        plt.close(fig)

def prepare_dataset(data_path, remove_canonical=False):
    """
    This function prepares the dataset for visualization.

    Args:
        data_path (str): The path to the dataset.
        remove_canonical (bool): Whether to remove canonical solutions.

    Returns:
        pd.DataFrame: The prepared dataset.
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        df[LABEL_KEY] = df[LABEL_KEY].astype(int)
    else:
        dataset = load_from_disk(data_path)

        # Get unique prompts by filtering for unique
        if remove_canonical:
            # user_messages = dataset.filter(lambda x: len(x['']) == 2)
            user_messages = [x['prompt'][0]['content'] for x in dataset]
            unique_prompts = set(user_messages)
            dataset = dataset.select(range(0, len(dataset) - len(unique_prompts)))

        # convert list types to string in dataset
        for key in dataset.column_names:
            if isinstance(dataset[key][0], list):
                dataset = dataset.map(lambda x: {key: str(x[key])}, remove_columns=[key])
        df = dataset.to_pandas()
    return df

def create_viz(data_path, seed, output_dir, remove_canonical=False, oracle=False):
    """
    This function creates visualizations for the given CSV file.

    Args:
        data_path (str): The path to the CSV file.
        seed (int): The random seed for the best of N plot.
        output_dir (str): The output directory.
        remove_canonical (bool): Whether to remove canonical solutions.
        oracle (bool): Whether to visualize the ground-truth label as the 'reward'

    Side Effects:
       - Creates output directory if it does not exist.
       - Creates visualizations in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = prepare_dataset(data_path, remove_canonical)

    plot_label_distribution(df, f"{output_dir}/label_distribution.png") # Plot label distribution

    if REWARD_KEY in df:
        plot_accuracy_and_precision(df, output_dir) # Plot accuracy and precision
        plot_reward_distribution(df, f"{output_dir}/reward_distribution.png") # Plot reward distribution
        seed_all(seed) # Seed best of N sampling
        plot_best_of_n(df, f"{output_dir}/best_of_n.png") # Plot best of N
    if oracle:
        seed_all(seed)
        plot_best_of_n(df, f"{output_dir}/best_of_n_oracle.png", oracle=True)
    plot_pass_at_k(df, f"{output_dir}/pass_at_k.png") # Plot pass@k

def create_overlay_viz(data_paths, data_labels, seed, output_dir, data_colors=None, remove_canonical=False, oracle=False):
    """
    This function creates overlayed visualizations for the given CSV files.

    Args:
        data_paths (list): List of paths to the CSV files.
        data_labels (list): List of labels for the CSV files.
        seed (int): The random seed for the best of N plot.
        output_dir (str): The output directory.
        data_colors (list): List of colors for the CSV files.
        remove_canonical (bool): Whether to remove canonical solutions.
        oracle (bool): Whether to visualize the ground-truth label as the 'reward'

    Side Effects:
       - Creates output directory if it does not exist.
       - Creates visualizations in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if data_colors is None:
        data_colors = sns.color_palette("husl", len(data_paths))

    dfs = [prepare_dataset(data_path, remove_canonical) for data_path in data_paths]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, df in enumerate(dfs):
        plot_pass_at_k(df, f"{output_dir}/pass_at_k_{data_labels[i]}.png", label=data_labels[i], color=data_colors[i], ax=ax, save_fig=False, close=False)
    fig.savefig(f"{output_dir}/combined_pass_at_k.png")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    # Single data path
    argparser.add_argument("--data_path", type=str, help="Path to the data to parse.")
    
    # Multiple data paths
    argparser.add_argument("--data_paths", type=str, default=False, help="Comma-separated list of data paths to overlay visualizations.")
    argparser.add_argument("--data_labels", type=str, default=False, help="Comma-separated list of data labels for overlaying visual.")
    argparser.add_argument("--data_colors", type=str, default=False, help="Comma-separated list of data colors for overlaying visual.")

    argparser.add_argument("--output_dir", required=True, type=str, help="Path to the output directory.")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for the best of N plot.")
    argparser.add_argument("--remove_canonical", type=bool, default=False, help="Whether to remove canonical solution.")
    argparser.add_argument("--oracle", action='store_true', help="Whether to use the ground-truth label as the 'reward'")
    args = argparser.parse_args()

    if args.data_path:
        create_viz(args.data_path, args.seed, args.output_dir, args.remove_canonical, args.oracle)
    elif args.data_paths and args.data_labels:
        data_paths = args.data_paths.split(',')
        data_labels = args.data_labels.split(',')
        data_colors = args.data_colors.split(',') if args.data_colors else None
        create_overlay_viz(data_paths, data_labels, args.seed, args.output_dir, data_colors, args.remove_canonical, args.oracle)
