"""
This library contains metrics used in our visualizations
"""
import math

PROMPT_KEY = 'prompt'
LABEL_KEY = 'label'
REWARD_KEY = 'reward'

def group_by_first_user_message(df):
    """
    This function groups the dataframe by the first user message.

    Args:
        df (pd.DataFrame): The rollouts dataframe

    Returns:
        pd.DataFrame: The grouped dataframe
    """
    return df.groupby(df[PROMPT_KEY].apply(lambda x: str(eval(x)[0])))

def calculate_best_k_of_n_accuracy(df, K, N):
    """
    Calculate the best K of N accuracy for a Pandas Dataframe.
    
    We assume that if there are D unique keys in the Dataframe then each key has
    N successive rows. We calculate the K of N accuracy for each group of N rows.
    
    Args:
        df (pd.DataFrame): The Pandas DataFrame
        K (int): The number of best rows to consider
        N (int): The number of rows in each group
    
    Returns:
        float: The accuracy of the best K of N
    
    Raises:
        ValueError: If the Dataframe size is not a multiple of N
    """
    # if len(df) % N != 0:
    #     raise ValueError("Dataframe size is not a multiple of N.")
    
    groups = group_by_first_user_message(df)
    results = []
    for _, group in groups:
        top_k = group.nlargest(K, REWARD_KEY)
        k_accuracy = top_k[LABEL_KEY].sum() / K
        results.append(k_accuracy)
    return sum(results) / len(results)

def calculate_best_k_of_n_precision(df, K, N):
    """
    Calculate the best K of N precision for a Pandas Dataframe.
    
    We assume that if there are D unique keys in the Dataframe then each key has
    N successive rows. We calculate the K of N precision for each group of N rows.
    
    Args:
        df (pd.DataFrame): The Pandas DataFrame
        K (int): The number of best rows to consider
        N (int): The number of rows in each group
    
    Returns:
        float: The precision of the best K of N
    
    Raises:
        ValueError: If the Dataframe size is not a multiple of N
    """
    # if len(df) % N != 0:
    #     raise ValueError("Dataframe size is not a multiple of N.")
    
    groups = group_by_first_user_message(df)
    results = []
    for _, group in groups:
        top_k = group.nlargest(K, REWARD_KEY)
        k_precision = (top_k[REWARD_KEY] > 0).eq(top_k[LABEL_KEY] > 0.5).mean()
        results.append(k_precision)
    return sum(results) / len(results)

def calculate_best_of_n(df, N, oracle=False):
    """
    Calculate the best of N for a Pandas Dataframe.
    
    We assume that if there are D unique keys in the Dataframe then each key has
    N successive rows. We calculate the best of N for each group.

    Args:
        df (pd.DataFrame): The Pandas DataFrame
        N (int): The number of rows in each group
    
    Returns:
        float: The best of N
    """
    groups = group_by_first_user_message(df)
    results = []
    for _, group in groups:
        samples = group.sample(min(N, len(group))) # Sample N completions
        best_sample = samples.nlargest(1, LABEL_KEY if oracle else REWARD_KEY).iloc[0] # Choose the best completion
        results.append(best_sample[LABEL_KEY] == 1) # Check if the best completion is correct
    return sum(results) / len(results)

def calculate_pass_at_k(df, n, k):
    """
    Calculate the pass rate at K for a Pandas Dataframe.
    
    We assume that if there are D unique keys in the Dataframe then each key has
    N successive rows. We calculate the pass rate at k for each group.
    
    Args:
        df (pd.DataFrame): The Pandas DataFrame
        n (int): The number of completions
        k (int): The number of 'top' completions to consider
    
    Returns:
        float: The pass rate at K
    """
    groups = group_by_first_user_message(df)
    results = []
    total_combinations = math.comb(n, k) # n should be the same for all groups
    for _, group in groups:
        c = group[LABEL_KEY].sum() # Number of correct completions
        fail_prob = math.comb(n - c, k) / total_combinations
        results.append(1 - fail_prob) # 1 - fail_prob = pass_prob = pass @ k
    return sum(results) / len(results)