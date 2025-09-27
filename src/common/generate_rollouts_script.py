"""
This script generates rollouts for a given dataset using the SGLang API.

There are two ways to use this script.

Option 1: Host an SGLang server
First, ensure an SGLang server is running. The SGLang server can be started using the following command:
```
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --host 0.0.0.0
```

You can then generate 100 rollouts with Llama-3.1-8B Instruct on single-step HumanEval using the 
following command: 
```
python -m src.common.generate_rollouts_script \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.7 \
    --base_url http://127.0.0.1:30000/v1 \
    --dataset openai_humaneval \
    --split test \
    --save_path data/humaneval/test/generated_rollouts \
    --rollouts 100 \
    --max_steps 1 \
    --parallel
```

Option 2: Start a new SGLang server in the script
Alternatively, you can start a new SGLang server in the script. Simply specify an open port.
```
python -m src.common.generate_rollouts_script \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --temperature 0.7 \
    --port 30000 \
    --dataset openai_humaneval \
    --split test \
    --save_path data/humaneval/test/generated_rollouts \
    --rollouts 100 \
    --max_steps 1 \
    --parallel
```
"""
import argparse
import atexit
import subprocess
import signal
import asyncio
import functools
import traceback
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset, load_dataset
import openai
import gymnasium as gym
import random
import re
from tqdm import tqdm
import sglang as sgl
import torch
from accelerate import Accelerator
import os
import logging
import time
import gc

print("Loaded imports")

# Disable info-level logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.env.benchmarks import (
    MBPPWrapperEnv,
    MBPPTrainWrapperEnv,
    MBPPPlusWrapperEnv,
    CodeContestsWrapperEnv,
    HumanEvalPackWrapperEnv,
    HumanEvalPlusWrapperEnv
)
print("Loaded benchmarks")

def combine_successive_role_messages(messages):
    """Combines successive role messages into a single message.

    LLM models like Llama-2 only allow user/assistant/user/assistant/... message order. This
    function formats the messages to comply with this requirement.

    Args:
        messages (List[Dict[str, str]]): The messages to combine.

    Returns:
        new_messages (List[Dict[str, str]]): The messages with successive role messages combined.
    """
    new_messages = [messages[0]]
    last_role = None
    for message in messages[1:]:
        if last_role == message["role"]:
            new_messages[-1]["content"] += f"\n{message['content']}"
        else:
            new_messages.append(message)
        last_role = message["role"]
    return new_messages

def start_sglang_server(model_path, port, tp, dist_url="localhost:29500", deterministic=False):
    """
    Starts an SGLang server on the given port.

    We reserve the GPUs with the highest IDs for SGLang. For example,
    for 4 GPUs and tp=2, we reserve GPUs 2 and 3 for SGLang.

    If you are running this to evaluate during training, you must set
    dist_url to the same MASTER_ADDR:MASTER_PORT as the training script.

    Args:
        model_path (str): The model to use for code generation.
        port (int): The port to run the server on.
        tp (int): The number of GPUs to use for tensor parallelism.
        dist_url (str): The URL for distributed training.
        deterministic (bool): Whether to set deterministic flags for SGLang.
    """
    num_gpus = torch.cuda.device_count()
    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dist-init-addr", dist_url,
        "--base-gpu-id", str(num_gpus - tp), # Reserve highest ID GPUs for SGLang
        "--tp", str(tp),
        "--grammar-backend", "xgrammar", # Better support for multi-node jobs
    ]
    if deterministic:
        # https://docs.sglang.ai/references/faq.html
        command += ["--disable-radix-cache", "--max-running-request", "1"]
    print(command)
    # Start the process in a new session
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # Start a new session
    )

    # Ensure the process group is terminated when the script exits
    def cleanup():
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            # Process has already been terminated
            return
    atexit.register(cleanup)

    # Wait for the server to start
    base_url = f"http://0.0.0.0:{port}"
    print(f"Starting SGLang server on {base_url}")
    sgl.utils.wait_for_server(base_url)
    print(f"SGLang server has started on {base_url}")

    return process, f"{base_url}/v1"

def generate_code(messages, model_path="meta-llama/Llama-3.1-8B-Instruct", tokenizer=None, temperature=0.7, top_p=1.0, base_url=None):
    """
    Generates code using a local model or SGLang through the OpenAI-like API.

    Args:
        messages (List[Dict[str, str]]): The conversation messages.
        model_path (str or AutoModelForCausalLM): The model to use for code generation.
        tokenizer (AutoTokenizer): The tokenizer to use for the model. This is only used if model_path is an AutoModelForCausalLM.
        temperature (float): The sampling temperature for the model.
        top_p (float): The nucleus sampling parameter.
        base_url (str): The base URL for the SGLang API.
    
    Returns:
        str: The generated code.
    """
    if base_url is None:
        # Model and tokenizer passed directly.
        assert type(model_path) != str, "Local model must be an AutoModelForCausalLM if base_url is not provided."
        assert tokenizer is not None, "Tokenizer must be provided if model_path is an AutoModelForCausalLM"
        with torch.no_grad():
            model = model_path
            templated_messages = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(templated_messages, return_tensors="pt", padding=True, truncation=True)
            accelerator = Accelerator()
            input_ids, attention_mask = inputs["input_ids"].to(accelerator.device), inputs["attention_mask"].to(accelerator.device)
            outputs = model.generate(input_ids, 
                attention_mask=attention_mask,
                temperature=temperature,
                max_new_tokens=1000,
                synced_gpus=True # Necessary to avoid deadlocking w/ DeepSpeed inference
            )
            output_len = len(outputs[0]) - len(inputs["input_ids"][0])
            code = tokenizer.decode(outputs[0][-output_len:], skip_special_tokens=True)
            # Remove assistant tag from the beginning of the code string w/ regex
            code = re.sub(r"^assistant(\n)*", "", code)
    else:
        # SGLang API
        client = openai.Client(base_url=base_url, api_key="None") # SGLang
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=1000,
        )
        code = response.choices[0].message.content
    return code

def create_unpaired_preference_dataset(data_ids, conversations, labels, feedbacks):
    """
    Converts conversations (lists of messages) into a HuggingFace dataset of unpaired preference data.
    
    This function generates a dataset with len(conversations) * N datapoints, where N is the number 
    of assistant messages in all conversations.

    In addition, the data IDs are included in the dataset so that Gym environments can be easily
    created for training or evaluation.

    Args:
        data_ids (List[int]): The data IDs for the conversations.
        conversations (List[List[Dict[str, str]]]): The conversations to convert.
        labels (List[bool]): The labels for the conversations.
        feedbacks (List[Dict]): Feedback on public and private tests
    
    Returns:
        dataset (Dataset): A Hugging Face dataset formatted with the conversation, completion, and label.
    """
    data = {"data_id": [], "prompt": [], "completion": [], "label": [], "feedbacks": []}
    for data_id, conversation, label, feedback in zip(data_ids, conversations, labels, feedbacks):
        assert conversation[-2]["role"] == "assistant", "The last message in the conversation must be from the assistant."
        prompt = conversation[:-2] # Collect past interactions and feedback before the assistant's message
        completion = [conversation[-2]]
        data["data_id"].append(data_id)
        data["prompt"].append(prompt)
        data["completion"].append(completion)
        data["label"].append(label)
        data['feedbacks'].append(feedback)
    dataset = Dataset.from_dict(data)
    return dataset

def load_or_create_dataset(dataset_path):
    """
    Loads a dataset from disk if it exists, otherwise creates an empty dataset.

    Args:
        dataset_path (str): The path to the dataset.
    
    Returns:
        dataset (Dataset): The loaded or created dataset.
    """
    if os.path.exists(dataset_path):
        dataset = Dataset.load_from_disk(dataset_path)
    else:
        os.makedirs(dataset_path, exist_ok=True)
        dataset = create_unpaired_preference_dataset([], [], [], [])
        dataset.save_to_disk(dataset_path) 
    return dataset

def generate_rollout(data_id, optimal=False, **kwargs):
    """
    Generates a rollout for a given data ID.

    Args:
        data_id (int): The data ID to generate a rollout for.
        optimal (bool): Whether to generate an optimal rollout.
        kwargs (dict): Keyword arguments of the following (relevant to this function):
            - model (str): The model to use for code generation.
            - temperature (float): The sampling temperature for the model.
            - base_url (str): The base URL for the SGLang API.
            - dataset (str): The dataset to rollout on.
            - split (str): The split to use for the dataset.
            - max_steps (int): The maximum number of steps to run.
            - debug (bool): Print debug information.
    
    Returns:
        messages (List[Dict[str, str]]): The generated conversation messages.
        label (bool): The label for the conversation.
    
    Raises:
        ValueError: If the dataset is not supported.
    """
    # Initialize the environment
    if kwargs.get("dataset") == "mbpp":
        env = gym.make('MBPPWrapperEnv-v0', task_id=data_id, split=kwargs.get("split"), max_steps=kwargs.get("max_steps"))
    elif kwargs.get("dataset") == "mbpp_train":
        env = gym.make('MBPPTrainWrapperEnv-v0', task_id=data_id, split=kwargs.get("split"), max_steps=kwargs.get("max_steps"))
    elif kwargs.get("dataset") == "deepmind/code_contests":
        env = gym.make('CodeContestsWrapperEnv-v0', name=data_id, split=kwargs.get("split"), max_steps=kwargs.get("max_steps"))
    elif "bigcode/humanevalpack" == kwargs.get('dataset'):
        env = gym.make('HumanEvalPackWrapperEnv-v0', data_id=data_id, max_steps=kwargs.get("max_steps"))
    elif "evalplus/humanevalplus" == kwargs.get('dataset'):
        env = gym.make('HumanEvalPlusWrapperEnv-v0', data_id=data_id, max_steps=kwargs.get("max_steps"))
    elif "evalplus/mbppplus" == kwargs.get('dataset'):
        env = gym.make('MBPPPlusWrapperEnv-v0', task_id=data_id, split=kwargs.get("split"), max_steps=kwargs.get("max_steps"))
    else:
        raise ValueError(f"Unsupported dataset {kwargs.get('dataset')}")
    obs, _ = env.reset()

    # Setup the conversation
    prompt = obs["prompt"]
    if kwargs.get("messages"):
        messages = kwargs.get("messages")
    else:
        messages = [{"role": "user", "content": prompt}]

    feedbacks = []
    if optimal:
        wrapper = env.env.env
        if kwargs.get("dataset") in ["bigcode/humanevalpack", "evalplus/humanevalplus"]:
            code = f'{wrapper.datapoint["prompt"]}\n{wrapper.datapoint["canonical_solution"]}'
        elif kwargs.get("dataset") == "mbpp" or kwargs.get("dataset") == "mbpp_train" or kwargs.get("dataset") == "evalplus/mbppplus":
            code = wrapper.datapoint["code"]
        elif kwargs.get("dataset") == "deepmind/code_contests":
            # Partially supported; not all Code Contest problems have Python solutions.
            solutions = wrapper.datapoint["solutions"]
            assert 1 in solutions["language"], "No Python solution found."
            index = solutions["language"].index(1)
            code = solutions["code"][index]
        else:
            raise ValueError(f"Unsupported dataset {kwargs.get('dataset')}")
        messages.append({"role": "assistant", "content": code})
        feedbacks.append({"public": 'Code passed all tests', "private": 'Code passed all tests'})
        return messages, True, feedbacks
    
    for _ in range(kwargs.get("max_steps")):
        # Generate code and step

        if kwargs.get("debug", False):
            # User prompt + feedback for steps range(2, max_steps)
            print(f"PROMPT:\n\n{messages[-1]['content']}\n\n")

        action = generate_code(
            messages,
            model_path=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            base_url=kwargs.get("base_url")
        )

        if kwargs.get("debug", False):
            # Assistant response
            print(f"RESPONSE:\n\n{action}\n\n")

        messages.append({"role": "assistant", "content": action})
        obs, reward, done, _, info = env.step(action)

        # Add feedback to chat history
        feedback = obs["feedback"]
        messages.append({"role": "user", "content": f"Feedback: {feedback['public']}"})
        feedbacks.append(feedback)
        if done:
            break
    
    if kwargs.get("debug"):
        # Final user prompt + feedback
        print(f"PROMPT:\n\n{messages[-1]['content']}\n\n")
    
    env.close()
    return messages, info["success"], feedbacks

GLOBAL_EXECUTOR = None
def fetch_executor(max_workers=os.cpu_count()):
    """Fetches a global ProcessPoolExecutor with the maximum number of workers.
    
    Args:
        max_workers (int): The maximum number of workers to use.
    """
    global GLOBAL_EXECUTOR
    if GLOBAL_EXECUTOR is None:
        GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
    return GLOBAL_EXECUTOR

def safe_generate_rollout(data_id, optimal, message, **kwargs):
    try:
        new_kwargs = kwargs.copy()
        new_kwargs["messages"] = message # Avoid modifying the original kwargs
        response = generate_rollout(data_id, optimal, **new_kwargs)
        if response is not None:
            return response
        raise RuntimeError("Rollout failed")
    except Exception as e:
        traceback.print_exc() # Print actual error
        raise e

async def parallel_runner(executor, data_ids, rollouts_tqdm, **kwargs):
    """
    Runs the rollouts in parallel using asyncio.

    Args:
        executor (ProcessPoolExecutor): The executor to run the rollouts on.
        data_ids (List[int]): The data IDs to run rollouts on.
        rollouts_tqdm (tqdm): The tqdm object to update.
        kwargs (dict): Keyword arguments to pass to the rollout function.
    
    Returns:
        conversations (List[List[Dict[str, str]]]): The generated conversations.
        labels (List[bool]): The labels for the conversations
    """
    # Setup the futures to run in parallel
    loop = asyncio.get_event_loop()
    if kwargs.get("messages_dataset_path"):
        dataset_messages = kwargs.pop("messages")
    else:
        dataset_messages = [[] for _ in data_ids]

    partial_generate_rollout = functools.partial(safe_generate_rollout, **kwargs) # Necessary for passing kwargs
    optimal = rollouts_tqdm.n == rollouts_tqdm.total - 1
    async def exception_generate_rollout(data_id, optimal, message, stop_flag):
        try:
            return await loop.run_in_executor(executor, partial_generate_rollout, data_id, optimal, message)
        except Exception as e:
            stop_flag.set() # Stop all other tasks
            raise e
    stop_flag = asyncio.Event()
    futures = [asyncio.create_task(exception_generate_rollout(data_id, optimal, dataset_messages[idx], stop_flag)) for idx, data_id in enumerate(data_ids)]
    run_tasks_tqdm = tqdm(total=len(data_ids), desc="Running tasks", leave=False, position=3)
    try:
        for completed_future in asyncio.as_completed(futures):
            if stop_flag.is_set():
                break
            await completed_future
            run_tasks_tqdm.update(1)
    except Exception as e:
        print("\nCancelling all tasks\n")
        for future in futures:
            future.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        executor.shutdown(wait=True)
        raise RuntimeError(f"One or more tasks failed. Traceback has been printed above.")

    
    # Collect the results
    conversations = []
    labels = []
    feedbacks = []
    collect_results_tqdm = tqdm(total=len(futures), desc="Collecting results", leave=False, position=3)
    for i, future in enumerate(futures):
        messages, label, feedback = future.result()
        conversations.append(messages)
        labels.append(label)
        feedbacks.append(feedback)
        collect_results_tqdm.update(1)
    return conversations, labels, feedbacks

def generate_rollouts(**kwargs):
    """
    Generates rollouts for a given dataset and saves the results to disk.

    This function saves the rollouts to disk in unpaired preference format. Each
    new rollout will be immediately appended to the existing dataset (created if 
    it does not exist).

    Args:
        kwargs (dict): Keyword arguments of the following (relevant to this function):
            - dataset (str): The dataset to rollout on.
            - split (str): The split to use for the dataset.
            - save_path (str): The path to save the dataset.
            - rollouts (int): The number of rollouts to run.
            - max_steps (int): The maximum number of steps to run.
            - parallel (bool): Run the rollouts in parallel.
    
    Side Effects:
        - Saves the generated rollouts to disk.
    
    Raises:
        ValueError: If the dataset is not supported.
    """
    if kwargs.get("messages_dataset_path"):
        messages_dataset = Dataset.load_from_disk(kwargs.get("messages_dataset_path"))
        feedback_msg = lambda x: {"role": "user", "content": f"Feedback: {x}"}
        dataset_messages = []
        for prompt, completion, feedback in zip(messages_dataset["prompt"], messages_dataset["completion"], messages_dataset["feedbacks"]):
            public_feedback_str = feedback[-1]["public"]
            dataset_messages.append(prompt + completion + [feedback_msg(public_feedback_str)])
        if "_train" in kwargs.get("dataset"):
            dataset = load_dataset(kwargs.get('dataset').replace("_train", ""), split=kwargs.get("split")) # Hack for train
        else:
            dataset = load_dataset(kwargs.get('dataset'), split=kwargs.get("split"))
        # Create a list of indices for the rows where data_id
        assert kwargs.get("dataset") in ["mbpp", "mbpp_train", "bigcode/humanevalpack", "evalplus/humanevalplus", "evalplus/mbppplus", "deepmind/code_contests"], f"{kwargs.get('dataset')} not supported"
        data_ids = messages_dataset["data_id"]
    elif kwargs.get("dataset") in ["mbpp", "bigcode/humanevalpack", "evalplus/humanevalplus", "evalplus/mbppplus"]:
        dataset = load_dataset(kwargs.get('dataset'), split=kwargs.get("split"))
        data_ids = range(len(dataset))
    elif "_train" in kwargs.get("dataset"):
        dataset = load_dataset(kwargs.get('dataset').replace("_train", ""), split=kwargs.get("split")) # Hack for train
        data_ids = range(len(dataset))
    elif kwargs.get("dataset") == "deepmind/code_contests":
        filtered_dataset = CodeContestsWrapperEnv.fetch_data(kwargs.get("split"))
        data_ids = range(len(filtered_dataset))
    else:
        raise ValueError(f"Unsupported dataset {kwargs.get('dataset')}")
    
    # Continue from the last saved dataset (if any)
    current_dataset = load_or_create_dataset(args.save_path)
    curr_num_rollouts = len(current_dataset) // len(data_ids)
    if curr_num_rollouts >= kwargs.get("rollouts"):
        print(f"Already generated {curr_num_rollouts} rollouts of {len(current_dataset)} datapoints.")
        return
    assert (len(current_dataset) % len(data_ids)) % kwargs.get("rollout_batch") == 0, "Rollout batch size does not match with current dataset rollouts"
    curr_num_rollout_batches = (len(current_dataset) % len(data_ids)) // kwargs.get("rollout_batch")
    if curr_num_rollout_batches > 0 or curr_num_rollouts > 0:
        print(f"Continuing from {curr_num_rollouts} / {kwargs.get('rollouts')} rollouts and {curr_num_rollout_batches} / {len(data_ids) // kwargs.get('rollout_batch') + 1} batches with {len(current_dataset)} datapoints.")
    
    # Start server if not already running
    process = None
    if kwargs.get("base_url") is None:
        # Run the SGLang server
        model_path = kwargs.get("model")
        port = kwargs.get("port")
        tp = kwargs.get("tp") # Tensor parallelism
        dist_url = kwargs.get("dist_url")
        deterministic = kwargs.get("deterministic")
        process, base_url = start_sglang_server(model_path, port, tp, dist_url, deterministic)
        kwargs["base_url"] = base_url
    
    # Rollout
    rollouts_tqdm = tqdm(range(kwargs.get("rollouts") - curr_num_rollouts + 1), desc="Running rollouts", position=1)
    for i in rollouts_tqdm:
        rollouts_batch_range = range(curr_num_rollout_batches * kwargs.get("rollout_batch"), len(data_ids), kwargs.get("rollout_batch"))
        for batch_id in tqdm(rollouts_batch_range, desc="Running rollouts batch", leave=False, position=2):
            data_ids_batch = data_ids[batch_id:batch_id + kwargs.get("rollout_batch")]
            if kwargs.get("messages_dataset_path"):
                dataset_messages_batch = dataset_messages[batch_id:batch_id + kwargs.get("rollout_batch")]
            if rollouts_tqdm.n == rollouts_tqdm.total - 1 and not kwargs.get("add_correct"):
                # Ignore the last iteration if we don't need to add a correct rollout
                break
            if kwargs.get("parallel"):
                potential_workers = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", os.cpu_count()))
                workers = min(len(data_ids), potential_workers)
                max_workers = max(1, workers)
                executor = fetch_executor(max_workers)
                if kwargs.get("messages_dataset_path"):
                    kwargs["messages"] = dataset_messages_batch
                conversations, labels, feedbacks = asyncio.run(parallel_runner(executor, data_ids_batch, rollouts_tqdm, **kwargs))
            else:
                conversations = []
                labels = []
                feedbacks = []
                for j, data_id in tqdm(enumerate(data_ids_batch), total=len(data_ids_batch), desc="Running tasks", leave=False, position=3):
                    if kwargs.get("messages_dataset_path"):
                        kwargs["messages"] = dataset_messages_batch[j]
                    if j == rollouts_tqdm.total - 1:
                        # Add a correct rollout to the dataset to ensure 1 positive example
                        messages, label, feedback = generate_rollout(data_id, solution=True, **kwargs)
                    else:
                        messages, label, feedback = generate_rollout(data_id, **kwargs)
                    conversations.append(messages)
                    labels.append(label)
                    feedbacks.append(feedback)
                    rollouts_tqdm.set_description(f"Completed {j + 1}/{len(data_ids_batch)} data_ids ({data_ids_batch})")

            # Concatenate the datasets
            current_dataset = load_or_create_dataset(args.save_path)
            new_dataset = create_unpaired_preference_dataset(data_ids_batch, conversations, labels, feedbacks)
            dataset_to_save = Dataset.from_dict({k: current_dataset[k] + new_dataset[k] for k in current_dataset.column_names})
            dataset_to_save.save_to_disk(args.save_path)
        curr_num_rollout_batches = 0
        gc.collect()
        torch.cuda.empty_cache() 
    
    if kwargs.get("parallel"):
        executor = fetch_executor()
        executor.shutdown(wait=True)
        print("Executor shutdown")
    
    if process is not None:
        # Clean up the SGLang server if started
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        print("SGLang server terminated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct", help="The model to use for code generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The sampling temperature for the model.")
    parser.add_argument("--top_p", type=float, default=1.0, help="The nucleus sampling parameter.")
    parser.add_argument("--base_url", type=str, help="The base URL for an existing SGLang server.")
    parser.add_argument("--port", type=int, help="The port to run a new SGLang server on.")
    parser.add_argument("--tp", type=int, default=1, help="The number of GPUs to use for tensor parallelism.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mbpp", "mbpp_train", "deepmind/code_contests", "bigcode/humanevalpack", "evalplus/humanevalplus", "evalplus/mbppplus"], help="The dataset to rollout on.")
    parser.add_argument("--messages_dataset_path", type=str, help="The dataset to use for existing messages (if continuing from a previous run).")
    parser.add_argument("--split", type=str, required=True, help="The split to use for the dataset.")
    parser.add_argument("--save_path", type=str, required=True, help="The path to save the dataset.")
    parser.add_argument("--rollouts", type=int, required=True, help="The number of rollouts to run.")
    parser.add_argument("--max_steps", type=int, default=1, help="The maximum number of steps to run.")
    parser.add_argument("--rollout_batch", type=int, default=512, help="The number of rollouts to save at a time.")
    parser.add_argument("--add_correct", action="store_true", help="Add a correct rollout to the dataset to ensure 1 positive example.")
    parser.add_argument("--parallel", action="store_true", help="Run the rollouts in parallel.")
    parser.add_argument("--debug", action="store_true", help="Print debug information.")
    parser.add_argument("--dist_url", type=str, default="localhost:29500", help="The URL for distributed training.")
    parser.add_argument("--deterministic", action="store_true", help="Set deterministic flags for SGLang.")
    args = parser.parse_args()

    assert args.base_url is not None or args.port is not None, "Please specify either a base URL to an existing SGLang server or a port to run a new SGLang server on."
    assert not (args.base_url is not None and args.port is not None), "You cannot specify both a base URL and a port. Specify one or the other."
    kwargs = vars(args)

    if kwargs.get("debug"):
        # Measure performance
        start = time.perf_counter()

    generate_rollouts(**kwargs)

    if kwargs.get("debug"):
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
