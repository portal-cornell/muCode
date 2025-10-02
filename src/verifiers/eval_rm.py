import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple

from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
    PreTrainedTokenizer
)

from peft import LoraConfig, TaskType

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    DatasetConfig,
)

from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    get_reward,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    save_with_accelerate,
)

from open_instruct.utils import (
    ArgumentParserPlus,
    get_wandb_tags,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

from src.common.utils import (
    BinaryDatasetProcessor,
    INPUT_IDS_PROMPT_KEY,
    LABEL_KEY,
    PreferenceDatasetProcessor,
    SimpleBinaryCollator,
    PROMPT_KEY,
    COMPLETION_KEY,
    INPUT_IDS_CHOSEN_KEY,
    INPUT_IDS_REJECTED_KEY
)

from src.common.create_viz_script import create_viz
from src.env.benchmarks.commit0_utils import extract_code_blocks

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    DatasetConfig,
    SimplePreferenceCollator,
)


@dataclass
class Args:
    dataset_path: str = None
    """Path to the dataset for evaluation"""
    pref_dataset: bool = False
    """If the dataset for evaluation is a preference dataset"""
    remove_canonical: bool = False
    """If we should remove the GT solution from the rollouts."""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""

    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "LLM_RM"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""

    resize_token_embeddings: bool = False
    """Whether to resize the token embeddings to a factor of 8 for utilizing tensor cores better"""
    code_parsing: bool = False
    """[Special] Whether to parse the code before training the model"""


def find_shared_text(chosen_text: str, rejected_text: str):
    """return shared (prompt) text between chosen and rejected text"""
    for i in range(min(len(chosen_text), len(rejected_text))):
        if chosen_text[i] != rejected_text[i]:
            break

    return chosen_text[:i]


def evaluate_bin(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = -1
) -> Tuple[dict, dict]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_batches = 0
    table = defaultdict(list)

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data[INPUT_IDS_PROMPT_KEY]
            labels = data[LABEL_KEY]
            prompts = data[PROMPT_KEY]
            completions = data[COMPLETION_KEY]
            _, predicted_reward, _ = get_reward(model, queries, tokenizer.pad_token_id, 0)

            accuracy = ((predicted_reward > 0) == (labels > .5)).float().mean()
            loss = criterion(predicted_reward, labels)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_batches += 1

            table[PROMPT_KEY].extend(prompts)
            table[COMPLETION_KEY].extend(completions)
            table["prompt_completion"].extend(tokenizer.batch_decode(queries))
            table["label"].extend(labels.tolist())
            table["reward"].extend(predicted_reward.tolist())


            if "data_id" in data:
                table["data_id"].extend(data["data_id"])
            if "ID" in data:
                table["ID"].extend(data["ID"])
            
            # if "feedbacks" in data:
            table["feedbacks"].extend(data["feedbacks"])

            if num_samples > 0 and len(table) > num_samples:
                break

    model.train()
    return {
        "eval/rm/accuracy": total_accuracy / total_batches,
        "eval/rm/loss": total_loss / total_batches,
    }, table


def evaluate_pref(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = -1
) -> Tuple[dict, dict]:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    total_reward_margin = 0
    total_batches = 0
    table = None

    table = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            query_responses = torch.cat((data[INPUT_IDS_CHOSEN_KEY], data[INPUT_IDS_REJECTED_KEY]), dim=0)
            _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
            chosen_rewards = predicted_reward[: data[INPUT_IDS_CHOSEN_KEY].shape[0]]
            rejected_rewards = predicted_reward[data[INPUT_IDS_CHOSEN_KEY].shape[0] :]
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_chosen_rewards += chosen_rewards.mean().item()
            total_rejected_rewards += rejected_rewards.mean().item()
            total_reward_margin += (chosen_rewards - rejected_rewards).mean().item()
            total_batches += 1

            chosen_texts = tokenizer.batch_decode(data[INPUT_IDS_CHOSEN_KEY])
            rejected_texts = tokenizer.batch_decode(data[INPUT_IDS_REJECTED_KEY])
            # remove padding
            chosen_texts = [item.replace(tokenizer.pad_token, "") for item in chosen_texts]
            rejected_texts = [item.replace(tokenizer.pad_token, "") for item in rejected_texts]
            rewards_rounded = [
                [round(chosen.item(), 4), round(rejected.item(), 4)]
                for chosen, rejected in zip(chosen_rewards, rejected_rewards)
            ]
            correct_prediction = [
                bool((chosen > rejected)) for chosen, rejected in zip(chosen_rewards, rejected_rewards)
            ]
            shared_texts = [
                find_shared_text(chosen_text, rejected_text)
                for chosen_text, rejected_text in zip(chosen_texts, rejected_texts)
            ]
            chosen_response_texts = [
                chosen_text[len(shared_text) :] for chosen_text, shared_text in zip(chosen_texts, shared_texts)
            ]
            rejected_response_texts = [
                rejected_text[len(shared_text) :]
                for rejected_text, shared_text in zip(rejected_texts, shared_texts)
            ]
            table["shared prompt text"].extend(shared_texts)
            table["chosen response text"].extend(chosen_response_texts)
            table["rejected response text"].extend(rejected_response_texts)
            table["chosen reward, rejected reward"].extend(rewards_rounded)
            table["correct prediction"].extend(correct_prediction)

            if "ID" in data:
                table["ID"].extend(data["ID"])

            if num_samples > 0 and len(table) > num_samples:
                break

    model.train()
    return {
        "eval/rm/accuracy": total_accuracy / total_batches,
        "eval/rm/loss": total_loss / total_batches,
        "eval/rm/chosen_rewards": total_chosen_rewards / total_batches,
        "eval/rm/rejected_rewards": total_rejected_rewards / total_batches,
        "eval/rm/reward_margin": total_reward_margin / total_batches,
    }, table


def calculate_runtime_args_and_accelerator(args: Args, model_config: ModelConfig) -> Accelerator:
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    # set a unique run name with the current timestamp
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking and accelerator.is_main_process:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
    return accelerator


def layer_init(layer: nn.Module, std: float):
    torch.nn.init.normal_(layer.weight, std=std)
    return layer


def run_evaluation_and_plot(
    model,
    tokenizer,
    accelerator,
    dataset_config,
    args
):

    dataset = Dataset.load_from_disk(f'{args.dataset_path}')

    if args.remove_canonical:
        num_unique_prompts = len(set([x['prompt'][0]['content'] for x in dataset]))
        assert sum(dataset['label'][-num_unique_prompts:]) == num_unique_prompts
        dataset = dataset.select(range(len(dataset) - num_unique_prompts))

    new_column = range(len(dataset))
    dataset = dataset.add_column("ID", new_column)

    if args.pref_dataset:
        dataset_processor = PreferenceDatasetProcessor(
            tokenizer=tokenizer,
            config=dataset_config
        )
    else:
        dataset_processor = BinaryDatasetProcessor(
            tokenizer=tokenizer,
            config=dataset_config
        )
    with accelerator.main_process_first():
        if args.code_parsing:
            def process_example(example):
                def get_first_code_block(data):
                    assert len(data) == 1
                    blocks = extract_code_blocks(data[0]['content'])
                    data[0]['content'] = blocks[0] if blocks else ""
                    return data
                if args.pref_dataset:
                    example['chosen'] = get_first_code_block(example['chosen'])
                    example['rejected'] = get_first_code_block(example['rejected'])
                else:
                    example['completion'] = get_first_code_block(example['completion'])
                return example
            dataset = dataset.map(process_example)
        dataset = dataset_processor.tokenize(dataset)
        dataset = dataset_processor.filter(dataset)

    if args.pref_dataset:
        data_collator = SimplePreferenceCollator(
            pad_token_id=tokenizer.pad_token_id
        )
    else:
        data_collator = SimpleBinaryCollator(
            pad_token_id=tokenizer.pad_token_id
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    """
    Evaluation
    """
    if args.pref_dataset:
        eval_metrics, table = evaluate_pref(model, dataloader, tokenizer)
    else:
        eval_metrics, table = evaluate_bin(model, dataloader, tokenizer)

    for key in table:
        table[key] = gather_object(table[key])
    df = pd.DataFrame(table)
    df = df.drop_duplicates(subset='ID', keep='first')
    df.to_csv(f"{args.output_dir}/data.csv")

    if accelerator.is_main_process:
        print_rich_single_line_metrics(eval_metrics)


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    accelerator = calculate_runtime_args_and_accelerator(args, model_config)
    local_seed = args.seed + accelerator.process_index

    """
    Experiment Tracking
    """
    all_configs = {}
    all_configs.update(**asdict(args), **asdict(dataset_config), **asdict(model_config))
    if accelerator.is_main_process:
        if not os.path.isdir(f"{args.output_dir}"):
            os.makedirs(f'{args.output_dir}')

        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=all_configs,
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    """
    Seeding
    """
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    """
    Tokenizer
    """
    config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    )
    if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
        tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding

    """
    Reward Model
    """
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        num_labels=1
    )
    if model_config.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
        )
        model.add_adapter(peft_config)
    if args.resize_token_embeddings:  # optimize for tensor core
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    disable_dropout_in_model(model)  # see p.3. in https://arxiv.org/pdf/1909.08593

    """
    Prepare dataloader
    """
    run_evaluation_and_plot(
        model,
        tokenizer,
        accelerator,
        dataset_config,
        args
    )

if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
