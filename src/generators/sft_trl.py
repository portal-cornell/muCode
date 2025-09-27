from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, TrainerCallback, AutoModelForCausalLM
from accelerate import Accelerator

import multiprocessing
import os
import torch

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)
from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
)

from peft import PeftModel

# TODO(chalo2000): Move functions from script to library file
from src.common.generate_rollouts_script import generate_rollout, start_sglang_server
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

class RolloutEvaluationCallback(TrainerCallback):
    def __init__(self, eval_fn, eval_epoch_interval, eval_dir, rollout_ids, kwargs):
        super().__init__()
        self.eval_fn = generate_rollout
        self.eval_epoch_interval = eval_epoch_interval
        self.eval_dir = eval_dir
        self.rollout_ids = rollout_ids
        self.eval_kwargs = kwargs
        
        self.current_step = 1

    def on_step_end(self, args, state, control, **kwargs):
        accelerator = Accelerator()
        epoch_threshold = self.eval_epoch_interval * self.current_step
        if epoch_threshold - state.epoch < 1e-6 and accelerator.is_main_process:
            print(f"Evaluating model at epoch {state.epoch:.2f} (step {state.global_step})")                

            # Save current model and adapters as checkpoint
            print("Saving current model and tokenizer")
            model_dir = f"{self.eval_dir}/checkpoint-{state.global_step}"
            os.makedirs(model_dir, exist_ok=True)
            model = kwargs["model"]
            model.save_pretrained(model_dir)
            model.get_base_model().save_pretrained(model_dir)
            tokenizer = kwargs["processing_class"]
            tokenizer.save_pretrained(model_dir)
            
            # Load model to merge and save without DeepSpeed
            print("Merging model and saving")
            new_base_model = AutoModelForCausalLM.from_pretrained(model_dir)
            new_peft_model = PeftModel.from_pretrained(new_base_model, model_dir)
            merged_model = new_peft_model.merge_and_unload()
            merged_model._hf_peft_config_loaded = False # https://github.com/huggingface/transformers/issues/26972
            merged_model_dir = f"{self.eval_dir}/merged_checkpoint-{state.global_step}"
            self.eval_kwargs["model"] = merged_model_dir
            merged_model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
        
            # Add current model and tokenizer to kwargs
            port = self.eval_kwargs["port"]
            tp = self.eval_kwargs["tp"]
            dist_addr = os.getenv("MASTER_ADDR", "localhost")
            dist_port = os.getenv("MASTER_PORT", "29500")
            dist_url = f"{dist_addr}:{dist_port}"
            process, base_url = start_sglang_server(merged_model_dir, port, tp, dist_url=dist_url)
            self.eval_kwargs["base_url"] = base_url

            model.eval()
            table = {"data_id": [], "prompt": [], "completion": [], "label": []}
            for data_id in tqdm(self.rollout_ids):
                table["data_id"].append(data_id)
                messages, label = self.eval_fn(data_id, **self.eval_kwargs)
                for i, message in enumerate(messages):
                    if message["role"] == "assistant":
                        prompt = messages[:i]
                        completion = [message]
                        table["prompt"].append(prompt)
                        table["completion"].append(completion)
                table["label"].append(label)
            # Convert table to a Dataset
            dataset = Dataset.from_dict(table)
            # Save dataset to disk
            os.makedirs(self.eval_dir, exist_ok=True)
            dataset.save_to_disk(f"{self.eval_dir}/evaluation-{state.global_step}")
            model.train()
            self.current_step += 1
            process.terminate()
        accelerator.wait_for_everyone()
        return control


def pad_dataset(dataset, batch_size):
    padding_amount = batch_size - len(dataset) % batch_size
    padding_indices = torch.randint(len(dataset), (padding_amount,))
    padding_dataset = dataset.select(padding_indices)
    return concatenate_datasets([dataset, padding_dataset])


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    accelerator = Accelerator()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = DatasetDict({f"{args.dataset_test_split}": Dataset.load_from_disk(f'{args.dataset_name}/{args.dataset_test_split}'),
                           f"{args.dataset_train_split}": Dataset.load_from_disk(f'{args.dataset_name}/{args.dataset_train_split}')})

    completion_token = "<|response|>"

    def process(row):
        row["completion"][0]["content"] = f"{completion_token} {row['completion'][0]['content']}"
        row["text"] = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=False
        )
        return row

    train_dataset = dataset[args.dataset_train_split].map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    # Pad the dataset to the nearest multiple of the batch size
    batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    train_dataset = pad_dataset(train_dataset, batch_size)
    assert len(train_dataset) % batch_size == 0

    eval_dataset = dataset[args.dataset_test_split].map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    eval_dataset = pad_dataset(eval_dataset, batch_size)
    assert len(eval_dataset) % batch_size == 0
    collator = DataCollatorForCompletionOnlyLM(
        completion_token,
        tokenizer=tokenizer,
        return_tensors="pt")

    ###################
    # Training Callback
    ###################

    eval_epoch_interval = 0.25 # TODO(chalo2000): Pass this as an argument
    eval_dir = f"{training_args.output_dir}/evaluation"
    unique_test_ids = list(set(eval_dataset["data_id"]))
    rollout_kwargs = { # TODO(chalo2000): Pass these as an argument
            "eval_epoch_interval": 0.25,
            "eval_dir": eval_dir,
            "temperature": 0.7,
            "dataset": "mbpp",
            "split": args.dataset_test_split.split(os.sep)[0],
            "max_steps": 1,
            "port": 34567,
            "tp": 1,
        }
    training_callback = RolloutEvaluationCallback(generate_rollout, eval_epoch_interval, eval_dir, unique_test_ids, rollout_kwargs)

    ################
    # Training
    ################

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
        data_collator=collator,
        # callbacks=[training_callback],
    )

    trainer.train()
