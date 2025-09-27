import logging
from typing import Dict, List, Optional, Union
import copy 

import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from open_instruct.dataset_processor import DatasetProcessor, get_num_proc

logging.basicConfig(level=logging.INFO)


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
INPUT_IDS_PROMPT_KEY = "prompt_code"
LABEL_KEY = "label"

INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
GROUND_TRUTHS_KEY = "raw_output"
DATASET_SOURCE_KEY = "dataset"
PROMPT_KEY = 'prompt'
COMPLETION_KEY = 'completion'

# FLAGS for Preference dataset
INPUT_IDS_CHOSEN_KEY = 'input_ids_chosen'
INPUT_IDS_REJECTED_KEY = 'input_ids_rejected'

APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130


class SFTGroundTruthDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            # if len(row['prompt']) == 1:
            #     prompt = row['prompt']
            # else:
            prompt = row['prompt']

            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row['prompt'])
            # row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_KEY])] = [-100] * len(row[INPUT_IDS_KEY])
            row[LABELS_KEY] = labels

            row[GROUND_TRUTHS_KEY] = row['raw_output']
            row[DATASET_SOURCE_KEY] = 'HumanEval'
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            # if len(row['prompt']) == 1:
            #     prompt = row['prompt']
            # else:
            prompt = row['prompt']

            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row['prompt'])
            # row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_KEY])] = [-100] * len(row[INPUT_IDS_KEY])
            row[LABELS_KEY] = labels

            row[GROUND_TRUTHS_KEY] = row['raw_output']
            row[DATASET_SOURCE_KEY] = 'HumanEval'
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class BinaryDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        '''
        Converts the prompt and agent rollout to a chat template and extracts label 
        '''
        def tokenize_fn(row):
            # row[ATTENTION_MASK_KEY] = [1] * len(row['rollouts'])
            row[PROMPT_KEY] = row[PROMPT_KEY]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                row[PROMPT_KEY][:1] + row[COMPLETION_KEY]
            )
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        '''
        FIXME: Clean this function !
        '''
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class PreferenceDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        def tokenize_fn(row):

            row[INPUT_IDS_CHOSEN_KEY] = self.tokenizer.apply_chat_template(
                row[PROMPT_KEY] + row[self.config.preference_chosen_key]
            )
            row[INPUT_IDS_REJECTED_KEY] = self.tokenizer.apply_chat_template(
                row[PROMPT_KEY] + row[self.config.preference_rejected_key]
            )
            row[PROMPT_KEY] = self.tokenizer.apply_chat_template(
                row[PROMPT_KEY],
                add_generation_prompt=True,
            )
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(
                len(dataset),
                self.config.num_proc,
                APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
            ),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        def filter_fn(row):
            return (
                len(row[PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SimpleBinaryCollator:
    '''
    Copy of SimplePreferenceCollator in open_instruct and adpated to data columns we are using.
    '''
    def __init__(self, pad_token_id: int):
        """Simple collator for preference dataset (always pad from the RIGHT)"""
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, int]]):
        """the input will have input_ids_chosen, input_ids_rejected"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i]["prompt_code"]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the right
            padding = [self.pad_token_id] * pad_length
            padded_sequence = batch[i][INPUT_IDS_PROMPT_KEY] + padding
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences_chosen = torch.tensor(padded_sequences)
        rewards = torch.tensor([batch[i][LABEL_KEY] for i in range(len(batch))]).float()

        # Convert to unique prompt keys list
        prompt_keys = [batch[i][PROMPT_KEY] for i in range(len(batch))]
        completion_keys = [batch[i][COMPLETION_KEY] for i in range(len(batch))]

        # Hack for evaluation pipeline
        if "ID" in batch[i]:
            id_keys = [batch[i]["ID"] for i in range(len(batch))]
        else:
            id_keys = [0 for i in range(len(batch))]

        if "data_id" in batch[i]:
            data_id_keys = [batch[i]["data_id"] for i in range(len(batch))]
        else:
            data_id_keys = [0 for i in range(len(batch))]
        feedback_keys = [batch[i]["feedbacks"] for i in range(len(batch))]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences_chosen,
            LABEL_KEY: rewards,
            PROMPT_KEY: prompt_keys,
            COMPLETION_KEY: completion_keys,
            "ID": id_keys,
            "data_id": data_id_keys,
            "feedbacks": feedback_keys
        }


class SimplePreferenceCollator:
    def __init__(self, pad_token_id: int):
        """Simple collator for preference dataset (always pad from the RIGHT)"""
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, int]]):
        """the input will have input_ids_chosen, input_ids_rejected"""
        # Find max length in the batch
        max_length_chosen = -1
        max_length_rejected = -1
        for i in range(len(batch)):
            max_length_chosen = max(max_length_chosen, len(batch[i]["input_ids_chosen"]))
            max_length_rejected = max(max_length_rejected, len(batch[i]["input_ids_rejected"]))
        max_length = max(max_length_chosen, max_length_rejected)
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences_chosen = []
        padded_sequences_rejected = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length_chosen = max_length - len(batch[i][INPUT_IDS_CHOSEN_KEY])
            pad_length_rejected = max_length - len(batch[i][INPUT_IDS_REJECTED_KEY])

            # Pad from the right
            padding_chosen = [self.pad_token_id] * pad_length_chosen
            padding_rejected = [self.pad_token_id] * pad_length_rejected
            padded_sequence_chosen = batch[i][INPUT_IDS_CHOSEN_KEY] + padding_chosen
            padded_sequence_rejected = batch[i][INPUT_IDS_REJECTED_KEY] + padding_rejected
            padded_sequences_chosen.append(padded_sequence_chosen)
            padded_sequences_rejected.append(padded_sequence_rejected)

        # Convert to tensors
        padded_sequences_chosen = torch.tensor(padded_sequences_chosen)
        padded_sequences_rejected = torch.tensor(padded_sequences_rejected)

        # Hack for evaluation pipeline
        if "ID" in batch[i]:
            id_keys = [batch[i]["ID"] for i in range(len(batch))]
        else:
            id_keys = [0 for i in range(len(batch))]

        return {
            INPUT_IDS_CHOSEN_KEY: padded_sequences_chosen,
            INPUT_IDS_REJECTED_KEY: padded_sequences_rejected,
            "ID": id_keys
        }


class SimpleGenerateCollatorWithGroundTruth:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the left
            padding = [self.pad_token_id] * pad_length
            padded_sequence = padding + batch[i][INPUT_IDS_PROMPT_KEY]
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences)

        # ground truths
        ground_truths = [x[GROUND_TRUTHS_KEY] for x in batch]

        # datasets
        datasets = [x[DATASET_SOURCE_KEY] for x in batch]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences,
            GROUND_TRUTHS_KEY: ground_truths,
            DATASET_SOURCE_KEY: datasets,
        }