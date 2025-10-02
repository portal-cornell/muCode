from src.env.multistep_code_env import MultistepCodeEnv
from datasets import load_dataset
from .commit0_utils import generate_prompt, generate_testless_prompt, extract_code_blocks

import builtins
import multiprocessing
import re
import io
import queue
import subprocess
import signal
import tempfile
import uuid
import os

from .utils import subprocess_timeout_exec

# Sandbox configuration
DISALLOWED_BUILTINS = ['open', 'input', 'exec']
DISALLOWED_LIBRARIES = [
    'subprocess', 'shutil', 'socket', 'ctypes', 'multiprocessing',
    'threading', 'pickle', 'popen', 'boto3', 'paramiko', 'ftplib', 'requests', 
    'smtplib', 'pyautogui', 'win32api', 'pywin32'
]
# from https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/HumanEval/human_eval/evaluation.py
ALLOWED_IMPORTS = ['import math', 'import re', 'import copy', 'import datetime', 'import itertools', 'import collections', 'import heapq', 'import functools', 'import hashlib', 'import numpy', 'import numpy as np', 'import string', 'from typing import *', 'from collections import *', 'import os', 'import sys']
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in DISALLOWED_LIBRARIES:
        raise ImportError(f"Importing '{name}' is not allowed.")
    return builtins.__import__(name, globals, locals, fromlist, level)

MBPP_DATA = {
    "train": load_dataset("mbpp", split="train"),
    "validation": load_dataset("mbpp", split="validation"),
    "test": load_dataset("mbpp", split="test")
}

class MBPPTrainWrapperEnv(MultistepCodeEnv):

    def __init__(self, *args, task_id=None, split=None, testless=False, **kwargs):
        """
        Initializes the MBPPTrainWrapperEnv.

        This environment has a specific public/private test split used for training.

        Args:
            task_id (int): The ID of the MBPP data to use.
            split (str): The split of the data to use. Either 'train', 'validation', or 'test'.
            testless (bool): If True, uses a prompt without tests
        """
        assert split in ['train', 'validation', 'test'], "split must be either 'train', 'validation', or 'test'."
        assert task_id is not None, "task_id must be provided for MBPPWrapperEnv."
        self.datapoint = MBPP_DATA[split][task_id]
        prompt = self.datapoint['text']
        test_setup = self.datapoint['test_setup_code']
        test_code = "\n".join(self.datapoint['test_list'])
        self.eval_code = f"{test_setup}\n{test_code}"
        self.private_eval_code = f"{test_setup}\n{self.datapoint['challenge_test_list']}" if len(self.datapoint['challenge_test_list']) else None
        if testless:
            pattern = r"(def\s+.*?:)"
            function_def = [defn for defn in re.findall(pattern, self.datapoint['code'], re.DOTALL) if "__init__" not in defn][0]
            initial_prompt = generate_testless_prompt(prompt, function_def)
        else:
            initial_prompt = generate_prompt(prompt, self.eval_code)
        super().__init__(*args, **kwargs, initial_prompt=initial_prompt)
        self.task_id = task_id
        self.split = split
    
    @classmethod
    def generate_restricted_globals(cls):
        """
        Generates a restricted globals dictionary for sandboxing.

        This provides general protection against imports and builtins that could be used
        for malicious purposes but is not exhaustive. Use with caution.

        Returns:
            dict: A restricted globals dictionary.
        """
        restricted_globals = globals().copy()
        restricted_globals['__builtins__'] = builtins.__dict__.copy()
        restricted_globals['__builtins__']['__name__'] = None # Avoid __name__ == '__main__' runs
        restricted_globals['__builtins__']['__import__'] = restricted_import
        def create_raise_error(builtin_name):
            def raise_error(*args, **kwargs):
                    raise AssertionError(f"{builtin_name}() is not allowed.")
            return raise_error
        for disallowed_builtin in DISALLOWED_BUILTINS:
            restricted_globals['__builtins__'][disallowed_builtin] = create_raise_error(disallowed_builtin)
        return restricted_globals
        
    def _local_evaluate_code(self, code):
        """
        Evaluates code via execution via subprocess; there is no sandboxing

        Args:
            code (string): The code to evaluate.
        
        Returns:
            tuple: A tuple (feedback, reward, success)
                feedback is in the form {"public": <public feedback>, "private": <private feedback>}; private feedback may be None if there was no private test
                reward is 1 if both public and private tests pass, else 0
                success is whether or not both public and private tests passed
        """
        matches = extract_code_blocks(code)
        if len(matches) > 0:
            formatted_code = matches[0]
        else:
            pattern = r"```python\n(.*?)"
            matches = re.finditer(pattern, code, re.DOTALL)
            matches = [match.group(1).strip() for match in matches]
            if len(matches) > 0:
                formatted_code = matches[0]
            else:
                formatted_code = code
        
        generic_success_msg = 'Code passed all tests'
        feedbacks = {}
        any_failed = False
        print_msg = f"Code passed all tests {hash(uuid.uuid4())}"
        for eval_code, public_private in zip([self.eval_code, self.private_eval_code], ['public' ,'private']):
            if eval_code:
                exec_code = '\n'.join(ALLOWED_IMPORTS) + f"\n{formatted_code}\n{eval_code}\nprint('{print_msg}')"
                errs, outs = subprocess_timeout_exec(exec_code)
                if not print_msg in outs:
                    feedbacks[public_private] = errs
                    any_failed = True
                else:
                    feedbacks[public_private] = generic_success_msg
            else:
                feedbacks[public_private] = ""
        
        success = not any_failed
        reward = int(success)
        return feedbacks, reward, success