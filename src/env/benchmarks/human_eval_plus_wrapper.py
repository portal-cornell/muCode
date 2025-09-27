from src.env.multistep_code_env import MultistepCodeEnv
from datasets import load_dataset
from .commit0_utils import generate_prompt, extract_code_blocks

import builtins
import multiprocessing
import re
import queue
import os
import uuid

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

HUMANEVALPACK_DATA = load_dataset("bigcode/humanevalpack", "python", trust_remote_code=True)["test"]
HUMANEVALPLUS_DATA = load_dataset("evalplus/humanevalplus", trust_remote_code=True)["test"]

class HumanEvalPlusWrapperEnv(MultistepCodeEnv):

    def __init__(self, *args, data_id=None, **kwargs):
        """
        Initializes the HumanEvalPlusWrapperEnv.

        Args:
            data_id (str): The ID of the HumanEval data to use.
        """
        assert data_id is not None, "data_id must be provided for HumanEvalPlusWrapperEnv."
        self.datapoint = HUMANEVALPACK_DATA[data_id]
        self.datapoint["test"] = HUMANEVALPLUS_DATA[data_id]["test"] # Replace HumanEvalPack private tests with HumanEval+ tests
        initial_prompt = generate_prompt(self.datapoint['prompt'], self.datapoint['example_test'])
        super().__init__(*args, **kwargs, initial_prompt=initial_prompt)

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
    
    def extract_public_private_tests(self):
        """
        Extracts the public and private tests from the datapoint.

        Returns:
            tuple: A tuple (public_tests, private_tests)
        """
        public_tests = self.datapoint['example_test'].strip()
        private_tests = self.datapoint['test'].strip()
        
        assert_pattern = re.compile(r'assert .+?(?=\n)')

        matches = assert_pattern.findall(public_tests)
        if len(matches) == 0:
            return public_tests, private_tests

        first_assert = matches[min(1, len(matches) - 1)] # Get second assert statement since first is usually debug assert True
        public_tests_without_first = re.sub(re.escape(first_assert), '', public_tests, count=1).strip()

        private_tests = f"{private_tests}\ncheck({self.datapoint['entry_point']})\n{public_tests_without_first}"
        return public_tests, private_tests
    
    def _local_evaluate_code(self, code):
        """
        Evaluates code via execution via subprocess; there is no sandboxing

        Args:
            code (string): The code to evaluate.
        
        Returns:
            tuple: A tuple (feedback, reward, success)
                feedback is in the form {"public": <public feedback>, "private": <private feedback>}
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
        prompt = self.datapoint['prompt']

        generic_success_msg = 'Code passed all tests'
        feedbacks = {}
        any_failed = False
        print_msg = f"Code passed all tests {hash(uuid.uuid4())}"
        public_tests, private_tests = self.extract_public_private_tests()
        for test_code, public_private in zip([public_tests, private_tests], ['public', 'private']):
            exec_code = '\n'.join(ALLOWED_IMPORTS) + '\n' + prompt[: prompt.index(f"def {self.datapoint['entry_point']}")] + "\n\n" + formatted_code + f"\n{test_code}\nprint('{print_msg}')"
            errs, outs = subprocess_timeout_exec(exec_code)
            if not print_msg in outs:
                feedbacks[public_private] = errs
                any_failed = True
            else:
                feedbacks[public_private] = generic_success_msg
        success = not any_failed
        reward = int(success)
        return feedbacks, reward, success