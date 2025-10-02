from src.env.multistep_code_env import MultistepCodeEnv
from datasets import load_dataset, Dataset
from .commit0_utils import generate_prompt, extract_code_blocks

import builtins
import multiprocessing
import re
import io
import sys
import queue

# Sandbox configuration
DISALLOWED_BUILTINS = ['open', 'exec']
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

def filter_missing_public(dataset):
    """
    Filters out datapoints with missing public tests.

    Args:
        dataset (list): The dataset to filter.

    Returns:
        list: The filtered dataset.
    """
    return dataset.filter(lambda x: len(x['public_tests']['input']) > 0)

class CodeContestsWrapperEnv(MultistepCodeEnv):

    def __init__(self, *args, name=None, split=None, **kwargs):
        """
        Initializes the CodeContestsWrapperEnv.

        Args:
            name (str): The name of the CodeContests data to use.
            split (str): The split of the data to use. Either 'train', 'validation', or 'test'.
        """
        # assert split in ['train', 'validation', 'test', 'train_1k', ''], "split must be either 'train', 'validation', or 'test'."
        # assert name is not None, "name must be provided for CodeContestsWrapperEnv."
        # filtered = CODE_CONTESTS_DATA[split].filter(lambda x: x['name'] == name)
        # assert len(filtered) > 0, f"Invalid name [{name}]."
        index = name # Name is actually hardcoded to be an index
        self.datapoint = CodeContestsWrapperEnv.fetch_data(split)[index]
        prompt = self.datapoint['description']
        initial_prompt = f"""
Provide a Python solution for the following competitive programming question: \${prompt}.
Your code should be enclosed in triple backticks like so: ```python YOUR CODE HERE```. Use the backticks for your code only.
"""
        super().__init__(*args, **kwargs, initial_prompt=initial_prompt)
        self.task_id = name
        self.split = split
    
    @classmethod
    def fetch_data(cls, split):
        """
        Fetches the CodeContests data for a given split.

        Args:
            split (str): The split of the data to fetch. Either 'train', 'validation', or 'test'.

        Returns:
            list: The CodeContests data for the given split.
        """
        if not hasattr(cls, 'CODE_CONTESTS_DATA'):
            cls.CODE_CONTESTS_DATA = {
                # 'train': filter_missing_public(load_dataset("deepmind/code_contests", split="train")),
                'validation': filter_missing_public(load_dataset("deepmind/code_contests", split="valid")),
                'test': filter_missing_public(load_dataset("deepmind/code_contests", split="test")),
                'train_1k': filter_missing_public(load_dataset("rl-llm-coders/cc_1k", split="train")),
                # 'train_2k': filter_missing_public(Dataset.load_from_disk("data/codecontest/data_2k/train")),
                # 'train_1500': filter_missing_public(Dataset.load_from_disk("data/codecontest/data_1500/train"))
            }
        return cls.CODE_CONTESTS_DATA[split]
    
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

    def _timeout_exec(self, func, timeout=100):
        """
        Executes code with a timeout.

        Args:
            func (function): The function to execute.
            timeout (int): The timeout in seconds.
        
        Raises:
            TimeoutError: If the code execution times out.
            Exception: If the code execution raises an exception.
        """
        def worker(func, output_queue):
            try:
                func()
                output_queue.put(None)
            except Exception as e:
                exception_info = (type(e), str(e)) # Reliably reconstruct exception
                output_queue.put(exception_info)

        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker, args=(func, output_queue))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            output_queue.close()
            output_queue.join_thread()
            raise TimeoutError("Code execution timed out")

        try:
            result = output_queue.get(timeout=timeout)
            if result is not None:
                exec_type, exec_message = result
                # Code execution error
                raise exec_type(exec_message)
        except queue.Empty:
            raise Exception("Code execution failed") # If no result, code execution failed. Prevents hanging.
        finally:
            output_queue.close()
            output_queue.join_thread()
    
    def _custom_stdin_stdout_test(self, code, custom_globals, input_output_dict, timeout=1):
        """
        Executes and evaluates code with custom stdin and stdout from input_output_dict.
        
        Allows for tests that read in input from stdin and print to stdout.

        Args:
            code (string): The code to execute.
            custom_globals (dict): The globals to execute the code in.

        Raises:
            Exception: If the code execution raises an exception.
        """
        # Redirect stdin and stdout
        def run_code(solution_code, conn, test_input):
            sys.stdin = io.StringIO(test_input)
            sys.stdout = io.StringIO()
            try:
                exec(solution_code, custom_globals)
                conn.send(sys.stdout.getvalue())
            except Exception as e:
                conn.send(e)
        parent_conn, child_conn = multiprocessing.Pipe()
        tmp_var = False

        for input_data, output_data in zip(input_output_dict["input"], input_output_dict["output"]):
            process = multiprocessing.Process(target=run_code, args=(code, child_conn, input_data))
            process.start()
            process.join(timeout)

            if process.is_alive():
                process.terminate()
                process.join()
                parent_conn.close()
                child_conn.close()
                tmp_var = True
                raise TimeoutError("Code execution timed out")

            if tmp_var:
                print ("LOL")

            actual_output = parent_conn.recv()
            if isinstance(actual_output, Exception):
                parent_conn.close()
                child_conn.close()
                raise actual_output
            output_data = output_data.strip('\n')
            actual_output = actual_output.strip('\n')
            assert actual_output == output_data, f"Expected output={output_data} for input={input_data}, but got actual_output={actual_output}"
        parent_conn.close()
        child_conn.close()
    
    def construct_tests(self, f_name, input_output_dict):
        """
        Constructs a string of tests from input_output_dict.

        Args:
            f_name (string): The name of the function to test.
            input_output_dict (dict): A dictionary of input and output pairs.

        Returns:
            string: A string of tests.
        """
        tests = []
        for input_data, output_data in zip(input_output_dict["input"], input_output_dict["output"]):
            tests.append(f"assert {f_name}({repr(input_data)}) == {repr(output_data)}")
        return '\n'.join(tests)
        
    def _local_evaluate_code(self, code):
        """
        Evaluates code with strict sandboxing using RestrictedPython.

        This function does not allow any I/O operations.

        Args:
            code (string): The code to evaluate.
        
        Returns:
            tuple: A tuple (feedback, reward, success)
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
        formatted_code = '\n'.join(ALLOWED_IMPORTS) + '\n' + formatted_code

                # Evaluate generated function
        restricted_globals = CodeContestsWrapperEnv.generate_restricted_globals()
        generic_success_msg = 'Code passed all tests'
        feedbacks = {}
        any_failed = False
        if self.datapoint['time_limit'] is not None:
            timeout = self.datapoint['time_limit']['seconds'] + 1e-9 * self.datapoint['time_limit']['nanos']
        else:
            timeout = 1
        for test_code, public_private in zip([self.datapoint['public_tests'], self.datapoint['private_tests']], ['public', 'private']):
            try:
                func = lambda: self._custom_stdin_stdout_test(formatted_code, restricted_globals, test_code, timeout)
                self._timeout_exec(func, timeout=timeout * (len(test_code['input']) + 1))
                # func
                # self._custom_stdin_stdout_test(formatted_code, restricted_globals, test_code, timeout)
                feedbacks[public_private] = generic_success_msg
            except Exception as e:
                exception_name = e.__class__.__name__
                error = f"{exception_name}{f': {str(e)}' if str(e) else ''}"
                feedbacks[public_private] = f"[Error] {error}"
                any_failed = True
        success = not any_failed
        reward = int(success)
        return feedbacks, reward, success
