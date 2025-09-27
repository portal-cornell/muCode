"""
Utility prompt parsers from commit0 https://github.com/commit-0/commit0
"""
import re
from typing import List

def generate_prompt(prompt: str, test: str) -> str:
    """Generate a Python code request prompt string.
    
    Args:
        prompt (str): The prompt text.
        test (str): The test cases.
    
    Returns:
        str: The formatted prompt string.
    """
    return f"""Write a Python function implementation for the following prompt:

{prompt}

Your code should satisfy these tests:

{test}

Return only the implementation code, no tests or explanations. Be sure to include the relevant import statements:
```python
code
```
"""

def generate_testless_prompt(prompt: str, function_def: str) -> str:
    """Generate a Python code request prompt string.
    
    Args:
        prompt (str): The prompt text.
        test (str): The test cases.
    
    Returns:
        str: The formatted prompt string.
    """
    return f"""Write a Python function implementation for the following prompt:

{prompt}

Your code should use the following function definition:

{function_def}

Please follow the following instructions:
- Reason about the problem and any base cases before writing the code.
- You must return the implementation code in the following format:
```python
<CODE GOES HERE>
```
- You must only return a single code block since we only parse the first code block.
- Do not include any tests in your code - we will run the suite and return any error feedback.
- Include relevant import statements.
"""

def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from a given text wrapped in markdown markers.

    This function identifies and extracts all Python code blocks within a provided
    text. The code blocks should be surrounded by markdown-style markers, such as
    ```python ... ```.

    Args:
    ----
        text (str): The input text containing Python code blocks marked with
                    ```python ... ```.

    Returns:
    -------
        List[str]: A list of strings, each containing a Python code block extracted
                   from the text.

    """
    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1).strip() for match in matches]

def format_solution(text: str, prompt: str) -> str:
    """Extracts a code block from the given text and formats it as a Python code snippet.

    Args:
        text (str): The input text which may contain code blocks.
        prompt (str): A string that will be returned if no code block is found.

    Returns:
        str: A formatted code snippet if a code block exists, otherwise the prompt and text.
    """
    matches = extract_code_blocks(text)
    if len(matches) > 0:
        solution = matches[0]
        # solution = f"```python\n{solution}\n```"
    else:
        solution = text #prompt + "\n\n" + text
    return solution