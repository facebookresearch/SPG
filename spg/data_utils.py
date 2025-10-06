from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""
short_example_1 = "Question:\nSolve the following Sudoku puzzle: 3014002020004130\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 3 0 1 4\nR2: 0 0 2 0\nR3: 2 0 0 0\nR4: 4 1 3 0\n\nFill easy singles:\nR1 missing 2 → R1C2=2.\nR4 missing 2 → R4C4=2.\nBox D (R3-4,C3-4) then needs {1,4}; column4 can only accept 1 → R3C4=1, R3C3=4.\nR3 now missing 3 → R3C2=3.\nColumn1 missing 1 → R2C1=1.\nColumn2 missing 4 → R2C2=4.\nLast cell R2C4=3.\n\nFinal grid:\nR1: 3 2 1 4\nR2: 1 4 2 3\nR3: 2 3 4 1\nR4: 4 1 3 2\n</reasoning>\n<answer>\n3214142323414132\n</answer>"
short_example_2 = "Question:\nSolve the following Sudoku puzzle: 0000100420013142\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 0 0 0 0\nR2: 1 0 0 4\nR3: 2 0 0 1\nR4: 3 1 4 2\n\nFill easy singles:\nCol1 missing 4 → R1C1=4.\nCol4 missing 3 → R1C4=3.\nBox A (R1-2,C1-2) missing {2,3} and R1 now needs {1,2} → R1C2=2, R2C2=3.\nR1C3=1.\nR2 now missing 2 → R2C3=2.\nCol2 missing 4 → R3C2=4, then R3C3=3.\n\nFinal grid:\nR1: 4 2 1 3\nR2: 1 3 2 4\nR3: 2 4 3 1\nR4: 3 1 4 2\n</reasoning>\n<answer>\n4213132424313142\n</answer>"
short_example_3 = "Question:\nSolve the following Sudoku puzzle: 2001403002001420\nAnswer:\n<reasoning>\nInterpret puzzle as 4 rows of 4:\nR1: 2 0 0 1\nR2: 4 0 3 0\nR3: 0 2 0 0\nR4: 1 4 2 0\n\nFill easy singles:\nR1 missing {3,4}; Col2 can't be 1 so R1C2=3 → R1C3=4.\nR4 missing 3 → R4C4=3.\nCol4 missing {2,4}; R2 must take 2 → R2C4=2 → R2C2=1.\nCol1 missing 3 → R3C1=3.\nCol3 missing 1 → R3C3=1 → R3C4=4.\n\nFinal grid:\nR1: 2 3 4 1\nR2: 4 1 3 2\nR3: 3 2 1 4\nR4: 1 4 2 3\n</reasoning>\n<answer>\n2341413232141423\n</answer>"

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )

def get_sudoku_questions_new(few_shot=0) -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/train_sudoku_split_new.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)
    assert few_shot <= 3, "few_shot must be less than or equal to 3"
    few_shot_examples = [short_example_1, short_example_2, short_example_3][:few_shot]
    few_shot_prompt = "\n\n".join(few_shot_examples)
    system_prompt = f"{SUDOKU_SYSTEM_PROMPT}\n\n{few_shot_prompt}"

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    # "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                    "content": f"{system_prompt}\n\nQuestion: Solve the following Sudoku puzzle: {x['Puzzle']}\nAnswer:\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )

def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore
