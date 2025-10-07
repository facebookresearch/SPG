# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import pandas as pd
from gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset
import os
from parsers import Parser

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
    


class SudokuDataset(GSM8KDataset):

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SUDOKU_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.sudoku_file_path = f"{cur_path}/../dataset/test_sudoku_split_new.csv"
        # self.sudoku_file_path = f"{cur_path}/../dataset/4x4_test_sudoku.csv"
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        df = pd.read_csv(self.sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
        # Convert pandas DataFrame to HuggingFace Dataset using from_pandas
        self.dataset = HFDataset.from_pandas(df)
        print("Loaded Testing Sudoku dataset with {} examples".format(len(self.dataset)))

    def format_sudoku_grid(self, sudoku_str):
        """Simplified function to format a sudoku string."""
        # Simply pass through the raw string as requested
        return sudoku_str

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = [short_example_1, short_example_2, short_example_3][:self.num_examples]
        self.few_shot_prompt = "\n\n".join(few_shot_examples)

    def validate_sudoku(self, solution_str, ground_truth=None, question=None):
        if len(question) == 16:
            puzzle_str = question
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)
        print(f"Empty cells: {empty_cells}")
        print(puzzle_str)
        if solution_str is None or len(solution_str) == 0:
            return 0, empty_cells, 0.0

        # Handle length issues
        if len(solution_str) < 16:
            # Pad with zeros if too short
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            # Truncate if too long
            solution_str = solution_str[:16]

        assert len(puzzle_str) == 16
        # Count correct cells among originally empty cells
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        accuracy = correct_cells / empty_cells
        return correct_cells, empty_cells, accuracy

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        puzzle = self.dataset[self.subsample[idx].item()]["Puzzle"]
        solution = self.dataset[self.subsample[idx].item()]["Solution"]

        # Modified question format to reference the examples in the system prompt
        question = f"Solve the following Sudoku puzzle: {puzzle}\n"

        assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"

        prompt = self.create_prompt(question)
        return prompt, question, solution
