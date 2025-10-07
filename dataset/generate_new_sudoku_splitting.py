# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import random
from collections import defaultdict

def generate_train_test_split(input_file='4x4_sudoku_unique_puzzles.csv', 
                             train_solutions=200, 
                             test_size=256,
                             random_seed=42,
                             forced_train_solutions=None,
                             exclude_puzzles=None):
    """
    Generate train-test split for Sudoku dataset.
    
    Args:
        input_file: Path to input CSV file with puzzle-solution pairs
        train_solutions: Number of unique solutions for training set
        test_size: Total number of puzzles in test set
        random_seed: Random seed for reproducibility
        forced_train_solutions: List of solutions that must be in training set
        exclude_puzzles: List of puzzles to exclude from both train and test sets
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Set default values for optional parameters
    if forced_train_solutions is None:
        forced_train_solutions = []
    if exclude_puzzles is None:
        exclude_puzzles = []
    
    forced_train_solutions = set(forced_train_solutions)
    exclude_puzzles = set(exclude_puzzles)
    
    print(f"Reading data from {input_file}...")
    # Read the CSV file
    df = pd.read_csv(input_file, dtype={'Puzzle': str, 'Solution': str})
    print(f"Total rows: {len(df)}")
    
    # Filter out excluded puzzles
    if exclude_puzzles:
        initial_size = len(df)
        df = df[~df['Puzzle'].isin(exclude_puzzles)]
        print(f"Excluded {initial_size - len(df)} puzzles, remaining: {len(df)}")
    
    # Group puzzles by their solutions
    solution_to_puzzles = defaultdict(list)
    for idx, row in df.iterrows():
        puzzle = row['Puzzle']
        solution = row['Solution']
        solution_to_puzzles[solution].append(puzzle)
    
    unique_solutions = list(solution_to_puzzles.keys())
    print(f"Total unique solutions: {len(unique_solutions)}")
    
    # Handle forced training solutions
    available_solutions = [sol for sol in unique_solutions if sol not in forced_train_solutions]
    
    # Check if forced solutions exist in the dataset
    missing_forced = forced_train_solutions - set(unique_solutions)
    if missing_forced:
        print(f"Warning: Forced training solutions not found in dataset: {missing_forced}")
    
    existing_forced = forced_train_solutions & set(unique_solutions)
    print(f"Forced training solutions: {len(existing_forced)}")
    
    # Calculate how many more solutions we need for training
    remaining_train_needed = train_solutions - len(existing_forced)
    
    if remaining_train_needed < 0:
        print(f"Warning: More forced solutions ({len(existing_forced)}) than requested train solutions ({train_solutions})")
        remaining_train_needed = 0
    
    # Randomly select additional training solutions
    random.shuffle(available_solutions)
    additional_train_solutions = available_solutions[:remaining_train_needed]
    
    # Combine forced and randomly selected training solutions
    train_solutions_set = existing_forced | set(additional_train_solutions)
    test_solutions_set = set(available_solutions[remaining_train_needed:])
    
    print(f"Train solutions: {len(train_solutions_set)}")
    print(f"Test solutions: {len(test_solutions_set)}")
    
    # Create training data - all puzzles with train solutions
    train_data = []
    for solution in train_solutions_set:
        assert len(solution) == 16, f"Invalid solution length: {len(solution)}"
        puzzles = solution_to_puzzles[solution]
        for puzzle in puzzles:
            assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"
            train_data.append({'Puzzle': puzzle, 'Solution': solution})
    
    print(f"Training data size: {len(train_data)}")
    
    # Create test data - sample puzzles from test solutions
    test_data = []
    n_test_solutions = len(test_solutions_set)
    puzzles_per_solution = test_size // n_test_solutions
    
    print(f"Puzzles per test solution: {puzzles_per_solution}")
    
    # Sample puzzles for each test solution
    for solution in test_solutions_set:
        available_puzzles = solution_to_puzzles[solution]
        
        # If there are more puzzles than needed, sample randomly
        if len(available_puzzles) >= puzzles_per_solution:
            selected_puzzles = random.sample(available_puzzles, puzzles_per_solution)
        else:
            # If fewer puzzles available, take all of them
            selected_puzzles = available_puzzles
            
        for puzzle in selected_puzzles:
            assert len(solution) == 16, f"Invalid solution length: {len(solution)}"
            assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"
            test_data.append({'Puzzle': puzzle, 'Solution': solution})
    
    # If we need more puzzles to reach exactly 256, sample additional ones
    current_test_size = len(test_data)
    if current_test_size < test_size:
        remaining_needed = test_size - current_test_size
        print(f"Need {remaining_needed} more test puzzles to reach {test_size}")
        
        # Find test solutions that still have available puzzles
        available_test_solutions = []
        for solution in test_solutions_set:
            available_puzzles = solution_to_puzzles[solution]
            already_selected = [item['Puzzle'] for item in test_data if item['Solution'] == solution]
            remaining = [p for p in available_puzzles if p not in already_selected]
            if remaining:  # Only include solutions that have remaining puzzles
                available_test_solutions.append(solution)
        
        print(f"Available test solutions with remaining puzzles: {len(available_test_solutions)}")
        
        # Randomly select solutions to fill the remaining needed puzzles
        if len(available_test_solutions) >= remaining_needed:
            selected_solutions = random.sample(available_test_solutions, remaining_needed)
        else:
            # If not enough solutions, take all available ones
            selected_solutions = available_test_solutions
            print(f"Warning: Only {len(available_test_solutions)} solutions available, but need {remaining_needed}")
        
        # For each selected solution, randomly pick one puzzle
        for solution in selected_solutions:
            available_puzzles = solution_to_puzzles[solution]
            already_selected = [item['Puzzle'] for item in test_data if item['Solution'] == solution]
            remaining = [p for p in available_puzzles if p not in already_selected]
            
            if remaining:
                # Randomly select one puzzle from this solution
                selected_puzzle = random.choice(remaining)
                assert len(selected_puzzle) == 16, f"Invalid puzzle length: {len(selected_puzzle)}"
                assert len(solution) == 16, f"Invalid solution length: {len(solution)}"
                test_data.append({'Puzzle': selected_puzzle, 'Solution': solution})
    
    print(f"Final test data size: {len(test_data)}")
    
    # random shuffle the train and test data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Convert to DataFrames and save
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Save to CSV files
    train_file = 'train_sudoku_split_new.csv'
    test_file = 'test_sudoku_split_new.csv'
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nSaved training data to {train_file} ({len(train_df)} puzzles)")
    print(f"Saved test data to {test_file} ({len(test_df)} puzzles)")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Unique solutions in train set: {len(train_solutions_set)}")
    print(f"- Unique solutions in test set: {len(test_solutions_set)}")
    print(f"- Training puzzles: {len(train_df)}")
    print(f"- Test puzzles: {len(test_df)}")
    
    return train_df, test_df

if __name__ == "__main__":
    # In-context solutions to force into training set
    forced_train_solutions = [
        "3214142323414132",
        "4213132424313142", 
        "2341413232141423"
    ]
    
    # In-context puzzles to exclude from both sets
    exclude_puzzles = [
        "3014002020004130",
        "0000100420013142", 
        "2001403002001420"
    ]
    
    train_df, test_df = generate_train_test_split(
        forced_train_solutions=forced_train_solutions,
        exclude_puzzles=exclude_puzzles
    )
