from typing import Callable, List, Tuple
from dsl import Grid
import numpy as np
from grammar import GridExpression, InputGrid
from utils import (
    calculate_pixel_fitness,
    calculate_shape_fitness,
    calculate_placement_fitness,
    count_nodes,
    program_uses_input,
    create_program,
    Task,
)
from functools import lru_cache


def single_pixel_fitness(program: Callable, dataset: List[Tuple[Grid, Grid]]) -> float:
    """
    Evaluate a candidate program on a dataset (list of (input, expected_output) pairs).
    Returns the average percentage of pixels that the program gets right.
    """
    total_score = 0.0
    total_examples = 0
    try:
        for input_grid, expected_output in dataset:
            if expected_output is None:
                continue  # Skip if no ground truth
            actual_output = program(input_grid)
            arr_actual = np.array(actual_output)
            arr_expected = np.array(expected_output)
            if arr_actual.shape != arr_expected.shape:
                # If output shape is wrong, score is 0 for this example
                continue
            total_pixels = arr_expected.size
            correct_pixels = np.sum(arr_actual == arr_expected)
            total_score += correct_pixels / total_pixels
            total_examples += 1
        return total_score / total_examples if total_examples > 0 else 0.0
    except Exception:
        return -1


def multi_pixel_fitness(
    program: Callable, dataset: List[Tuple[Grid, Grid]]
) -> List[float]:
    """
    Evaluates a program on a dataset and returns a LIST of scores, one for each example.
    This is required for Lexicase and Multi-Objective selection.
    """
    scores = []
    for input_grid, expected_output in dataset:
        score = 0.0
        if expected_output is not None:
            try:
                actual_output = np.array(program(input_grid))
                expected_output = np.array(expected_output)
                if actual_output.shape == expected_output.shape:
                    score = (
                        np.sum(actual_output == expected_output) / expected_output.size
                    )
            except Exception:
                score = -1
        scores.append(score)
    return scores


def balanced_fitness(
    program: Callable, dataset: List[Tuple[Grid, Grid]]
) -> List[float]:
    all_scores = []
    for input_grid, target_grid in dataset:
        # Run the evolved program
        predicted_grid = program(input_grid=input_grid)

        # Calculate the three different fitness scores
        pixel_score = calculate_pixel_fitness(predicted_grid, target_grid)
        shape_score = calculate_shape_fitness(predicted_grid, target_grid)
        placement_score = calculate_placement_fitness(predicted_grid, target_grid)

        print(pixel_score, shape_score, placement_score)

        w_pixel = 1 / 3  # Reward getting pixels right
        w_shape = 1 / 3  # Reward correct general shape
        w_placement = 1 / 3  # Reward correct object placement

        combined_score = (
            (w_pixel * pixel_score)
            + (w_shape * shape_score)
            + (w_placement * placement_score)
        )

        all_scores.append(combined_score)

    return all_scores


hack = {"best": -1}


@lru_cache(maxsize=None)
def train_fitness_function(individual, task: Task):
    train, _ = task
    program = create_program(individual)
    try:
        all_scores = balanced_fitness(program, train)
    except Exception:
        # If the program crashes, it gets a negative score
        return [-1] * len(train)

    if np.mean(all_scores) > hack["best"]:
        hack["best"] = np.mean(all_scores)

    if np.mean(all_scores) >= 0:
        print(
            f"{individual.__class__.__name__} {np.mean(all_scores):.2f} - {hack['best']:.2f}"
        )

    if isinstance(individual, InputGrid):
        return [-1] * len(all_scores)

    # Program simply returns input grid
    num_nodes = count_nodes(individual)

    if not program_uses_input(individual):
        return [-1] * len(all_scores)

    # Apply parsimony pressure to each score
    parsimony_coefficient = 0.0001
    penalty = num_nodes * parsimony_coefficient
    final_scores = [max(0, s - penalty) for s in all_scores]

    return final_scores


def test_fitness_function(expr: GridExpression, task: Task) -> float:
    _, test = task
    program = create_program(expr)
    return single_pixel_fitness(program, test)
