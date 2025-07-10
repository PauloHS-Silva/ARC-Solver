from __future__ import annotations

import numpy as np
from skimage.transform import resize
from collections import Counter
from dsl import Grid
import subprocess
from typing import Callable, Any, Tuple, List
from grammar import Expression, InputGrid
import os
import json

Example = Tuple[Grid, Grid]
Task = Tuple[Tuple[Example], Tuple[Example]]


def get_most_common_color(grid: Grid) -> int:
    """Finds the most frequent color in a grid."""
    # Flatten the grid to a 1D list of colors
    colors = [color for row in grid for color in row]
    if not colors:
        return 0  # Default to 0 if grid is empty

    # Find and return the most common color
    return Counter(colors).most_common(1)[0][0]


def calculate_shape_fitness(predicted_grid, target_grid, size=(10, 10)):
    """Downscales grids and compares them to get a shape similarity score."""
    try:
        predicted_resized = resize(
            np.array(predicted_grid), size, anti_aliasing=True, preserve_range=True
        ).astype(int)
        target_resized = resize(
            np.array(target_grid), size, anti_aliasing=True, preserve_range=True
        ).astype(int)

        matches = np.sum(predicted_resized == target_resized)
        return matches / (size[0] * size[1])
    except Exception:
        return -1


def calculate_placement_fitness(predicted_grid, target_grid):
    """Binarizes grids and compares them to get a placement similarity score."""
    # Determine the background color from the target grid
    try:
        background_color = get_most_common_color(target_grid)

        # Binarize both grids based on the identified background color
        # Cells matching the background become 0 (black).
        # All other cells become 1 (white).
        predicted_binary = (np.array(predicted_grid) != background_color).astype(int)
        target_binary = (np.array(target_grid) != background_color).astype(int)

        if predicted_binary.shape != target_binary.shape:
            return 0.0

        # Compare the binarized grids and calculate the score
        matches = np.sum(predicted_binary == target_binary)
        return matches / predicted_binary.size
    except Exception:
        return -1


def calculate_pixel_fitness(predicted_grid, target_grid):
    """The original pixel-perfect comparison."""
    try:
        predicted_arr = np.array(predicted_grid)
        target_arr = np.array(target_grid)

        # Grids must have the same shape to be compared
        if predicted_arr.shape != target_arr.shape:
            return 0.0

        matches = np.sum(predicted_arr == target_arr)
        return matches / predicted_arr.size
    except Exception:
        return -1


def get_git_commit_hash() -> str:
    """Gets the current git commit hash of the repository."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except Exception:
        return "N/A"


def pretty_print_program(expression: "Expression", indent_level: int = 0) -> str:
    """
    Recursively builds a clean, Python-like string representation of the
    expression tree with indentation, ignoring internal metadata.
    """
    if not any(
        isinstance(v, Expression) for v in getattr(expression, "__dict__", {}).values()
    ):
        return str(expression)

    child_indent = "    " * (indent_level + 1)
    args = []

    for field, value in expression.__dict__.items():
        if field.startswith("gengy_"):
            continue

        if isinstance(value, Expression):
            arg_str = pretty_print_program(value, indent_level + 1)
            args.append(f"\n{child_indent}{field}={arg_str}")

    closing_indent = "    " * indent_level

    if not args:
        return f"{expression.__class__.__name__}()"
    else:
        return f"{expression.__class__.__name__}({''.join(args)}\n{closing_indent})"


def count_nodes(expression: Expression) -> int:
    """
    Recursively counts the number of nodes in an expression tree.
    This version correctly handles both single nodes and lists of nodes.
    """
    # Start the count at 1 for the current node
    count = 1

    # Check if the node has attributes to recurse into
    if hasattr(expression, "__dict__"):
        for field, value in expression.__dict__.items():
            # Ignore internal attributes from the genetic engine
            if field.startswith("gengy_"):
                continue

            # The attribute is a single Expression node
            if isinstance(value, Expression):
                count += count_nodes(value)

            # The attribute is a list of nodes
            elif isinstance(value, list):
                for item in value:
                    # Check if the item in the list is an Expression node
                    if isinstance(item, Expression):
                        count += count_nodes(item)
    return count


def program_uses_input(expression: Expression) -> bool:
    """Recursively checks if the expression tree contains an InputGrid node."""
    if isinstance(expression, InputGrid):
        return True
    for field, value in expression.__dict__.items():
        if isinstance(value, Expression) and program_uses_input(value):
            return True
    return False


def create_program(expression: Expression) -> Callable:
    """
    Creates a callable function from an evolved expression tree.
    """

    def program(*args, **kwargs) -> Any:
        if "input_grid" not in kwargs and args:
            kwargs["input_grid"] = args[0]

        return expression.evaluate(**kwargs)

    return program


def to_hashable_grid(grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """
    Recursively converts a list of lists into a hashable tuple of tuples.
    Handles cases where the grid might be None.
    """
    if grid is None:
        return None
    return tuple(map(tuple, grid))


def load_arc_task_by_id(task_id: str) -> Task:
    """
    Loads an ARC task from a JSON file and ensures all components,
    including the grids, are hashable tuples.
    """
    arc_data_path = os.environ.get(
        "ARC_DATA_PATH", "/Users/paulo/Desktop/ARC-AGI/data/training"
    )
    file_path = os.path.join(arc_data_path, f"{task_id}.json")

    with open(file_path) as f:
        data = json.load(f)

    train = tuple(
        [
            (to_hashable_grid(ex["input"]), to_hashable_grid(ex["output"]))
            for ex in data["train"]
        ]
    )

    test = tuple(
        [
            (to_hashable_grid(ex["input"]), to_hashable_grid(ex.get("output")))
            for ex in data["test"]
        ]
    )

    return (train, test)
