import pandas as pd
import os
import glob
import argparse
import matplotlib.pyplot as plt
from upsetplot import from_contents, UpSet


def get_solved_tasks_set(csv_directory: str) -> set[str]:
    """
    Analyzes all CSV files in a directory and returns a set of unique task IDs
    that were solved in at least one run.

    Args:
        csv_directory (str): The path to the directory containing result CSVs.

    Returns:
        A set of task_id strings for all solved tasks.
    """
    all_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not all_files:
        print(f"Warning: No CSV files found in directory '{csv_directory}'.")
        return set()

    df_list = []
    for filename in all_files:
        try:
            if os.path.getsize(filename) > 0:
                df = pd.read_csv(filename, usecols=["task_id", "test_fitness"])
                df_list.append(df)
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"Warning: Skipping malformed file {filename}: {e}")
            continue

    if not df_list:
        print(f"Warning: No valid data could be read from '{csv_directory}'.")
        return set()

    results_df = pd.concat(df_list, ignore_index=True)

    # Ensure 'test_fitness' is numeric, coercing errors
    results_df["test_fitness"] = pd.to_numeric(
        results_df["test_fitness"], errors="coerce"
    )

    # A task is considered "solved" if its test_fitness is exactly 1.0
    solved_df = results_df[results_df["test_fitness"] == 1.0]

    return set(solved_df["task_id"].unique())


def plot_task_overlap(solved_sets: dict[str, set[str]]):
    """
    Generates and saves an Upset plot showing the intersection of solved tasks.

    Args:
        solved_sets (dict): A dictionary mapping algorithm names to their
                            sets of solved task IDs.
    """
    if not solved_sets or all(len(s) == 0 for s in solved_sets.values()):
        print("Error: No solved tasks found in any of the provided directories.")
        return

    upset_data = from_contents(solved_sets)

    fig = plt.figure(figsize=(12, 7))

    upset = UpSet(
        upset_data, subset_size="count", show_counts=True, sort_by="cardinality"
    )

    upset.plot(fig=fig)

    plt.suptitle(
        "Task Solving Overlap Between Algorithms", fontsize=16, fontweight="bold"
    )

    # Save the plot to a file
    output_filename = "task_overlap_upset_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{output_filename}'")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an Upset plot to show task solving overlap between ARC solvers."
    )

    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="A list of directories containing the result files for each algorithm.",
    )

    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="A list of names for each algorithm, corresponding to the directories provided via --dirs.",
    )

    args = parser.parse_args()

    if len(args.dirs) != len(args.names):
        raise ValueError("The number of directories must match the number of names.")

    # Main Logic
    solved_task_sets = {}
    for i, directory in enumerate(args.dirs):
        algo_name = args.names[i]
        print(f"Analyzing solved tasks for '{algo_name}' in directory '{directory}'...")
        solved_set = get_solved_tasks_set(directory)
        if solved_set:
            solved_task_sets[algo_name] = solved_set
            print(f" -> Found {len(solved_set)} uniquely solved tasks.")
        else:
            print(" -> No solved tasks found.")

    plot_task_overlap(solved_task_sets)
