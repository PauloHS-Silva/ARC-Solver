import pandas as pd
import os
import glob
import argparse


def count_solved_tasks(csv_directory: str):
    """
    Reads all result CSVs from a directory and counts how many tasks
    were solved completely (i.e., have a test_fitness of 1.0).

    Args:
        csv_directory (str): The path to the directory containing the result CSVs.
    """
    # Load and Combine Data
    all_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in directory '{csv_directory}'.")
        return

    df_list = []
    for filename in all_files:
        try:
            if os.path.getsize(filename) > 0:
                df_list.append(pd.read_csv(filename))
            else:
                print(f"Warning: Skipping empty file: {filename}")
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping malformed or empty file: {filename}")
            continue

    if not df_list:
        print("Error: All CSV files were empty or could not be read.")
        return

    results_df = pd.concat(df_list, ignore_index=True)

    # Count Solved Tasks
    # Ensure the test_fitness column is numeric, coercing errors
    results_df["test_fitness"] = pd.to_numeric(
        results_df["test_fitness"], errors="coerce"
    )

    # A task is considered solved if its test_fitness is exactly 1.0
    # We use a small tolerance to account for potential floating point inaccuracies
    solved_tasks_df = results_df[results_df["test_fitness"] >= 0.99999]

    total_tasks_attempted = len(results_df)
    num_solved = len(solved_tasks_df)

    # Print Results
    print("\n" + "=" * 35)
    print("      Task Completion Summary")
    print("=" * 35)
    print(f"Total Tasks Attempted: {total_tasks_attempted}")
    print(f"Tasks Completely Solved: {num_solved}")

    if total_tasks_attempted > 0:
        solve_rate = (num_solved / total_tasks_attempted) * 100
        print(f"Overall Solve Rate:   {solve_rate:.2f}%")

    print("=" * 35)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count completely solved tasks from ARC solver results."
    )

    # Set the default directory to 'arc_results'
    parser.add_argument(
        "--csv_dir",
        default="arc_results_train_lexicase",
        type=str,
        help="Directory containing the individual CSV result files.",
    )

    args = parser.parse_args()

    count_solved_tasks(args.csv_dir)
