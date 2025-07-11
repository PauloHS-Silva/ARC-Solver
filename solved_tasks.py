import os
import glob
import pandas as pd
import argparse


def find_solved_tasks(csv_directory: str):
    """
    Reads all result CSVs from a directory and identifies which tasks
    were completely solved (test_fitness == 1.0).

    Args:
        csv_directory (str): The path to the directory containing the result CSVs.
    """
    # Find all CSV files
    all_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in directory '{csv_directory}'.")
        return

    print(f"Scanning {len(all_files)} files in '{csv_directory}'...\n")

    solved_tasks = []

    # Loop through each file and check for solved tasks
    for filename in all_files:
        try:
            # Read the CSV file
            df = pd.read_csv(filename)

            # Check if the required columns exist and the dataframe is not empty
            if (
                not df.empty
                and "test_fitness" in df.columns
                and "task_id" in df.columns
            ):
                # Convert test_fitness to numeric, coercing errors to NaN
                df["test_fitness"] = pd.to_numeric(df["test_fitness"], errors="coerce")

                # Check if any row has a test_fitness of 1.0
                if (df["test_fitness"] == 1.0).any():
                    # Get the task_id for the first solved entry
                    task_id = df.loc[df["test_fitness"] == 1.0, "task_id"].iloc[0]
                    solved_tasks.append(task_id)

        except Exception as e:
            print(f"Error processing file {os.path.basename(filename)}: {e}")

    # Print the summary
    if solved_tasks:
        print("=" * 30)
        print("      Completely Solved Tasks")
        print("=" * 30)
        for task in sorted(solved_tasks):
            print(task)
        print("\n" + f"Total solved: {len(solved_tasks)}")
        print("=" * 30)
    else:
        print("No completely solved tasks found in this directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find completely solved tasks from ARC result CSVs."
    )

    parser.add_argument(
        "--csv_dir",
        default="arc_results_train_lexicase",
        type=str,
        help="Directory containing the CSV files to scan.",
    )

    args = parser.parse_args()

    find_solved_tasks(args.csv_dir)
