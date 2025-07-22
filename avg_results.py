import pandas as pd
import os
import glob
import argparse


def calculate_solver_performance(csv_directory: str):
    """
    Reads all result CSVs from a directory, calculates the solver's overall
    performance, including the average number of tasks solved per run.

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
            # Extract seed from filename and add as a column
            basename = os.path.basename(filename)
            # Assumes filename format like 'taskid_seed_1.csv'
            seed = int(basename.rsplit("_", 1)[1].split(".")[0])

            if os.path.getsize(filename) > 0:
                df = pd.read_csv(filename)
                df["seed"] = seed  # Add seed column for grouping
                df_list.append(df)
            else:
                print(f"Warning: Skipping empty file: {filename}")
        except (pd.errors.EmptyDataError, IndexError, ValueError):
            print(f"Warning: Skipping malformed or empty file: {filename}")
            continue

    if not df_list:
        print("Error: All CSV files were empty or could not be read.")
        return

    results_df = pd.concat(df_list, ignore_index=True)

    # Convert test_fitness to numeric, coercing errors to NaN
    results_df["test_fitness"] = pd.to_numeric(
        results_df["test_fitness"], errors="coerce"
    )
    # Clip fitness to be non-negative
    results_df["test_fitness"] = results_df["test_fitness"].clip(lower=0)

    # Calculate tasks solved per seed run
    # A task is "solved" if its test_fitness is exactly 1.0
    results_df["is_solved"] = results_df["test_fitness"] == 1.0

    # Group by seed and count the number of solved tasks for each run
    solved_counts_per_seed = results_df.groupby("seed")["is_solved"].sum()

    if solved_counts_per_seed.empty:
        print("No valid data to calculate solved task statistics.")
        avg_solved = 0
        std_dev_solved = 0
    else:
        # Calculate the average and standard deviation of solved tasks across all seeds
        avg_solved = solved_counts_per_seed.mean()
        std_dev_solved = solved_counts_per_seed.std()

    # Define required columns for overall averages
    required_columns = ["test_fitness", "time_taken", "solution_size", "evaluations"]

    for col in required_columns:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
        else:
            print(f"Warning: Column '{col}' not found. It will be ignored.")
            results_df[col] = pd.NA

    results_df.dropna(subset=required_columns, inplace=True)

    if results_df.empty:
        print("No valid data available to calculate overall performance averages.")
        return

    average_fitness = results_df["test_fitness"].mean()
    average_time = results_df["time_taken"].mean()
    average_size = results_df["solution_size"].mean()
    average_evaluations = results_df["evaluations"].mean()

    # Print Results
    print("\n" + "=" * 55)
    print("              Overall Performance Summary")
    print("=" * 55)
    print(f"Number of seed runs analyzed:      {len(solved_counts_per_seed)}")
    print(
        f"Tasks Solved (per run):          {avg_solved:.2f} ± {std_dev_solved:.2f} (mean ± std)"
    )
    print("-" * 55)
    print(f"Average Test Fitness:            {average_fitness:.4f}")
    print(f"Average Time Taken per Task:     {average_time:.2f} seconds")
    print(f"Average Solution Size (Nodes):   {average_size:.2f}")
    print(f"Average Evaluations per Task:    {average_evaluations:,.0f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average performance from ARC solver results."
    )
    parser.add_argument(
        "--csv_dir",
        default="lexicase_pixel_new_grammar",
        type=str,
        help="Directory containing the individual CSV result files.",
    )
    args = parser.parse_args()
    calculate_solver_performance(args.csv_dir)
