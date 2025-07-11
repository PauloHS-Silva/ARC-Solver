import pandas as pd
import os
import glob
import argparse


def calculate_average_results(csv_directory: str):
    """
    Reads all result CSVs from a directory, combines them, and calculates
    the average test fitness, time taken, and solution size.

    Args:
        csv_directory (str): The path to the directory containing the result CSVs.
    """
    # Load and Combine Data
    # Find all CSV files in the specified directory
    all_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not all_files:
        print(f"Error: No CSV files found in directory '{csv_directory}'.")
        return

    # Read each CSV and append it to a list
    df_list = []
    for filename in all_files:
        try:
            # Check if file is not empty to avoid errors
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

    # Concatenate all dataframes into one
    results_df = pd.concat(df_list, ignore_index=True)

    print(f"Successfully loaded and combined data from {len(results_df)} tasks.")

    # Calculate Averages
    # Ensure data types are correct for calculation
    results_df["test_fitness"] = pd.to_numeric(
        results_df["test_fitness"], errors="coerce"
    )
    results_df["time_taken"] = pd.to_numeric(results_df["time_taken"], errors="coerce")
    results_df["solution_size"] = pd.to_numeric(
        results_df["solution_size"], errors="coerce"
    )

    # Clip fitness to be non-negative
    results_df["test_fitness"] = results_df["test_fitness"].clip(lower=0)

    # Drop rows where conversion might have failed for any of the key metrics
    results_df.dropna(
        subset=["test_fitness", "time_taken", "solution_size"], inplace=True
    )

    if results_df.empty:
        print("No valid data available to calculate averages.")
        return

    # Calculate the mean values
    average_fitness = results_df["test_fitness"].mean()
    average_time = results_df["time_taken"].mean()
    average_size = results_df["solution_size"].mean()

    # Print Results
    print("\n" + "=" * 35)
    print("      Overall Performance Summary")
    print("=" * 35)
    print(f"Average Test Fitness:          {average_fitness:.4f}")
    print(f"Average Time Taken per Task:   {average_time:.2f} seconds")
    print(f"Average Solution Size (Nodes): {average_size:.2f}")
    print("=" * 35)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average performance from ARC solver results."
    )

    # Set the default directory to 'arc_results'
    parser.add_argument(
        "--csv_dir",
        default="arc_results",
        type=str,
        help="Directory containing the individual CSV result files.",
    )

    args = parser.parse_args()

    calculate_average_results(args.csv_dir)
