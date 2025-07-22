import pandas as pd
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def get_solved_counts_per_seed(csv_directory: str) -> pd.Series:
    """
    Analyzes all CSV files in a directory and returns a Series with the
    number of tasks solved for each seed.

    Args:
        csv_directory (str): The path to the directory containing result CSVs.

    Returns:
        A pandas Series where the index is the seed number and the values are
        the counts of solved tasks. Returns an empty Series if no data found.
    """
    all_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    if not all_files:
        print(f"Warning: No CSV files found in directory '{csv_directory}'.")
        return pd.Series(dtype=int)

    df_list = []
    for filename in all_files:
        try:
            basename = os.path.basename(filename)
            # Assumes filename format like 'taskid_seed_1.csv'
            seed = int(basename.rsplit("_", 1)[1].split(".")[0])

            if os.path.getsize(filename) > 0:
                df = pd.read_csv(filename)
                df["seed"] = seed
                df_list.append(df)
        except (pd.errors.EmptyDataError, IndexError, ValueError):
            print(f"Warning: Skipping malformed or empty file: {filename}")
            continue

    if not df_list:
        print(f"Warning: No valid data could be read from '{csv_directory}'.")
        return pd.Series(dtype=int)

    results_df = pd.concat(df_list, ignore_index=True)

    # Ensure 'test_fitness' is numeric, coercing errors
    results_df["test_fitness"] = pd.to_numeric(
        results_df["test_fitness"], errors="coerce"
    )

    # A task is considered "solved" if its test_fitness is exactly 1.0
    results_df["is_solved"] = results_df["test_fitness"] == 1.0

    # Group by seed and count the number of solved tasks for each run
    solved_counts_per_seed = results_df.groupby("seed")["is_solved"].sum()

    return solved_counts_per_seed


def plot_performance_distribution(results: dict[str, pd.Series]):
    """
    Generates and saves a violin plot comparing the performance distributions
    of different algorithms.

    Args:
        results (dict): A dictionary mapping algorithm names to their
                        pandas Series of solved counts per seed.
    """
    # Prepare data for Seaborn by creating a long-form DataFrame
    plot_data = []
    for algo_name, solved_counts in results.items():
        for count in solved_counts:
            plot_data.append({"Algorithm": algo_name, "Tasks Solved": count})

    if not plot_data:
        print("Error: No data available to plot.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.violinplot(
        x="Algorithm",
        y="Tasks Solved",
        data=df_plot,
        ax=ax,
        palette="muted",
        inner="quartile",  # Shows the median and quartiles inside the violin
    )

    ax.set_ylabel("Number of Tasks Solved per Run", fontsize=14)
    ax.set_xlabel("")  # X-axis label is clear from the ticks
    ax.set_title(
        "Performance Distribution of Different Algorithms",
        fontsize=18,
        fontweight="bold",
    )

    plt.xticks(rotation=10, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the plot to a file
    output_filename = "violin_performance_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved as '{output_filename}'")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a violin plot comparing the performance distribution of ARC solvers."
    )

    # Argument to specify directories
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="A list of directories containing the result files for each algorithm.",
    )

    # Argument to specify the names for the algorithms in the plot
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="A list of names for each algorithm, corresponding to the directories provided via --dirs.",
    )

    args = parser.parse_args()

    if len(args.dirs) != len(args.names):
        raise ValueError("The number of directories must match the number of names.")

    # Main logic
    performance_results = {}
    for i, directory in enumerate(args.dirs):
        algo_name = args.names[i]
        print(f"Analyzing results for '{algo_name}' in directory '{directory}'...")
        solved_counts = get_solved_counts_per_seed(directory)
        if not solved_counts.empty:
            performance_results[algo_name] = solved_counts

    if performance_results:
        plot_performance_distribution(performance_results)
    else:
        print("\nNo valid results found to plot.")
