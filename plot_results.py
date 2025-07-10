import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
import numpy as np
import ast


def parse_and_average_fitness_list(fitness_str: str) -> float:
    """
    Parses a string representation of a list of floats and returns the average.
    Handles different formats like '[0.1, 0.2]' or '0.1|0.2|0.3'.
    """
    if not isinstance(fitness_str, str):
        return np.nan

    try:
        if fitness_str.startswith("[") and fitness_str.endswith("]"):
            scores = ast.literal_eval(fitness_str)
        elif "|" in fitness_str:
            scores = [float(s) for s in fitness_str.split("|")]
        else:
            return float(fitness_str)

        if isinstance(scores, list) and scores:
            return sum(scores) / len(scores)
    except (ValueError, SyntaxError):
        return np.nan
    return np.nan


def plot_results(csv_directory: str, output_directory: str):
    """
    Reads all CSV files from a directory, combines them, and generates
    a series of plots to analyze the solver's performance.

    Args:
        csv_directory (str): The path to the directory containing the result CSVs.
        output_directory (str): The path to save the generated plot images.
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
        except (pd.errors.EmptyDataError, IndexError):
            print(f"Warning: Skipping empty or malformed file: {filename}")
            continue

    if not df_list:
        print("Error: All CSV files were empty or could not be read.")
        return

    results_df = pd.concat(df_list, ignore_index=True)

    required_columns = ["test_fitness", "train_fitness", "time_taken", "solution_size"]
    for col in required_columns:
        if col in results_df.columns:
            if col == "train_fitness":
                results_df[col] = results_df[col].apply(parse_and_average_fitness_list)
            else:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
        else:
            print(
                f"Warning: Column '{col}' not found. Some plots may not be generated."
            )
            results_df[col] = np.nan

    results_df.dropna(subset=required_columns, inplace=True)
    print(f"Successfully loaded and cleaned {len(results_df)} results.")

    # Extract Metadata for Plot Titles and Filenames
    dsl_version = (
        results_df["dsl_version"].iloc[0]
        if "dsl_version" in results_df.columns and not results_df.empty
        else "N/A"
    )
    algorithm_version = (
        results_df["algorithm_version"].iloc[0]
        if "algorithm_version" in results_df.columns and not results_df.empty
        else "unknown_algorithm"
    )

    # Create a filename-safe version of the algorithm name
    safe_algo_name = algorithm_version.replace(" ", "_").lower()

    # Create Plots
    sns.set_theme(style="whitegrid")
    os.makedirs(output_directory, exist_ok=True)

    # Plot 1: Histogram of Test Fitness
    plt.figure(figsize=(12, 7))
    sns.histplot(data=results_df, x="test_fitness", binwidth=0.05, kde=False)
    title_str = (
        f"Distribution of Test Fitness Scores\n"
        f"(DSL: {dsl_version} | Algorithm: {algorithm_version})"
    )
    plt.title(title_str, fontsize=16)
    plt.xlabel("Test Fitness Score", fontsize=12)
    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlim(-0.05, 1.05)
    histogram_path = os.path.join(
        output_directory, f"fitness_histogram_{safe_algo_name}.png"
    )
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    print(f"Saved fitness histogram to: {histogram_path}")
    plt.close()

    # Plot 2: Fitness vs. Running Time
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=results_df, x="time_taken", y="test_fitness", alpha=0.6)
    title_str_scatter_time = (
        f"Test Fitness vs. Running Time\n"
        f"(DSL: {dsl_version} | Algorithm: {algorithm_version})"
    )
    plt.title(title_str_scatter_time, fontsize=16)
    plt.xlabel("Running Time (seconds)", fontsize=12)
    plt.ylabel("Test Fitness Score", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xscale("log")
    time_scatter_path = os.path.join(
        output_directory, f"fitness_vs_time_{safe_algo_name}.png"
    )
    plt.savefig(time_scatter_path, dpi=300, bbox_inches="tight")
    print(f"Saved fitness vs. time scatter plot to: {time_scatter_path}")
    plt.close()

    # Plot 3: Fitness vs. Solution Size
    if "solution_size" in results_df.columns:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=results_df, x="solution_size", y="test_fitness", alpha=0.6)
        title_str_size = (
            f"Test Fitness vs. Solution Size (Number of Nodes)\n"
            f"(DSL: {dsl_version} | Algorithm: {algorithm_version})"
        )
        plt.title(title_str_size, fontsize=16)
        plt.xlabel("Solution Size (Number of Nodes)", fontsize=12)
        plt.ylabel("Test Fitness Score", fontsize=12)
        plt.ylim(-0.05, 1.05)
        size_scatter_path = os.path.join(
            output_directory, f"fitness_vs_size_{safe_algo_name}.png"
        )
        plt.savefig(size_scatter_path, dpi=300, bbox_inches="tight")
        print(f"Saved fitness vs. solution size scatter plot to: {size_scatter_path}")
        plt.close()

    # Plot 4: Train Fitness vs. Test Fitness (Generalization)
    if "train_fitness" in results_df.columns:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=results_df, x="train_fitness", y="test_fitness", alpha=0.5)
        plt.plot(
            [0, 1],
            [0, 1],
            color="red",
            linestyle="--",
            label="Perfect Generalization (y=x)",
        )
        plt.title(
            f"Generalization: Test Fitness vs. Train Fitness\n"
            f"(DSL: {dsl_version} | Algorithm: {algorithm_version})",
            fontsize=16,
        )
        plt.xlabel("Average Train Fitness Score", fontsize=12)
        plt.ylabel("Test Fitness Score", fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True)
        generalization_path = os.path.join(
            output_directory, f"generalization_plot_{safe_algo_name}.png"
        )
        plt.savefig(generalization_path, dpi=300, bbox_inches="tight")
        print(f"Saved generalization plot to: {generalization_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate performance plots from ARC solver results."
    )
    parser.add_argument(
        "--csv_dir",
        default="arc_results",
        type=str,
        help="Directory containing the individual CSV result files.",
    )
    parser.add_argument(
        "--output_dir",
        default="plots",
        type=str,
        help="Directory to save the plot images.",
    )

    args = parser.parse_args()

    plot_results(args.csv_dir, args.output_dir)
