ARC-Solver: A Genetic Programming Approach to the Abstraction and Reasoning Corpus
This repository contains a research project aimed at solving the Abstraction and Reasoning Corpus (ARC) using Genetic Programming (GP). The goal is to evolve computer programs that can solve these abstract reasoning puzzles, which are designed to be a benchmark for artificial general intelligence.

Core Idea
The central approach is to represent potential solutions as program trees. These programs are constructed from a rich Domain-Specific Language (DSL) tailored for grid manipulation and are evolved over generations to find a solution that correctly transforms the input grids into the output grids for a given task.

This project leverages the GeneticEngine framework to handle the core evolutionary algorithm.

Project Structure
The repository is organized into several key components:

solver.py: The main entry point for running the evolutionary search. It configures the GP parameters, sets up the problem, and initiates the search for a specific ARC task.

grammar.py: Defines the Domain-Specific Language (DSL). This is the heart of the solver, containing over 200 functions that serve as the building blocks for the evolved programs. These functions range from simple arithmetic to complex grid operations like object detection, rotation, and painting.

fitness.py: Contains the logic for evaluating the quality of an evolved program.

utils.py: A collection of helper functions for loading ARC task data from JSON files, counting program nodes (for parsimony pressure), and pretty-printing the final solution trees.

How to Run
1. Setup
First, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

2. Running the Solver
You can run the solver on a specific ARC task by providing its ID as a command-line argument.

python solver.py --task_id "62c24649"

The solver will run for a predefined time budget (e.g., 175 seconds) and then print the best solution it found, along with its fitness score and the full program tree.

Experiments & Analysis
This repository also includes scripts for analyzing the results of multiple runs:

plot_results.py: Generates a series of plots (histograms, scatter plots) to visualize the performance of a given algorithm across all tasks.

These tools were used to compare different evolutionary strategies, such as Tournament Selection vs. Lexicase Selection, and to analyze key trade-offs like solution size vs. fitness and the generalization gap between training and test data.
