# ARC-Solver: A Genetic Programming Approach to the Abstraction and Reasoning Corpus

This repository contains a research project aimed at solving the Abstraction and Reasoning Corpus (ARC) using Genetic Programming (GP). The goal is to evolve computer programs that can solve these abstract reasoning puzzles, which are designed to be a benchmark for artificial general intelligence.

## Core Idea

The central approach is to represent potential solutions as program trees. These programs are constructed from a rich Domain-Specific Language (DSL) tailored for grid manipulation and are evolved over generations to find a solution that correctly transforms the input grids into the output grids for a given task.

This project uses the **GeneticEngine** framework to handle the core evolutionary algorithm.

## Project Structure

The repository is organized into the following components:

* **`solver.py`**: The main entry point for running the evolutionary search. It configures the GP parameters, sets up the problem, and initiates the search for a specific ARC task.

* **`dsl.py`**: Contains the core implementation of the Domain-Specific Language. This file includes over 150 functions for arithmetic, logic, and complex grid manipulations like object detection, rotation, and painting.

* **`grammar.py`**: Defines the typed structure of the DSL for the genetic programming engine. It wraps the functions from `dsl.py` into classes that allow GeneticEngine to construct syntactically correct program trees.

* **`fitness.py`**: Contains the logic for evaluating the quality of an evolved program.

* **`utils.py`**: A collection of helper functions for loading ARC task data from JSON files, counting program nodes (for parsimony pressure), and pretty-printing the final solution trees.

## How to Run

### 1. Setup

First, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

### 2. Running the Solver

You can run the solver on a specific ARC task by providing its ID as a command-line argument.

```bash
python solver.py --task_id "62c24649"
