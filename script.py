from __future__ import annotations

from typing import List
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.prelude import GeneticProgramming
from geneticengine.random.sources import NativeRandomSource
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.evaluation.tracker import ProgressTracker

from geneticengine.solutions.individual import Individual
import time
import csv
import argparse
import os
import numpy as np

from utils import (
    pretty_print_program,
    create_program,
    load_arc_task_by_id,
    count_nodes,
    get_git_commit_hash,
)
from grammar import grammar
from fitness import train_fitness_function, test_fitness_function


def solve_task(task_id: str, output_dir: str) -> None:
    # Get versioninig information
    dsl_version = get_git_commit_hash()
    algorithm_version = "multi_objective_lexicase"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define a unique CSV file path FOR THIS TASK inside the output directory
    csv_file_path = os.path.join(output_dir, f"{task_id}.csv")

    if os.path.exists(csv_file_path):
        print(f"Result for task {task_id} already exists. Skipping.")
        return

    task = load_arc_task_by_id(task_id)
    # Create the problem

    def task_train_fitness(individual: Individual) -> List[float]:
        """
        Evaluate the individual on the training set of the task.
        Returns a list of fitness scores for each example in the training set.
        """
        return train_fitness_function(individual, task)

    num_objectives = len(task[0])
    minimize_list = [False] * num_objectives
    problem = MultiObjectiveProblem(
        fitness_function=task_train_fitness, minimize=minimize_list
    )
    tracker = ProgressTracker(problem)

    # Configure GP parameters
    gp_params = {
        "population_size": 200,
        "n_elites": 4,
        "probability_mutation": 0.15,
        "probability_crossover": 0.8,
        "timer_limit": 175,
        "novelty": 15,
        "max_depth": 10,
        "tournament_size": 5,
    }

    # Create the GP step
    gp_step = ParallelStep(
        [
            ElitismStep(),
            SequenceStep(
                LexicaseSelection(),
                GenericCrossoverStep(gp_params["probability_crossover"]),
                GenericMutationStep(gp_params["probability_mutation"]),
            ),
        ],
        weights=[
            gp_params["n_elites"],
            gp_params["population_size"] - gp_params["n_elites"],
        ],
    )

    # Create and run the algorithm
    random = NativeRandomSource(42)
    alg = GeneticProgramming(
        problem=problem,
        budget=TimeBudget(time=gp_params["timer_limit"]),
        representation=TreeBasedRepresentation(
            grammar=grammar,
            decider=MaxDepthDecider(random, grammar, gp_params["max_depth"]),
        ),
        random=random,
        population_size=gp_params["population_size"],
        step=gp_step,
        tracker=tracker,
    )

    # evaluator = ParallelEvaluator()
    # tracker = ProgressTracker(problem=problem, evaluator=evaluator)

    # alg = RandomSearch(
    #     problem=problem,
    #     budget=TimeBudget(time=gp_params["timer_limit"]),
    #     representation=TreeBasedRepresentation(grammar, decider=MaxDepthDecider(random, grammar, gp_params["max_depth"])),
    #     random=random
    # )

    print(f"--- Starting search for task {task_id} ---")
    start_time = time.time()
    pareto_front = alg.search()
    end_time = time.time()
    time_taken = end_time - start_time
    num_evaluations = alg.tracker.get_number_evaluations()

    # Write Results to the unique CSV file

    with open(csv_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "dsl_version",
                "algorithm_version",
                "train_fitness",
                "test_fitness",
                "solution_size",
                "evaluations",
                "time_taken",
                "solution_tree",
            ]
        )

        if not pareto_front:
            print(f"No solution found for task {task_id}")
            result_row = [
                task_id,
                dsl_version,
                algorithm_version,
                0.0,
                num_evaluations,
                time_taken,
                "No solution found",
            ]
        else:
            best_individual = sorted(
                pareto_front,
                key=lambda ind: np.mean(ind.get_fitness(problem).fitness_components),
                reverse=True,
            )[0]
            final_program = create_program(best_individual.get_phenotype())
            train_fitness = best_individual.get_fitness(problem)
            test_fitness = test_fitness_function(final_program, task)
            solution_size = count_nodes(best_individual.get_phenotype())
            solution_str = pretty_print_program(best_individual.get_phenotype())
            result_row = [
                task_id,
                dsl_version,
                algorithm_version,
                train_fitness,
                test_fitness,
                solution_size,
                num_evaluations,
                time_taken,
                solution_str,
            ]

        writer.writerow(result_row)

    print(f"--- Finished task {task_id}. Results saved to {csv_file_path} ---")


if __name__ == "__main__":
    # Command-Line Argument Parsing
    parser = argparse.ArgumentParser(
        description="Solve a specific ARC task and log results to CSV."
    )
    parser.add_argument(
        "--task_id", required=True, type=str, help="The ID of the ARC task to solve."
    )
    parser.add_argument(
        "--output_dir",
        default="arc_results",
        type=str,
        help="Directory to save the CSV result files.",
    )

    args = parser.parse_args()

    # Call the solver with the provided arguments
    solve_task(args.task_id, args.output_dir)
