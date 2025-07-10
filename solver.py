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


from utils import pretty_print_program, create_program, load_arc_task_by_id
from grammar import grammar, GridExpression
from fitness import train_fitness_function, test_fitness_function


def solve_task(task_id: str) -> None:
    task = load_arc_task_by_id(task_id)
    # Create the problem

    def task_train_fitness(expr: GridExpression) -> List[float]:
        """
        Evaluate the individual on the training set of the task.
        Returns a list of fitness scores for each example in the training set.
        """
        return train_fitness_function(expr, task)

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

    # Run the search
    best_individuals = alg.search()
    best_individual = best_individuals[
        0
    ]  # Get the first (best) individual from the list

    print("\nBest solution found:")
    print(
        f"Train fitness: {train_fitness_function(best_individual.get_phenotype(), task)}"
    )
    print(
        f"Test fitness: {test_fitness_function(best_individual.get_phenotype(), task)}"
    )
    print("\nSolution tree:")
    print(pretty_print_program(best_individual.get_phenotype()))
    print(alg.tracker.get_number_evaluations())

    # Create and return the final program
    return create_program(best_individual)


if __name__ == "__main__":
    # Solve the specific task
    program = solve_task("62c24649")
