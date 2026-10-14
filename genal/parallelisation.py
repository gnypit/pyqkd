"""Private parallelisation utilities."""
from collections.abc import Callable
from copy import deepcopy
try:
    from .interfaces import GAArgumentsInterface
except ImportError:
    from interfaces import GAArgumentsInterface

"""Read-only configuration installed once in each process-pool worker.

The _worker_operators is a dictionary with combinations of selection and crossover operators as values. Each combination
has it's ID as the key. 
"""
_worker_fitness_function: Callable | None = None
_worker_operators: dict[int, tuple[Callable, Callable]] | None = None
_worker_args: GAArgumentsInterface | None = None
_worker_custom_mutation_operator: Callable | None = None

ParentGenomePair = tuple[tuple[int, list | dict], tuple[int, list | dict]]


def _evaluate_genome(genome: list | dict, fitness_function: Callable) -> float:
    """Evaluate a genome using the same failure policy as :meth:`Chromosome.evaluate`."""
    try:
        result = fitness_function(genome)
        if result is None:
            print(f"Warning: fitness function returned None for genome: {genome}")
            result = 0.0
        return result
    except Exception as error:
        print(f"Error evaluating genome {genome}: {error}")
        return 0.0


def _evaluate_fitness_batch(batch: list[tuple[int, list | dict]]) -> list[tuple[int, float]]:
    """Evaluate one IPC-efficient batch of ``(member ID, genome)`` pairs in a pool worker."""
    if _worker_fitness_function is None:
        raise RuntimeError("Parallel worker was not initialized with a fitness function.")
    return [
        (member_id, _evaluate_genome(genome, _worker_fitness_function))
        for member_id, genome in batch
    ]


def _mutate_and_evaluate_batch(
        batch: list[tuple[int, int, list | dict, bool]]
) -> list[tuple[int, int, list | dict, float]]:
    """Apply an expensive custom mutation where requested and evaluate each genome in the same worker task."""
    if _worker_fitness_function is None or _worker_custom_mutation_operator is None:
        raise RuntimeError("Parallel worker was not initialized with custom mutation configuration.")

    results = []
    for member_index, member_id, genome, should_mutate in batch:
        mutated_genome = deepcopy(genome)
        if should_mutate:
            operator_result = _worker_custom_mutation_operator(mutated_genome, _worker_args)
            if operator_result is not None:
                mutated_genome = operator_result
        fitness = _evaluate_genome(mutated_genome, _worker_fitness_function)
        results.append((member_index, member_id, mutated_genome, fitness))
    return results
