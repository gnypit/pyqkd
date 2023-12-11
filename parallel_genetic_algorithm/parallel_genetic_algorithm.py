import multiprocessing
from genetic_algorithm.genetic_algorithm import Population


class ParallelPopulation(Population):
    """This class is supposed to enable fitness calculations in parallel."""
    def __init__(self, operator_pairs: list, pop_size, fit_fun, genome_generator, args, elite_size, mutation_prob=0.0,
                 seed=None):
        super().__init__(pop_size, fit_fun, genome_generator, args, elite_size, mutation_prob=mutation_prob, seed=seed)
        self.operator_pairs = operator_pairs
        self.number_of_cpu_cores = multiprocessing.cpu_count()

    def create_new_generation(self, **kwargs):
        # TODO: work on pairs of operators in a loop, and in each of the pairs perform the simulations in parallel,
        # TODO: automatically adjusting to the number of cores available
        pass