import multiprocessing
from genetic_algorithm.genetic_algorithm import Population, sort_dict_by_fit


class ParallelPopulation(Population):
    """This class is supposed to enable fitness calculations in parallel."""
    def __init__(self, operator_pairs: list, pop_size, fit_fun, genome_generator, args, elite_size, mutation_prob=0.0,
                 seed=None):
        """The list of operators needs to be a dictionary of dictionaries of the following structure:
        operator_pairs = {
            'generation1': {'selection': ..., 'crossover': ...},
            'generation2': {'selection': ..., 'crossover': ...},
            ...,
            'generationN': {'selection': ..., 'crossover': ...}
        }
        """
        super().__init__(pop_size, fit_fun, genome_generator, args, elite_size, mutation_prob=mutation_prob, seed=seed)
        self.operator_pairs = operator_pairs
        self.number_of_cpu_cores = multiprocessing.cpu_count()

        # TODO: work on pairs of operators in a loop, and in each of the pairs perform the simulations in parallel,
        # TODO: automatically adjusting to the number of cores available

        """In the constructor we spawn a population for each pair of operators. They will be later evaluated and the 
        best one will become the official first """

    def evaluate_generations(self, process_id, work_start, work_complete):
        while True:
            work_start.wait()
            for

    def best_fit(self):  # we return gene sequence of the chromosome of the highest fitness value with it's fit value
        # TODO adjust to multiple generations to evaluate
        bf = [self.current_generation.members[self.current_fitness_ranking[0].get('index')].genes,
              self.current_fitness_ranking[0].get('fitness value')]
        return bf
