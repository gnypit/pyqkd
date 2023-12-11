import multiprocessing
from genetic_algorithm.genetic_algorithm import Population


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

        self.rival_generations = []
        for pair in self.operator_pairs:
            pass  # TODO how to create rival generations apart from the actual generation in the Population???

    def evaluate_generation(self, reverse=True):  # true for sorting from the highest fitness value to the lowest
        # TODO adjust to multiple generations to evaluate
        """This method applies the fitness function to the current generation and sorts the fitness ranking by
        the fitness values of current generation's members - 'reverse' means sorting will be performed
        from maximum fitness to minimum."""
        self.current_fitness_ranking = []

        for i in range(len(self.current_generation.members)):
            self.current_fitness_ranking.append(
                {'index': i, 'fitness value': self.fit_fun(self.current_generation.members[i])}
            )

        self.current_fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)
        self.fitness_rankings.append(self.current_fitness_ranking)

    def best_fit(self):  # we return gene sequence of the chromosome of the highest fitness value with it's fit value
        # TODO adjust to multiple generations to evaluate
        bf = [self.current_generation.members[self.current_fitness_ranking[0].get('index')].genes,
              self.current_fitness_ranking[0].get('fitness value')]
        return bf
