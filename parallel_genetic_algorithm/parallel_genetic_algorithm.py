import multiprocessing
from genetic_algorithm.genetic_algorithm import Population, Generation
from numpy import arange


class ParallelPopulation(Population):
    """This class is supposed to enable fitness calculations in parallel. It is based on Python's multiprocessing module
    https://docs.python.org/3/library/multiprocessing.html
    """
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

        """In the constructor we spawn rival generations for each pair of operators. They will be later evaluated and 
        the best one will become the official first one."""
        self.rival_generations = []  # list of instances of the class Generation
        for pair in self.operator_pairs:
            self.rival_generations.append(Generation(
                size=pop_size,
                fitness_function=self.fit_fun,
                genome_generator=self.genome_generator,
                genome_args=pair  # TODO: how to handle that?!
            ))

    def prepare_parallel_evaluation(self, members_dict: multiprocessing.Manager.dict):
        """For parallel computation, we firstly create a multiprocessing dicti of all members to be evaluated; this
        Array will be passed to the evaluate_generations method within parallel Processes.

        Each member should have a distinctive identification number, but for simplicity we save in the key of the
        dictionary both generations position in the rival_generations list and the ID, too. This way after the
        individual evaluation of fitness values, we'll be able to update the members in their respective generations.
        """
        for generation in self.rival_generations:
            gen_index = 0
            for member in generation.members:
                members_dict[str(gen_index) + str(member.id)] = member

        return members_dict

    def parallel_evaluation(self, process_id, work_start, work_complete, continue_flag, members_dict):
        """While the flag signals we are to evaluate members, we unlock the work_start barrier. Next, we iterate over
        members subscribed to a given process (which has the given process_id). Then, we're calling on their intrinsic
        method for fitness calculation, so that, in the end, we unlock the work_complete barrier.
        """
        while continue_flag.value:
            work_start.wait()
            for key, member in members_dict.items():
                member.evaluate()

    def best_fit(self):  # we return a gene sequence of the chromosomes with the highest fitness value with it's fit value
        # TODO adjust to multiple generations to evaluate
        bf = [self.current_generation.members[self.current_fitness_ranking[0].get('index')].genes,
              self.current_fitness_ranking[0].get('fitness value')]
        return bf
