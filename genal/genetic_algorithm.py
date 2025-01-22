"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import floor
import crossover_operators
import selection_operators
from simulator_ver1 import fitness_functions
from collections.abc import Callable  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints

"""Global variable to hold IDs of chromosomes for backtracking"""
identification = 0


def sort_dict_by_fit(dictionary):
    """Used as a key in 'sort' method applied to a dict with chromosomes and their fitness values."""
    return dictionary['fitness value']


class Chromosome:
    """Basic class representing chromosomes, the most fundamental objects in genetic algorithms.
    Based on author's experience, both fitness function and value are remembered directly in chromosomes to resolve any
    problems with sharing memory in parallel programming.
    """
    fit_val: float = None
    genome: type[list | dict]

    def __init__(self, genome: type[list | dict], fitness_function=None):
        """Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc.

        Here, `genome` is either a dict with genes as values and names provided by the User as keys, or simply a list.

        For computational purposes of parallel programming, the fitness function can be passed to
        the Chromosome on its initiation/construction.
        """
        self.genome = genome
        self.fit_fun = fitness_function  # special variable

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return (f"{type(self).__name__}(genes={self.genome}, fitness function={self.fit_fun}, "
                f"fitness value={self.fit_val})")

    def change_genes(self, genes):
        """Method meant to be used when mutation occurs, to modify the genes in an already created chromosome.
        Can be called upon manually."""
        self.genome = genes

    def evaluate(self, fitness_function=None):
        """Method for applying fitness function to this chromosome (it's genes, to be precise).
        If the fitness function was passed on in the constructor of this class, it has to be provided as an argument of
        this method. Fitness value is remembered in a field of this classed and returned on output. If no fitness
        function is provided, the assigned fitness value is 0."""
        if fitness_function is None:
            self.fit_fun = fitness_function
        elif self.fit_fun is not None:  # fitness function was provided on initialisation
            self.fit_val = self.fit_fun(self.genome)
        else:
            self.fit_val = 0

        return self.fit_val


class Member(Chromosome):
    """This class is a child of the 'Chromosome' class and is designated to store a unique ID, enabling tracking a
    genealogical tree of chromosomes in a population of a genetic algorithm.
    """
    id: int
    parents_id: list  # it's a list with IDs of the parents

    def __init__(self, genome: type[list | dict], identification_number: int, fitness_function=None):
        """Apart from what 'Chromosome' class constructor needs, here identification number should be passed."""
        super().__init__(genome=genome, fitness_function=fitness_function)
        self.id = identification_number

    def add_parents_id(self, parents_id: list):
        """This method is meant for 'genealogical tree' tracking;
        it assigns to the current member IDs of its parents.
        """
        self.parents_id = parents_id

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return f"{type(self).__name__}(genes={self.genome}, id={self.id}, parents_id={self.parents_id})"


class Generation:  # TODO: we need constructor to take members, method for changes caused by mutation, method for evaluation and to return best fit; in the future add diversity measures
    """This class is meant to represent a single (rival) generation in a (parallel) genetic algorithm. It has methods
    for adding members either in constructor or manually and to evaluate the generation as a whole.

    Args:
        generation_members (list[Member]): chromosomes of the generation with their and parents' IDs
        num_parents_pairs (int): how many pairs of members can be parents, e.g., 20 pairs means 40 mating chromosomes
        elite_size (int): number of members to be copy-pasted directly into a new generation
        pool_size (int): parameter for the tournament selection operator
    """
    members: list[Member]
    num_parents_pairs: int
    elite_size: int
    pool_size: int
    size: int  # number of members in the generation
    fitness_ranking: list[dict]  # dicts in this list have the index of a member in the generation and its fitness value

    def __init__(self, generation_members: list[Member], num_parents_pairs: int, elite_size: int, pool_size: int):
        """Constructor for any generation: initial, current or rival."""
        self.members = generation_members
        self.num_parents_pairs = num_parents_pairs
        self.elite_size = elite_size
        if 0 < pool_size <= self.num_parents_pairs:
            self.pool_size = pool_size
        else:
            raise ValueError(f"Pool size = {pool_size} is not between 0 and number of parents mating "
                             f"({self.num_parents_pairs})")
        self.size = len(generation_members)

    def mutate_member(self, prob: float):
        """Method for applying a basic mutation operator to this generation - it randomly chooses a member to have their
        genome rested with the genome generator based on passed mutation probability `prob`."""
        pass

    def evaluate(self, reverse=True):
        """This method uses the fitness function stored in members of the generation to create and then sort the fitness
        ranking by the computed fitness values; 'reverse' means sorting will be performed from max fitness value to min.
        """
        for i in range(self.size):
            self.fitness_ranking.append(
                {'index': i, 'fitness value': self.members[i].evaluate()}
            )

        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)


class GeneticAlgorithm:  # TODO: separate constructor and creating the initial population & separate comments
    """Fundamental class for execution of the genetic algorithm. It implements a simple slave-master construction
    of a parallel genetic algorithm, but computationally it is executed with a single thread/process.

    Args:
        initial_pop_size (int): size of the population (each generation)
        number_of_generations (int): how many consecutive accepted generations are supposed to be created and evaluated
        elite_size (int): number of best members of each generation to be copy-pasted into the new generation
        args (dict): are arguments to be used in genome_generator & selection/crossover operators
        fitness_function (Callable): func passed to members of the population; returns a float value
            based on a member's genome
        genome_generator (Callable): func which returns genome of a single member
        selection (list[Callable] | Callable): list of func from selection_operators.py for parent selection
        crossover (list[Callable] | Callable): list of func from crossover_operators.py for children creation
        no_parents_pairs (int): optional; is the designated number of parent pairs for future generations,
            e.g., if the initial population size is 1000 and no_parents_pairs = 200,
            there will be 2 * 200 = 400 children
        mutation_prob (int): 0.0 by default; probability of selecting a member of a generation to reset its genome
            with the genome_generator
        seed (int | float | str | bytes | bytearray | None = None): optional; parameter 'a' for random.seed
    """
    pop_size: int
    no_generations: int
    elite_size: int

    args: dict
    """What the args dict should look like:
    
    args = {
        'genome': (g1, g2, ...),
        'selection': [(s11, s12, ...), ..., (sN1, sN2, ...)],
        'crossover': [(c11, c12, ...), ..., (cM1, cM2, ...)]
    }
    
    Where:
        1) g1, g2, etc., are args for the genome_generator func; 
        2) s11, s12, etc., are args for the 1st selection operator passed in the selection_operators list of func 
            and sN1, sN2, etc., are args of the Nth selection operator;
        3) c11, c12, etc., are args for the 1st crossover operator passed in the crossover_operators list of func 
            and cM1, cM2, etc., are args of the Mth crossover operator;
    """

    fit_fun: Callable
    genome_gen: Callable
    operators: dict = {}

    no_parent_pairs: int
    mutation_prob: float

    current_gen: Generation
    rival_gen: dict[int: Generation]
    best_fit_history: list = []

    def __init__(self, initial_pop_size: int, number_of_generations: int, elite_size: int, args: dict,
                 fitness_function: Callable, genome_generator: Callable,
                 selection: list[Callable] | Callable, crossover: list[Callable] | Callable,
                 pool_size, no_parents_pairs=None, mutation_prob=0.0, seed=None):  # TODO: put pool_size in the args dict for self.selection_args = args.get('selection') below
        """GeneticAlgorithm class constructor"""
        self.pop_size = initial_pop_size
        self.no_generations = number_of_generations
        self.elite_size = elite_size

        self.genome_generator_args = args.get('genome')
        self.selection_args = args.get('selection')
        self.crossover_args = args.get('crossover')

        self.fit_fun = fitness_function
        self.mutation_prob = mutation_prob
        if seed is not None:
            random.seed(a=seed)  # useful for debugging

        """If the provided number of parents pairs would require more Members than the current (initial) generation has,
        it'll be limited to the maximum possible number. Also, if no specific number of parent pairs is provided,
        the initial population size is assumed to be a constant throughout the whole algorithm."""
        if no_parents_pairs is None or no_parents_pairs > initial_pop_size // 2:
            self.no_parents_pairs = initial_pop_size // 2
        else:
            self.no_parents_pairs = no_parents_pairs

        """Even though for the initial population we can pass the genome generator with it's arguments
        directly to the __init__ method within the Generation class, we need to memorise it for mutation later on."""
        self.genome_generator = genome_generator

        """Based on lists of (callable) function selected by the User from selection_operators.py 
        and crossover_operators.py, a more general dict is created with all the possible combinations of the operators.
        """
        operators_list = [(sel_op, cross_op) for sel_op in selection for cross_op in crossover]
        for i in range(len(operators_list)):  # I prefer dicts, as they are faster than lists
            self.operators[i] = operators_list[i]

    def _create_initial_generation(self):
        """Creating the first - initial - generation in this population."""
        global identification
        new_members = []
        for _ in range(self.pop_size):
            genes = self.genome_generator()
            new_member = Member(genome=genes, identification_number=identification, fitness_function=self.fit_fun)
            identification += 1
        self.current_generation = Generation(

        )
        self.generations = [self.current_generation]

        if self.genome_generator is not None:  # ONLY for the initial generation within the population -> should it be in the GeneticAlgorithm class, or do we need it for mutation too?
            for index in range(self.size):
                genes = self.genome_generator(self.genome_generator_args)
                new_member = Member(
                    genome=genes,
                    identification_number=identification,
                    fitness_function=fitness_functions
                )
                identification += 1
                self.members.append(new_member)

        """What we need is to be able to sort whole generation based on fitness values AND remember chromosomes 
        indexes in their (generation) list in order to be able to crossbreed them with each other based on the
        fitness ranking. Thus, for each generation we create a list of dictionaries for this ranking. 
        These dictionaries shall have just two keys: index (of a chromosome) and a fitness value 
        (of the same chromosome). Once we compute such a fitness ranking for the whole generation, 
        we shall sort it using sort_dict_by_fit function."""
        self.fitness_rankings = []
        self.current_fitness_ranking = None

    def evaluate_generation(self, reverse=True):  # true for sorting from the highest fitness value to the lowest
        """This method applies the fitness function to the current generation and sorts the fitness ranking by
        the fitness values of current generation's members - 'reverse' means sorting will be performed
        from maximum fitness to the minimum."""
        self.current_fitness_ranking = []

        for i in range(len(self.current_generation.members)):
            self.current_fitness_ranking.append(
                {'index': i, 'fitness value': self.fit_fun(self.current_generation.members[i])}
            )

        self.current_fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)
        self.fitness_rankings.append(self.current_fitness_ranking)

    def best_fit(self):  # we return gene sequence of the chromosome of the highest fitness value with it's fit value
        bf = [self.current_generation.members[self.current_fitness_ranking[0].get('index')].genome,
              self.current_fitness_ranking[0].get('fitness value')]
        return bf

    def create_new_generation(  # TODO: take a selection and a crossover operator on input & create a new instance of the Generation class based on self.current_gen
            self):  # First to be parallelled & TODO: change the way the selection operators are handled
        """A method for combining selection and crossover operators over the current population to create a new one.
        Firstly, we have to match the selection operator; then in each case we have to match the crossover operator.

        In each of the selection-oriented cases we feed the selection operator name to the crossover operator
        method, so that it takes the parents lists designated for a given new generation creation, i.e., to
        always connect the chosen crossover to chosen selection and yet keep all probable parents lists
        from different selection processes in one object for multiple processes to access.

        Selection_operator is a function passed to this method for parents selection
        crossover_operator is a function passed to this method for the crossover of the parents
        """
        children_candidates = []
        for parents_candidates in self.current_parents:
            list_of_parents_pairs = parents_candidates.get(self.selection_operator)
            for parents_pair in list_of_parents_pairs:
                children_candidates.append(
                    self.crossover_operator(
                        parents_pair.get('parent1'),
                        parents_pair.get('parent2'),
                        args=self.crossover_args
                    )
                )
            self.current_children.append(
                {
                    'selection operator': parents_candidates.keys(),
                    'children': children_candidates
                }
            )

        """Secondly, we create the new generation with children being a result od selection and crossover operators
        on the current population:"""
        new_generation = Generation(size=self.no_parents_pairs * 2, fitness_function=self.fit_fun)

        for pair in self.current_children[0].get('children'):
            new_generation.mutate_member(genome=pair[0])
            new_generation.mutate_member(genome=pair[1])

        """Thirdly, we add the elite - it doesn't matter that it's at the end of the new generation, because it'll be
        sorted anyway after new Members evaluation."""
        index = 0
        while index < self.elite_size:
            new_generation.mutate_member(
                genome=self.current_generation.members[self.current_fitness_ranking[index].get('index')].genome
            )
            new_generation.mutate_member(
                genome=self.current_generation.members[self.current_fitness_ranking[index + 1].get('index')].genome
            )
            index += 2

        """Finally, we overwrite the current generation with the new one: -> NOT IN PARALLEL VERSION!!!"""
        self.current_generation = new_generation

    def mutate(self):
        """Mutation probability is the probability of 'resetting' a member of the current generation, i.e. changing
        it genome randomly. For optimisation purposes instead of a loop over the whole generation, I calculate the
        number of members to be mutated and then generate pseudo-randomly a list of member indexes in the current
        generation to be mutated.
        """
        number_of_mutations = floor(self.mutation_prob * self.current_generation.size)

        """Size of generation is a constant, it has to be adjusted to the lack of elite; the elite Members are not
        supposed to be mutated. Additionally, number of mutations has to be an integer, e.g., 
        half of a mutation cannot be performed.
        """
        indexes = random.sample(
            range(self.current_generation.size - self.elite_size),
            int(number_of_mutations)  # has to be an integer, e.g. you can't make half of a mutation
        )

        """For new (mutated) genome creation I use the generator passed to the superclass in it's initialisation:"""
        for index in indexes:
            self.current_generation.members[index].change_genes(
                self.genome_generator(self.genome_generator_args)
            )

    def change_population_size(self,
                               pop_size):  # TODO isin't it in a conflict with the change to initial size and parent pairs number?
        self.pop_size = pop_size

    def run(self):
        for _ in range(self.no_generations):
            self.evaluate_generation()
            self.create_new_generation()
            self.mutate()

    def fitness_plot(self):
        historic_best_fits = []
        for old_fitness_ranking in self.fitness_rankings:
            historic_best_fits.append(old_fitness_ranking[0].get('fitness value'))

        generation_indexes = np.arange(start=0, stop=len(historic_best_fits), step=1)

        plt.plot(generation_indexes, historic_best_fits)
        plt.show()
