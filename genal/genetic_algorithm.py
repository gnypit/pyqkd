"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import random
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import \
    Callable  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints

"""Global variable to hold IDs of chromosomes for backtracking"""
identification = 0


def sort_dict_by_fit(dictionary: dict):
    """Used as a key in 'sort' method applied to a dict with chromosomes and their fitness values.

    Parameters:
        dictionary (dict): The dictionary in which we wish to sort members of GA's generation using fitness value.

    Returns:
        dict: The same dict as provided, but sorted by fitness value.
    """
    return dictionary['fitness value']


def uniform_gene_generator(ga_args: dict):  # TODO: should generators be inside GA or external functions?
    """Simple function for generating a sample of given length from the gene_space with a uniform probability.

    Parameters:
        ga_args (dict): This dictionary is stored within the GeneticAlgorithm class and contains info about args to be
            used by either genome generator, crossover operators or selection operators. For the genome generation,
            args are stored under key 'genome'. There should be gene space and length of chromosomes (their genome).

    Returns:
         ndarray: A numpy array containing genes randomised from the gene space. There should be at least two genes
            in each chromosome, so this function should never return a single int, str, etc.
    """
    gene_space, length = ga_args.get('genome')
    return np.random.choice(gene_space, length)


class Chromosome:
    """Basic class representing chromosomes, the most fundamental objects in genetic algorithms.

    Apart from genes, in this implementation of the Genetic Algorithm, the Chromosome class also stores the fitness
    function and value. This allows self-evaluation of each chromosome.

    Attributes:
        fit_val (float): Fitness value of the chromosome. None by default, stores a float number once the chromosome
            is evaluated.
        genome (type[list | dict]): Either list or a dictionary with genes of this chromosome.
        fit_fun (Callable): Fitness function used for computing fitness value based on chromosome's genes.
    """
    fit_val: float = None
    genome: type[list | dict]
    fit_fun: Callable

    def __init__(self, genome: type[list | dict], fitness_function: Callable=None):
        """Constructor of the Chromosome class.

        Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc. However, this class is limited to storage of genes, fitness function and value, and to fitness
        evaluation.

        Parameters:
            genome (type[list | dict]): Either a dict with genes as values and names provided by the User as keys,
                or simply a list of genes.
            fitness_function (Callable=None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Can be passed later, thus it is None by default.
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
        if self.fit_fun is not None:
            self.fit_val = self.fit_fun(self.genome)
        elif fitness_function is not None:
            self.fit_fun = fitness_function
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
        self.fitness_ranking = []

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


class GeneticAlgorithm:
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
    operators: list  # I usually prefer dicts, but I want to be able to iterate over combinations of operators in here

    no_parents_pairs: int
    mutation_prob: float

    current_gen: Generation
    rival_gen: dict[int, Generation]
    accepted_gen: list[Generation]
    best_fit_history: list[float]
      
    def __zip_crossover_selection(self, selection_operators, crossover_operators):
        """Creates a list that combines pairs of elements from 'selection_operators' 
        and 'crossover_operators'. For each index 'i', it adds tuples to the 'list_to_operator' list containing
        'selection_operator[i]' and 'crossover_operator[j]' for each index 'j'.

        This way there are tuples for all combinations of operators."""
        list_to_operator = []
        for i in range(len(selection_operators)):
            for j in range(len(crossover_operators)):
                list_to_operator.append((selection_operators[i],crossover_operators[j]))
        return list_to_operator

    def __init__(self, initial_pop_size: int, number_of_generations: int, elite_size: int, args: dict,
                 fitness_function: Callable, genome_generator: Callable,
                 selection: list[Callable] | Callable, crossover: list[Callable] | Callable,
                 pool_size, no_parents_pairs=None, mutation_prob=0.0,
                 seed=None):  # TODO: put pool_size in the args dict for self.selection_args = args.get('selection') below
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
        if type(selection) is not list:
            selection = [selection]
        if type(crossover) is not list:
            crossover = [crossover]

        self.operators = self.__zip_crossover_selection(selection_operators=selection, crossover_operators=crossover)
        # self.operators = [(sel_op, cross_op) for sel_op in selection for cross_op in crossover]
        self.pool_size = pool_size  # will be redundant after the selection args are properly handled

    def _create_initial_generation(self):
        """Creating the first - initial - generation in this population."""
        global identification
        first_members = []
        for _ in range(self.pop_size):
            genes = self.genome_generator(self.genome_generator_args)
            first_members.append(Member(
                genome=genes,
                identification_number=identification,
                fitness_function=self.fit_fun)
            )
            identification += 1
        self.current_generation = Generation(
            generation_members=first_members,
            num_parents_pairs=self.no_parents_pairs,
            elite_size=self.elite_size,
            pool_size=self.pool_size
        )
        self.current_generation.evaluate()
        self.accepted_gen = [self.current_generation]
        self.best_fit_history = [self.current_generation.fitness_ranking[0].get('fitness value')]

    def best_solution(self):  # we return genome of member with the highest fitness value with it's fit value
        bf = [self.current_generation.members[self.current_generation.fitness_ranking[0].get('index')].genome,
              self.current_generation.fitness_ranking[0].get('fitness value')]
        return bf

    def _create_rival_generations(self):  # TODO: Creating new generations, even before fitness evaluation, could be done in parallel with Pool / ProcessPoolExecutor
        """This method takes combinations of selection and crossover operators to create new, potential generations.
        Each such potential generation is a rival to the others - later only one will be accepted based on provided
        metrics, e.g. in which of the rival generations is a member with the highest fitness value."""
        global identification

        rival_id = 0
        for selection, crossover in self.operators:
            """We iterate over all combinations of operators, each time creating a new rival generation."""
            new_members = []
            parents_in_order = selection(self.current_generation)  # TODO: TypeError: tournament_selection() takes 1 positional argument but 2 were given; we still don't pass the args in a cohesive manner
            self.rival_gen = {}
            for index in range(self.no_parents_pairs):
                """We always take 2 consecutive members from the parents_in_order list and pass them to the crossover
                operator to get genomes of new members, for the rival generation, to be created."""
                child1_genome, child2_genome = crossover(
                    parents_in_order[2 * index],
                    parents_in_order[2 * index + 1],
                    self.crossover_args
                )
                new_members.append(Member(
                    genome=child1_genome,
                    identification_number=identification,
                    fitness_function=self.fit_fun)
                )
                new_members.append(Member(
                    genome=child2_genome,
                    identification_number=identification + 1,
                    fitness_function=self.fit_fun)
                )
                identification += 2
            self.rival_gen[rival_id] = Generation(
                generation_members=new_members,
                num_parents_pairs=self.no_parents_pairs,
                elite_size=self.elite_size,
                pool_size=self.pool_size  # let's keep it for now for debugging with a single rival generation
            )
            self.rival_gen[rival_id].evaluate()
            rival_id += 1

    def _choose_best_rival_generation(self):
        """This method selects one of the rival generations from the rival_gen dict, based on the highest max fitness
        value, to be accepted as a new current generation."""
        fitness_comparison = {}
        for id_of_rival, generation in self.rival_gen.items():
            fitness_comparison[id_of_rival] = generation.fitness_ranking[0].get('fitness value')
        self.current_generation = self.rival_gen.get(max(fitness_comparison, key=fitness_comparison.get))
        self.accepted_gen.append(self.current_generation)
        self.best_fit_history.append(self.current_generation.fitness_ranking[0].get('fitness value'))

    def mutate(self):
        """Mutation probability is the probability of 'resetting' a member of the current generation, i.e. changing
        it genome randomly. For optimisation purposes instead of a loop over the whole generation, I calculate the
        number of members to be mutated and then generate pseudo-randomly a list of member indexes in the current
        generation to be mutated.
        """
        number_of_mutations = np.floor(self.mutation_prob * self.current_generation.size)

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

    def run(self):
        self._create_initial_generation()
        for _ in range(self.no_generations):
            self._create_rival_generations()  # TODO: why are rival generations too short?
            self._choose_best_rival_generation()
            self.mutate()

    def fitness_plot(self):
        pass
