"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
Script is distributed under the license: https://github.com/gnypit/pyqkd/blob/main/LICENSE
"""
import random
from os import getpid
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import \
    Callable  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
from multiprocessing import Process, Manager, cpu_count
from multiprocessing.managers import ListProxy, DictProxy

"""Global variable to hold IDs of chromosomes for backtracking"""
identification = 0


def split_indexes(num_members, num_workers):
    indexes = list(range(num_members))
    return [indexes[i::num_workers] for i in range(num_workers)]


def sort_dict_by_fit(dictionary: dict) -> float:
    """Used as a key function for sorting a list of dictionaries by their 'fitness value'.

    Parameters:
        dictionary (dict): A dictionary with at least a 'fitness value' key.

    Returns:
        float: The fitness value to be used for sorting.
    """
    return dictionary['fitness value']


def uniform_gene_generator(ga_args: dict):  # TODO: just take a tuple at the start, genome args will be passed directly, not the whole args dict
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
    return list(np.random.choice(gene_space, length))


class Chromosome:
    """Basic class representing chromosomes, the most fundamental objects in genetic algorithms.

    Apart from genes, in this implementation of the Genetic Algorithm, the Chromosome class also stores the fitness
    function and value. This allows self-evaluation of each chromosome.

    Attributes:
        fit_val (float): Fitness value of the chromosome. None by default, stores a float number once the chromosome
            is evaluated.
        genome (type[ListProxy | DictProxy]): Either list or a dict, in shared memory, with genes of this chromosome.
        fit_fun (Callable): Fitness function used for computing fitness value based on chromosome's genes.
    """
    fit_val: float = None
    genome: type[ListProxy | DictProxy]
    fit_fun: Callable

    def __init__(self, genome: type[list | dict], manager: Manager, fitness_function: Callable=None):
        """Constructor of the Chromosome class.

        Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc. However, this class is limited to storage of genes, fitness function and value, and to fitness
        evaluation.

        Parameters:
            genome (type[list | dict]): Either a dict with genes as values and names provided by the User as keys,
                or simply a list of genes.
            manager (Manager): Manager from multiprocessing is only passed on for creating proxies for list/dict, it is
                not saved in Chromosome directly - it will be saved in outer scope.
            fitness_function (Callable=None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Can be passed later, thus it is None by default.
        """
        if type(genome) == list:
            self.genome = manager.list(genome)
        elif type(genome) == dict:
            self.genome = manager.dict(genome)
        else:
            raise TypeError
        self.fit_fun = fitness_function  # special variable

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return (f"{type(self).__name__}(genes={self.genome}, fitness function={self.fit_fun}, "
                f"fitness value={self.fit_val})")

    def change_genes(self, new_genes: type[list | dict], manager: Manager):
        """Method meant to be used when mutation occurs, to modify the genes in an already created chromosome.

        Manager is only passed on for creating proxies for list/dict, it is not saved in Chromosome directly - it will
        be saved in outer scope.

        Parameters:
            new_genes (type[list | dict]): New genome to be stored by the chromosome.
            manager (Manager): Manager from multiprocessing is only passed on for creating proxies for list/dict, it is
                not saved in Chromosome directly - it will be saved in outer scope.
        """
        if type(new_genes) == list:
            self.genome = manager.list(new_genes)
        elif type(new_genes) == dict:
            self.genome = manager.dict(new_genes)
        else:
            raise TypeError

    def evaluate(self, fitness_function: Callable=None):
        """Method for applying fitness function to this chromosome (it's genes, to be precise).

        If the fitness function was passed on in the constructor of this class, it has to be provided as an argument of
        this method. Fitness value returned by this method is also remembered in an attribute of this class. If no
        fitness function is provided, the fitness value assigned by default is 0.

        Parameters:
            fitness_function (Callable=None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Could have already been provided in the constructor,
                thus it is None by default.

        Returns:
            float: Fitness value as a float number.
        """
        try:
            if self.fit_fun is not None:
                result = self.fit_fun(self.genome)
            elif fitness_function is not None:
                self.fit_fun = fitness_function
                result = self.fit_fun(self.genome)
            else:
                print(f"Warning: no fitness function available for {self}")
                result = 0.0

            if result is None:
                print(f"⚠️ Warning: fitness function returned None for genome: {self.genome}")
                print(f"It should have been {self.fit_fun(self.genome)}")

            self.fit_val = result
        except Exception as e:
            print(f"Error evaluating member {self}: {e}")
            self.fit_val = 0.0
        # return self.fit_val


class Member(Chromosome):
    """This class is a child of the 'Chromosome' class and is designated to store a unique ID, enabling tracking a
    genealogical tree of chromosomes in a population of a genetic algorithm.

    Attributes:
        id (int): A unique identification number of this member in the particular run of a genetic algorithm, created
            based on a global variable. It is meant for backtracking of a genological tree of members.
        parents_id (list): It's a list with IDs of the parents (from previous generations in the GA) of this member
    """
    id: int
    parents_id: list = []

    def __init__(self, genome: type[list | dict], manager: Manager, identification_number: int,
                 fitness_function: Callable=None):
        """Apart from what 'Chromosome' class constructor needs, here identification number should be passed.

        Parameters:
            genome (type[list | dict]): Either a dict with genes as values and names provided by the User as keys,
                or simply a list of genes.
            identification_number (int): An ID to be created based on the global variable, for backtracking a
                genological tree of all members across different generations in a particular run of the GA.
            manager (Manager): Manager from multiprocessing is only passed on for creating proxies for list/dict, it is
                not saved in Chromosome directly - it will be saved in outer scope.
            fitness_function (Callable=None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Can be passed later, thus it is None by default.
        """
        super().__init__(genome=genome, manager=manager, fitness_function=fitness_function)
        self.id = identification_number

    def add_parents_id(self, parents_id: list):
        """This method is meant for 'genealogical tree' tracking; it assigns to the current member IDs of its parents.

        Parameters:
            parents_id (list): A list with IDs of members which are parents to this member, inside the GA.
        """
        self.parents_id = parents_id

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return f"{type(self).__name__}(genes={self.genome}, id={self.id}, parents_id={self.parents_id})"


class Generation:  # TODO: we need constructor to take members, method for changes caused by mutation, method for evaluation and to return best fit; in the future add diversity measures
    """This class is meant to represent a single generation in a genetic algorithm, i.e. a set of Members.

    Genetic Algorithm evaluates each Generation, selects Members for a crossover, to create Members for a new
    Generation. In the long run the goal is to create a Generation with Members having very high fitness values.
    Each Generation is in a way static. This means, that once created, its Members may only be mutated and evaluated.
    Inside an instance of the GeneticAlgorithm class multiple Generations might be stored at the same time.

    Current Generation: the initial Generation is treated as the current one in the first iteration of the algorithm. Members
    of the first Generation will sometimes be called 'parents'.

    New / rival Generation: depending on a classical / parallel variant of the algorithm, based on 'parent' Members from
    the current Generation one (new) or multiple (rival) Generations of 'children' Members are created, from crossovers
    between selected 'parents'.

    Accepted Generation: this Generation will become the 'current' one in the next iteration of the algorithm. Either
    a single new Generation is an accepted Generation, or based on a provided metric, the best one from rival
    Generations is accepted.

    Attributes:
        members (list[Member]): list of Members; chromosomes of the generation with their and parents' IDs.
        num_parents_pairs (int): number of pairs of Members can be parents, e.g., 20 pairs means 40 mating chromosomes.
        elite_size (int): number of Members to be copy-pasted directly into a new Generation.
        pool_size (int): parameter for the tournament selection operator.  # TODO: redundant, put it into args in the GeneticAlgorithm class
        size (int): number of Members in the generation.
        fitness_ranking (list[dict]): dicts in this list have the index of a Member in the Generation as keys and its
            fitness value as values.
    """
    members: ListProxy[Member]
    num_parents_pairs: int
    elite_size: int
    pool_size: int
    size: int
    fitness_ranking: list[dict]

    def __init__(self, manager: Manager, generation_members: list[Member], num_parents_pairs: int, elite_size: int,
                 pool_size: int):
        """Constructor for any Generation inside the GeneticAlgorithm.

        Parameters:
            generation_members (ListProxy[Member]): list of Members, in shared memory, to be put in this Generation.
            num_parents_pairs (int): number of Members' pairs that can be parents.
            elite_size (int): number of Members to be copy-pasted directly into a new Generation.
            pool_size (int): parameter for the tournament selection operator.  # TODO: redundant, put it into args in the GeneticAlgorithm class
        """
        self.members = manager.list(generation_members)
        self.num_parents_pairs = num_parents_pairs
        self.elite_size = elite_size
        if 0 < pool_size <= self.num_parents_pairs:
            self.pool_size = pool_size
        else:
            raise ValueError(f"Pool size = {pool_size} is not between 0 and number of parents mating "
                             f"({self.num_parents_pairs})")
        self.size = len(generation_members)
        self.fitness_ranking = []

    def mutate_member(self, prob: float):  # TODO: implement any mutation operator as the default AND coordinate with the GeneticAlgorithm class on how to implement it exactly
        """Method for applying a basic mutation operator to this generation - it randomly chooses a member to have their
        genome rested with the genome generator based on passed mutation probability `prob`."""
        pass

    def evaluate(self, reverse=True):
        """This method uses the fitness function stored in members of the generation to create and then sort the fitness
        ranking by the computed fitness values; 'reverse' means sorting will be performed from max fitness value to min.

        Parameters:
            reverse (Bool=True, optional): parameter which decided whether the fitness ranking should be sorted in
                ascending order of fitness values (reverse=False) or in descending order (reverse=True), which is
                the default.
        """
        members_to_evaluate = list(self.members)
        for i in range(self.size):
            members_to_evaluate[i].evaluate()
            self.fitness_ranking.append(
                {'index': i, 'fitness value': members_to_evaluate[i].fit_val}  # TODO: in here fitness ranking is built correctly, fitness value is calculated, but it is not saved in the Member/Chromosome!!!
            )
            self.members[i] = members_to_evaluate[i]

        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)


def _create_rival_generation(manager: Manager, id: int, selection: Callable, crossover: Callable, crossover_args: tuple,
                             parent_generation: Generation, fitness_function: Callable, generation_pool: DictProxy):
    """Method for creating a single new Generation of children based on the parent Generation with selected operators.

    Parameters:
        manager (Manager): Manager from the outer scope for handling shared memory.
        id (int): An integer ID mathing the key under which a selection and crossover operators combination is stored in
            the operators attribute of the GeneticAlgorithm class.
        selection (Callable): Selection operator, a function returning an ordered list of parents to mate.
        crossover (Callable): Crossover operator, a function returning two (children) Members based on two provided
            (parent) Members.
        crossover_args (tuple): All arguments that are required by the crossover operator. Could be None.
        parent_generation (Generation): Any Generation containing Members who will be treated as parents to Members
            in the Generation created by this function.
        fitness_function (Callable): Fitness function for Members evaluation, that is supposed to be passed to each
            Member in the new Generation.
        generation_pool (DictProxy): A dictionary in shared memory in which all new Generations are supposed to be
            stored under the same kay as the selection and crossover operators combination.
    """
    global identification
    # selection, crossover = self.operators.get(combination_id)

    print(f"Process {getpid()}: Creating a new rival Generation")

    new_members = []
    try:
        parents_in_order = selection(parent_generation)  # TODO: either add more useful debugging tools inside selection or instead of passing a Generation to the selection operator, inject the operator into the algorithm as a Callable attribute and then debug
    except TypeError as e:
        for member in parent_generation.members:
            print(f"In parent Generation Member = {member} has fitness function {member.fit_fun}. While applying the "
                  f"selection operator, the following error occurred: {e}")
        exit()

    for index in range(parent_generation.num_parents_pairs):
        """We always take 2 consecutive members from the parents_in_order list and pass them to the crossover
        operator to get genomes of new members, for the new generation, to be created."""
        child1_genome, child2_genome = crossover(
            parents_in_order[2 * index],
            parents_in_order[2 * index + 1],
            crossover_args
        )
        new_members.append(Member(
            genome=child1_genome,
            manager=manager,
            identification_number=identification,
            fitness_function=fitness_function)
        )
        new_members.append(Member(
            genome=child2_genome,
            manager=manager,
            identification_number=identification + 1,
            fitness_function=fitness_function)
        )
        identification += 2

    new_generation = Generation(
        manager=manager,
        generation_members=new_members,
        num_parents_pairs=parent_generation.num_parents_pairs,
        elite_size=parent_generation.elite_size,  # TODO: allow changes in the elite size
        pool_size=parent_generation.pool_size  # TODO: redundant, we should focus on selection_args
    )

    """Generation pool is created as a DictProxy and each process (worker) will add it's Generation under a different 
    key, so no additional lock is required."""
    generation_pool[id] = new_generation

def _evaluate_members(generation_pool: DictProxy[int, Generation], index_range: list[int], population_size: int):
    """This function evaluates Members across multiple rival Generations.

    Parameters:
        generation_pool (DictProxy[int, Generation]): a dictionary in shared memory containing rival Generation created
            inside the `GeneticAlgorithm` class, with Members up for evaluation.
        index_range (list[int]): list containing single indexes from which ID of a Generation from the generation_pool
            and indexes of Members inside it are computed, so that they (Members) can be told to evaluate themselves.
        population_size (int): Number of Members in each Generation from the generation_pool.
    """
    for index in index_range:
        generation_id = int(np.floor(index / population_size))  # make int from numpy's float 64 ID
        member_index = int(index - generation_id * population_size)  # make int from numpy's float 64 ID

        """Fetch the WHOLE Generation, because the `.members` attr. is "nested" and can only be copied, 
        not USED in SHARED MEMORY"""
        generation = generation_pool[generation_id]
        member_to_evaluate = generation.members[member_index]
        # print(f"I have member={member_to_evaluate} with fitness function {member_to_evaluate.fit_fun}")

        member_to_evaluate.evaluate()
        fitness_value = member_to_evaluate.fit_val
        # print(f"Member number {member_index} from generation {generation_id} has fitness value = {fitness_value}")
        # print(f"Member number {member_index} from generation {generation_id} has fitness value = "
        #      f"{member_to_evaluate.fit_val}")

        generation.members[member_index] = member_to_evaluate  # <-- Modify the member
        generation.fitness_ranking.append(
                {'index': member_index, 'fitness value': fitness_value}
            )
        generation_pool[generation_id] = generation  # <-- Save back the whole Generation!!!


class GeneticAlgorithm:
    """Class with a role of a container for the hierarchical parallel genetic algorithm.

    While the fitness evaluation of members from rival Generations is diversified between as many processes operating
    in parallel on different processor cores, also creating these rival generations (selection and crossover) is
    performed by parallel processes. Processes creating Generations and processes evaluating fitness are independent.

    Attributes:
        pop_size (int): a constant size of each Generation within the algorithm.
        no_generations (int): number of iterations of the algorithm, equal to the number of accepted Generations
        elite_size (int): number of the best Members of the current Generation to be copy-pasted into the new one
        fit_fun (Callable): function passed to Members of the population and stored as a fit_fun attribute;
            returns a float value based on a Member's genome and is used to compare Members, which represents a better
            potential solution to a given problem.
        genome_gen (Callable): function which returns genome of a single Member, used for initial Generation (first
            current and accepted one) and for mutation.
        operators (list[tuple[Callable]]): list of operators (selection and crossover) combinations based on which
            new, rival Generations of children are to e created from parents in the current Generation in ach iteration.
        no_parents_pairs (int): the designated number of parent pairs for future Generations, e.g., if the initial
            population size is 1000 and no_parents_pairs = 200, there will be 2 * 200 = 400 children. By default it is
            equal to pop_size // 2.
        mutation_prob (float): 0.0 by default; probability of selecting a Member of a Generation to reset its genome
            with the genome_generator
        current_gen (Generation): Members constituting population inside the Genetic Algorithm in a given iteration. It
            is the last accepted Generation from the previous iteration or the initial Generation.
        workers (list[Process]): dynamical list containing processes from the multiprocessing package, meant to operate
            in parallel and either execute creating new Generations or evaluating them.
        manager (Manager): Manager ('master') synchronising access of multiple workers to a rival_gen proxy for dict.
        rival_gen_pool (DictProxy[int, Generation]): in the Parallel Genetic Algorithm multiple children Generations may
            be created based on the current Generation of parents, based on different selection and crossover operators.
            These Generations are rival to one another, because only one will be accepted as the best and treated as the
            current Generation in the next iteration. In the rival_gen DictProxy each of these rival Generations is
            stored with its integer id as a key and parallel processes (workers) may add Generations to it after
            acquiring acces through a manager's lock.
        accepted_gen_list (list[Generation]): the best of the rival Generations is added to a list of the accepted
            Generations and treated as the current Generation in the next iteration of the algorithm. If there is only
            one new, 'rival' Generation, then automatically it is appended to the accepted Generations list+.
        best_fit_history (list[float]): List the best Members' fitness values in each of the accepted Generation.
        args (dict): dictionary with argument required by the genome generator and all the selection and crossover
            operators to work.

    What the args dict should look like:
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
            and cM1, cM2, etc., are args of the Mth crossover operator.
    """
    pop_size: int
    no_generations: int
    elite_size: int
    fit_fun: Callable
    genome_gen: Callable
    operators: dict[int, tuple[Callable]]
    no_parents_pairs: int
    mutation_prob: float
    current_gen: Generation
    workers: list[Process] = []
    manager: Manager
    rival_gen_pool: DictProxy[int, Generation]
    accepted_gen_list: list[Generation]
    best_fit_history: list[float]
    args: dict
      
    def __zip_crossover_selection(self, selection_operators: list[Callable], crossover_operators: list[Callable]):
        """Creates a dict that combines pairs of elements from 'selection_operators' and 'crossover_operators' with
        an ID as key. For each index 'i', it adds tuples to the 'operators_combinations_dict' dict, each tuple
        containing 'selection_operator[i]' and 'crossover_operator[j]' for each index 'j' with a unique ID. This way
        there are tuples for all combinations of operators, accessible by workers working in parallel under their IDs
        as keys.

        Parameters:
            selection_operators (list[Callable]): list of functions which are selection operators
                for the Genetic Algorithm
            crossover_operators (list[Callable]): list of functions which are crossover operators
                for the Genetic Algorithm

        Returns:
            dict[int, tuple[Callable]]: dict of (Callable) operators tuples, each representing a combination of
            selection and crossover method for creating a new Generation.
        """
        operators_combinations_dict = {}
        combination_id = 0
        for i in range(len(selection_operators)):
            for j in range(len(crossover_operators)):
                operators_combinations_dict[combination_id] = (selection_operators[i], crossover_operators[j])
                combination_id += 1
        return operators_combinations_dict

    def __init__(self, initial_pop_size: int, number_of_generations: int, elite_size: int, args: dict,
                 fitness_function: Callable, genome_generator: Callable,
                 selection: list[Callable] | Callable, crossover: list[Callable] | Callable,
                 pool_size, no_parents_pairs=None, mutation_prob: float=0.0,
                 seed=None):  # TODO: put pool_size in the args dict for self.selection_args = args.get('selection') below
        """GeneticAlgorithm class constructor.

        Parameters:
            initial_pop_size (int): size of the population (each Generation)
            number_of_generations (int): how many consecutive accepted Generations are supposed to be created and
                evaluated
            elite_size (int): number of the best Members of the current Generation to be copy-pasted into the new one
            args (dict): arguments to be used in genome_generator & selection/crossover operators
            fitness_function (Callable): func passed to Members of the population and stored as a fit_fun attribute;
                returns a float value based on a member's genome
            genome_generator (Callable): func which returns genome of a single Member
            selection (list[Callable] | Callable): list of func from selection_operators.py for parent selection
            crossover (list[Callable] | Callable): list of func from crossover_operators.py for children creation
            no_parents_pairs (int): optional; is the designated number of parent pairs for future Generations,
                e.g., if the initial population size is 1000 and no_parents_pairs = 200,
                there will be 2 * 200 = 400 children
            mutation_prob (int): 0.0 by default; probability of selecting a Member of a Generation to reset its genome
                with the genome_generator
            seed (int | float | str | bytes | bytearray | None = None): optional; parameter 'a' for random.seed
        """
        self.pop_size = initial_pop_size
        self.no_generations = number_of_generations
        self.elite_size = elite_size

        # self.genome_generator_args = args.get('genome')
        self.args = args
        self.selection_args = args.get('selection')  # TODO: we should stick to using self.args
        self.crossover_args = args.get('crossover')  # TODO: we should stick to using self.args

        self.fit_fun = fitness_function
        self.mutation_prob = mutation_prob
        if seed is not None:
            random.seed(a=seed)  # useful for debugging

        self.manager = Manager()
        self.rival_gen_pool = self.manager.dict()
        self.accepted_gen_list = self.manager.list()
        self.best_fit_history = self.manager.list()
        self._best_solution = self.manager.dict()  # new dict for best solutions

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
        self.pool_size = pool_size  # will be redundant after the selection args are properly handled

    def _create_initial_generation(self):
        """Creating the first - initial - generation in this population."""
        global identification
        first_members = []
        for _ in range(self.pop_size):
            genes = self.genome_generator(self.args)
            first_members.append(Member(
                genome=genes,
                manager=self.manager,
                identification_number=identification,
                fitness_function=self.fit_fun)
            )
            identification += 1
        self.current_generation = Generation(
            manager=self.manager,
            generation_members=first_members,
            num_parents_pairs=self.no_parents_pairs,
            elite_size=self.elite_size,
            pool_size=self.pool_size
        )
        self.current_generation.evaluate()
        self.accepted_gen_list = [self.current_generation]
        self.best_fit_history = [self.current_generation.fitness_ranking[0].get('fitness value')]

    def best_solution(self):
        """Returns genome of Member with the highest fitness value with it's fitness value, from the current Generation.

        Returns:
            tuple[type[list | dict], float]: tuple of the genome list/dict of the best Member and it's float fit. value
        """
        index_of_best_member = self.accepted_gen_list[-1].fitness_ranking[0].get('index')
        best_member = list(self.accepted_gen_list[-1].members)[index_of_best_member]  # TODO BrokenPipeError: [WinError 232] Trwa zamykanie potoku
        best_genome = list(best_member.genome)
        best_fit_val = best_member.fit_val

        bf = (best_genome, best_fit_val)
        return bf

    def _choose_best_rival_generation(self):
        """This method selects one of the rival generations from the rival_gen dict, based on the highest max fitness
        value, to be accepted as a new current generation."""
        fitness_comparison = {}
        for id_of_rival, generation in self.rival_gen_pool.items():
            fitness_comparison[id_of_rival] = generation.fitness_ranking[0].get('fitness value')
        self.current_generation = self.rival_gen_pool.get(max(fitness_comparison, key=fitness_comparison.get))  # TODO: resolve ValueError: max() iterable argument is empty - why is it now after the manager and shared memory implementation???
        self.accepted_gen_list.append(self.current_generation)
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
                self.genome_generator(self.args),
                self.manager
            )

    def run(self):
        """This is the main method for an automated run of the Genetic Algorithm, supposed to be used right after this
        class' instance initialisation. It creates the initial Generation and then performs the `no_generations`
        iterations of creating new/rival Generations, choosing the best one and mutation, if necessary."""
        print(f"Creating the initial population.")
        self._create_initial_generation()

        # For testing:
        """
        for member in self.current_generation.members:
            print(member.fit_val)
        """

        operator_combinations_ids = list(self.operators.keys())

        with self.manager as ga_manager:
            for _ in range(self.no_generations):
                """Rival generations are created based on accessible combinations of selection and crossover
                operators with different processes in parallel:"""
                print(f"Creating rival generations")
                for combination_id in operator_combinations_ids:
                    new_worker = Process(
                        target=_create_rival_generation(
                            ga_manager,
                            combination_id,  # id
                            self.operators.get(combination_id)[0],  # selection
                            self.operators.get(combination_id)[1],  # crossover
                            self.args.get('crossover'),  # crossover_args
                            self.current_generation,  # parent_generation
                            self.fit_fun,  # fitness_function
                            self.rival_gen_pool  # generation_pool
                        )
                    )
                    new_worker.start()
                    self.workers.append(new_worker)

                """After work done, processes are collected and their list reset for new batch of workers:"""
                for worker in self.workers:
                    worker.join()

                """
                #Just for testing:
                new_members = self.rival_gen_pool.get(0).members
                for member in new_members:
                    print(member.genome)
                """

                self.workers = []

                """For fitness evaluation as many workers as the CPU allows are created. All members are distributed
                 between these processes to be evaluated:"""
                no_workers = cpu_count()
                no_members = self.pop_size * len(self.rival_gen_pool)

                members_per_worker = no_members / no_workers
                if members_per_worker <= 1:
                    no_workers = int(no_members)

                indexes_batches = split_indexes(num_members=no_members, num_workers=no_workers)

                print(f"Evaluating fitness of the rival generations. It is iteration number {_}")

                for index in range(no_workers):
                    indexes_of_members_to_evaluate = indexes_batches[index]
                    # print(f"For step={index} we have indexes={indexes_of_members_to_evaluate}")
                    new_worker = Process(
                        target=_evaluate_members,  # now there's a problem with the function, not with multiprocessing
                        args=(
                            self.rival_gen_pool,
                            indexes_of_members_to_evaluate,
                            self.pop_size
                        )
                    )
                    new_worker.start()
                    self.workers.append(new_worker)

                """After evaluation, processes are again joined:"""
                for worker in self.workers:
                    worker.join()

                """
                # Just for testing:
                new_members = self.rival_gen_pool.get(0).members
                for member in new_members:
                    print(member.fit_val)
                """

                """Reset workers"""
                self.workers = []

                """Rebuild fitness ranking for each Generation"""
                for gen_id, generation in self.rival_gen_pool.items():
                    generation.fitness_ranking = []
                    for i, member in enumerate(generation.members):
                        if member.fit_val is None:
                            print(f"Skipping member {i} with fit fun. {member.fit_fun} in Generation {gen_id} due to "
                                  f"None fitness!")
                            # print(f"When computing fitness manually we get {member.evaluate()}!")
                            print(member)
                            continue  # <-- skip if fitness is None
                        generation.fitness_ranking.append({'index': i, 'fitness value': member.fit_val})
                    if generation.fitness_ranking:
                        generation.fitness_ranking.sort(key=sort_dict_by_fit, reverse=True)
                    else:
                        print(f"Warning: Generation {gen_id} has no valid members to rank!")
                    self.rival_gen_pool[gen_id] = generation  # reassign updated generation

                """Last stage of each iteration is to choose the next accepted Generation and mutate it:"""
                self._choose_best_rival_generation()
                # self.mutate()  # mutation in here introduces Members with their fitness not evaluated! TODO: make sure mutation is applied to each rival generation after children are created and before fitness is evaluated

    def fitness_plot(self):  # TODO: finish with an optional argument for using plotly or matplotlib
        """Method for plotting fitness values history of the best Members from each accepted Generation."""
        pass
