"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
Script is distributed under the license: https://github.com/gnypit/pyqkd/blob/main/LICENSE
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import \
    Callable  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
from multiprocessing import Pool, cpu_count

import numpy as np

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


def uniform_gene_generator(
        ga_args: dict):  # TODO: just take a tuple at the start, genome args will be passed directly, not the whole args dict
    """Simple function for generating a sample of a given length from the gene_space with a uniform probability.

    Parameters:
        ga_args (dict): This dictionary is stored within the GeneticAlgorithm class and contains info about args to be
            used by either genome generator, crossover operators or selection operators. For the genome generation,
            args are stored under the key 'genome'. There should be gene space and length of chromosomes (their genome).

    Returns:
         ndarray: A numpy array containing genes randomised from the gene space. There should be at least two genes
            in each chromosome, so this function should never return a single int, str, etc.
    """
    gene_space, length = ga_args.get('genome')
    return list(np.random.choice(gene_space, length))


class ChromosomeInterface(ABC):
    """Abstract class representing chromosomes, the most fundamental objects in genetic algorithms."""

    @abstractmethod
    def evaluate(self, fitness_function: Callable = None):
        pass

    @abstractmethod
    def change_genes(self, new_genes: type[list | dict]):
        pass


class Chromosome(ChromosomeInterface):
    """Simple implementation of chromosomes, the most fundamental objects in genetic algorithms.

    Apart from genes, in this implementation of the Genetic Algorithm, the Chromosome class also stores the fitness
    function and value. This allows self-evaluation of each chromosome.

    Attributes:
        fit_val (float): Fitness value of the chromosome. None, by default, stores a float number once the chromosome
            is evaluated.
        genome (type[ListProxy | DictProxy]): Either a list or a dict, in shared memory, with genes of this chromosome.
        fit_fun (Callable): Fitness function used for computing fitness value based on chromosome's genes.
    """
    fit_val: float = None
    genome: type[list | dict]
    fit_fun: Callable

    def __init__(self, genome: type[list | dict], fitness_function: Callable = None):
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

    def evaluate(self, fitness_function: Callable = None):
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

    def change_genes(self, new_genes: type[list | dict]):
        """Method meant to be used when mutation occurs, to modify the genes in an already created chromosome.

        Manager is only passed on for creating proxies for list/dict, it is not saved in Chromosome directly - it will
        be saved in outer scope.

        Parameters:
            new_genes (type[list | dict]): New genome to be stored by the chromosome.
        """
        self.genome = new_genes


class Member(Chromosome):
    """This class is a child of the 'Chromosome' class and is designated to store a unique ID, enabling tracking a
    genealogical tree of chromosomes in a population of a genetic algorithm.

    Attributes:
        id (int): A unique identification number of this member in the particular run of a genetic algorithm, created
            based on a global variable. It is meant for backtracking of members' genealogical tree.
        parents_id (list): It's a list with IDs of the parents (from previous generations in the GA) of this member
    """
    id: int
    parents_id: list = []

    def __init__(self, genome: type[list | dict], identification_number: int, fitness_function: Callable = None):
        """Apart from what 'Chromosome' class constructor needs, here identification number should be passed.

        Parameters:
            genome (type[list | dict]): Either a dict with genes as values and names provided by the User as keys,
                or simply a list of genes.
            identification_number (int): An ID to be created based on the global variable, for backtracking a
                genealogical tree of all members across different generations in a particular run of the GA.
            fitness_function (Callable=None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Can be passed later, thus it is None by default.
        """
        super().__init__(genome=genome, fitness_function=fitness_function)
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


class Generation:  # TODO: add diversity measures
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
        members (ListProxy[Member]): list of Members in shared memory; chromosomes of the generation with their and
            parents' IDs. Has to be accessible form multiple processes evaluating the Members in parallel.
        # TODO: add genome size for per-gene mutation
        num_parents_pairs (int): number of pairs of Members can be parents, e.g., 20 pairs means 40 mating chromosomes.
        elite_size (int): number of Members to be copy-pasted directly into a new Generation.
        size (int): number of Members in the generation.
        fitness_ranking (list[dict]): dicts in this list have the index of a Member in the Generation as keys and its
            fitness value as values.
    """
    members: list[Member]  # this needs to be accessible from multiple processes running in parallel
    genome_size: int
    num_parents_pairs: int
    elite_size: int
    size: int
    fitness_ranking: list[dict]

    def __init__(self, generation_members: list[Member], num_parents_pairs: int, elite_size: int):
        """Constructor for any Generation inside the GeneticAlgorithm.

        Parameters:
            generation_members (list[Member]): list of Members, in shared memory, to be put in this Generation.
            num_parents_pairs (int): number of Members' pairs that can be parents.
            elite_size (int): number of Members to be copy-pasted directly into a new Generation.
        """
        self.members = generation_members
        self.genome_size = len(generation_members[0].genome)
        self.num_parents_pairs = num_parents_pairs
        self.elite_size = elite_size
        self.size = len(generation_members)
        self.fitness_ranking = []

    def mutate_member(self, prob: float):
        """Method for applying a basic mutation operator to this generation - it randomly chooses a member to have their
        genome rested with the genome generator based on passed mutation probability `prob`."""
        pass

    def evaluate(self):
        """This method calls the 'evaluate' method on each Member of this Generation."""
        for i in range(self.size):
            self.members[i].evaluate()

    def create_fitness_ranking(self, reverse=True):
        """This method creates and then sorts the fitness ranking of the Members; 'reverse' means sorting will be
        performed from max fitness value to min.

        Parameters:
            reverse (Bool=True, optional): parameter which decided whether the fitness ranking should be sorted in
                ascending order of fitness values (reverse=False) or in descending order (reverse=True), which is
                the default.
        """
        for i in range(self.size):
            self.fitness_ranking.append({'index': i, 'fitness value': self.members[i].fit_val})

        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)


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
        genome_gen (Callable): function which returns the genome of a single Member, used for initial Generation (first
            current and accepted one) and for mutation.
        operators (list[tuple[Callable]]): list of operators (selection and crossover) combinations based on which
            new, rival Generations of children are to be created from parents in the current Generation in each
            iteration.
        no_parents_pairs (int): the designated number of parent pairs for future Generations, e.g., if the initial
            population size is 1000 and no_parents_pairs = 200, there will be 2 * 200 = 400 children. By default, it is
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
            These Generations are rival to one another because only one will be accepted as the best and treated as the
            current Generation in the next iteration. In the rival_gen DictProxy each of these rival Generations is
            stored with its integer id as a key, and parallel processes (workers) may add Generations to it after
            acquiring access through a manager's lock.
        members_to_evaluate (list[Member]): TODO: update this docstring
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
    rival_gen_pool: dict[int, Generation]
    accepted_gen_list: list[Generation]
    best_fit_history: list[float]
    parallel_workers: int | None
    args: dict

    def __zip_crossover_selection(self, selection_operators: list[Callable], crossover_operators: list[Callable]):
        """Creates a dict that combines pairs of elements from 'selection_operators' and 'crossover_operators' with
        an ID as a key. For each index 'i', it adds tuples to the 'operators_combinations_dict' dict, each tuple
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
                 no_parents_pairs=None, mutation_prob: float = 0.0,
                 seed=None, parallel_workers: int | None = None):
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
            parallel_workers (int | None): optional; maximum number of worker processes reused by the parallel run.
        """
        self.pop_size = initial_pop_size
        self.no_generations = number_of_generations
        self.elite_size = elite_size

        # self.genome_generator_args = args.get('genome')
        self.args = args

        self.fit_fun = fitness_function
        self.mutation_prob = mutation_prob
        self.parallel_workers = parallel_workers
        if seed is not None:
            random.seed(a=seed)  # useful for debugging

        self.rival_gen_pool = {}

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

    def _get_parallel_worker_count(self, no_members: int) -> int:
        """Returns the number of long-lived worker processes to use during a run."""
        if self.parallel_workers is not None:
            return max(1, min(self.parallel_workers, no_members))

        return max(1, min(cpu_count(), no_members))

    def _create_initial_generation(self):
        """Creating the first - initial - generation in this population."""
        global identification
        first_members = []

        for _ in range(self.pop_size):
            genes = self.genome_generator(self.args)
            first_members.append(Member(
                genome=genes,
                identification_number=identification,
                fitness_function=self.fit_fun)
            )
            identification += 1

        self.current_generation = Generation(
            generation_members=first_members,
            num_parents_pairs=self.no_parents_pairs,
            elite_size=self.elite_size
        )
        self.current_generation.evaluate()
        self.current_generation.create_fitness_ranking()
        self.accepted_gen_list = [self.current_generation]
        self.best_fit_history = [self.current_generation.fitness_ranking[0].get('fitness value')]

    @staticmethod
    def _create_members_for_rival_generation(
            combination_id: int,
            parent_generation: Generation,
            fitness_function: Callable,
            operators: dict,
            args: dict,
            first_identification_number: int
    ) -> tuple[int, list[Member]]:
        """Create and return one rival generation's members.

        Workers return plain Python objects instead of writing through a multiprocessing.Manager proxy. On Windows,
        high-frequency Manager proxy traffic uses named pipes and can fail with WinError 231 when the pipe server is
        saturated.
        """
        selection, crossover = operators.get(combination_id)
        selection_args = args.get("selection") if isinstance(args, dict) and "selection" in args else args
        crossover_args = args.get("crossover") if isinstance(args, dict) and "crossover" in args else None

        try:
            parents_in_order = selection(parent_generation, selection_args)
        except TypeError as e:
            member_details = "\n".join(
                f"Parent member {member} has fitness function {member.fit_fun}."
                for member in parent_generation.members
            )
            raise TypeError(
                f"Selection operator failed for operator combination {combination_id}: {e}\n{member_details}"
            ) from e

        new_members = []
        next_identification = first_identification_number
        for index in range(parent_generation.num_parents_pairs):
            child1_genome, child2_genome = crossover(
                parents_in_order[2 * index].genome,
                parents_in_order[2 * index + 1].genome,
                crossover_args
            )
            new_members.append(Member(
                genome=child1_genome,
                identification_number=next_identification,
                fitness_function=fitness_function)
            )
            new_members.append(Member(
                genome=child2_genome,
                identification_number=next_identification + 1,
                fitness_function=fitness_function)
            )
            next_identification += 2

        return combination_id, new_members

    @staticmethod
    def _evaluate_member_batch(
            combination_id: int,
            indexed_members: list[tuple[int, Member]]
    ) -> tuple[int, list[tuple[int, Member]]]:
        """Evaluate a batch of members and return their original indexes with the updated objects."""
        evaluated_members = []
        for member_index, member in indexed_members:
            member.evaluate()
            evaluated_members.append((member_index, member))
        return combination_id, evaluated_members

    def best_solution(self):
        """Returns the genome of Member with the highest fitness value with its fitness value,
        from the current Generation.

        Returns:
            tuple[type[list | dict], float]: tuple of the genome list/dict of the best Member and its float fit. value
        """
        index_of_best_member = self.current_generation.fitness_ranking[0].get('index')
        best_member = self.current_generation.members[index_of_best_member]
        best_genome = best_member.genome
        best_fit_val = best_member.fit_val

        bf = (best_genome, best_fit_val)
        return bf

    def _choose_best_rival_generation(self):
        """This method selects one of the rival generations from the rival_gen dict, based on the highest max fitness
        value, to be accepted as a new current generation."""
        fitness_comparison = {}
        for id_of_rival, generation in self.rival_gen_pool.items():
            fitness_comparison[id_of_rival] = generation.fitness_ranking[0].get('fitness value')
        self.current_generation = self.rival_gen_pool.get(max(fitness_comparison, key=fitness_comparison.get))
        self.accepted_gen_list.append(self.current_generation)
        self.best_fit_history.append(self.current_generation.fitness_ranking[0].get('fitness value'))

    def mutate(self, mutation_type: str = "member"):  # TODO: add adaptive mutation (diversity measures first?)
        """Applies mutation to the current generation.
        
        Mutation types:
        - "member": Entire genome reset
        - "gene": Individual genes replaced
        - "gaussian": Numeric genes perturbed (Gaussian noise)
        - "swap": Two random genes swap positions
        
        Parameters:
            mutation_type (str): Type of mutation to apply
        """
        if self.mutation_prob == 0.0:
            return

        match mutation_type:
            case "member":
                self._mutate_members()
            case "gene":
                self._mutate_genes()
            case "gaussian":
                self._mutate_gaussian()
            case "swap":
                self._mutate_swap()
            case _:
                print(f"Warning: Unknown mutation type '{mutation_type}'. Using 'member' by default.")
                self._mutate_members()

    def _mutate_members(self):
        """In this case mutation probability is the probability of 'resetting' a member of the current generation, i.e.,
        generating its genome from scratch. For optimisation purposes instead of a loop over the whole generation, the
        number of members to be mutated is calculated, and then a list of member indexes in the current generation to be
        mutated is generated pseudo-randomly.
        """
        number_of_mutations = np.floor(self.mutation_prob * (self.current_generation.size - self.elite_size))

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
            self.current_generation.members[index].change_genes(self.genome_generator(self.args))  # self.manager
            self.current_generation.members[index].evaluate()

    def _mutate_genes(self):
        """Mutates individual genes across members of the current generation.

        Uses mutation_prob to select random gene positions across all non-elite members.
        For each selected gene index, generates a complete temporary member using genome_generator,
        then takes the corresponding gene from it to replace the original gene.
        """
        non_elite_members_count = self.current_generation.size - self.elite_size
        total_available_genes = non_elite_members_count * self.current_generation.genome_size

        number_of_mutations = int(np.floor(self.mutation_prob * total_available_genes))

        if number_of_mutations == 0:
            return

        # Select random gene indexes across all non-elite members and genes
        gene_indexes = random.sample(range(total_available_genes), number_of_mutations)

        affected_members = set()

        for gene_index in gene_indexes:
            # Calculate which member and which gene position within that member
            member_index = gene_index // self.current_generation.genome_size
            gene_position = gene_index % self.current_generation.genome_size

            # Generate a complete temporary genome
            temp_genome = self.genome_generator(self.args)

            # Replace the selected gene with the one from temporary genome
            self.current_generation.members[member_index].genome[gene_position] = temp_genome[gene_position]
            affected_members.add(member_index)

        # Re-evaluate all members that had genes mutated
        for member_index in affected_members:
            self.current_generation.members[member_index].evaluate()

    def _mutate_gaussian(self):
        """Gaussian mutation: adds random noise to numeric genes only."""
        non_elite_members_count = self.current_generation.size - self.elite_size
        total_available_genes = non_elite_members_count * self.current_generation.genome_size

        number_of_mutations = int(np.floor(self.mutation_prob * total_available_genes))

        if number_of_mutations == 0:
            return

        gene_indexes = random.sample(range(total_available_genes), number_of_mutations)

        if isinstance(self.genome_spec, dict):
            spec_list = list(self.genome_spec.values())
        else:
            spec_list = self.genome_spec

        affected_members = set()

        for gene_index in gene_indexes:
            member_index = gene_index // self.current_generation.genome_size
            gene_position = gene_index % self.current_generation.genome_size

            gene_spec = spec_list[gene_position]

            # Only apply Gaussian mutation to numeric genes (tuples)
            if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
                current_value = self.current_generation.members[member_index].genome[gene_position]
                # Add Gaussian noise (std = 5% of range)
                range_size = gene_spec[1] - gene_spec[0]
                noise = np.random.normal(0, range_size * 0.05)
                new_value = np.clip(current_value + noise, gene_spec[0], gene_spec[1])

                self.current_generation.members[member_index].genome[gene_position] = new_value
                affected_members.add(member_index)

        for member_index in affected_members:
            self.current_generation.members[member_index].evaluate()

    def _mutate_swap(self):
        """Swap mutation: randomly swaps two genes within a Member's genome."""
        non_elite_members_count = self.current_generation.size - self.elite_size
        number_of_mutations = int(np.floor(self.mutation_prob * non_elite_members_count))

        if number_of_mutations == 0:
            return

        member_indexes = random.sample(range(non_elite_members_count), number_of_mutations)

        for member_index in member_indexes:
            # Randomly select two gene positions to swap
            pos1, pos2 = random.sample(range(self.current_generation.genome_size), 2)

            # Swap genes
            genome = self.current_generation.members[member_index].genome
            genome[pos1], genome[pos2] = genome[pos2], genome[pos1]

            # Re-evaluate
            self.current_generation.members[member_index].evaluate()

    def run(self):
        """This is the main method for an automated run of the Genetic Algorithm, supposed to be used right after this
        class' instance initialisation. It creates the initial Generation and then performs the `no_generations`
        iterations of creating new/rival Generations, choosing the best one and mutation, if necessary."""
        global identification
        # print(f"Creating the initial population.")
        self._create_initial_generation()

        operator_combinations_ids = list(self.operators.keys())
        members_per_rival = 2 * self.no_parents_pairs
        no_members = members_per_rival * len(operator_combinations_ids)
        no_workers = self._get_parallel_worker_count(no_members=no_members)

        with Pool(processes=no_workers) as worker_pool:
            for iteration in range(self.no_generations):
                """Rival generations are created based on accessible combinations of selection and crossover
                operators with different processes in parallel:"""
                first_member_id = identification
                creation_jobs = [
                    (
                        combination_id,
                        self.current_generation,
                        self.fit_fun,
                        self.operators,
                        self.args,
                        first_member_id + offset * members_per_rival
                    )
                    for offset, combination_id in enumerate(operator_combinations_ids)
                ]
                created_rivals = worker_pool.starmap(
                    GeneticAlgorithm._create_members_for_rival_generation,
                    creation_jobs
                )
                identification += no_members
                rival_members_container = dict(created_rivals)

                """For fitness evaluation, all members are distributed between the already-running workers."""
                evaluation_jobs = []
                for combination_id, members in rival_members_container.items():
                    indexes_batches = split_indexes(num_members=len(members), num_workers=no_workers)
                    for indexes_of_members_to_evaluate in indexes_batches:
                        if indexes_of_members_to_evaluate:
                            evaluation_jobs.append((
                                combination_id,
                                [
                                    (member_index, members[member_index])
                                    for member_index in indexes_of_members_to_evaluate
                                ]
                            ))

                # print(f"Evaluating fitness of the rival generations. It is iteration number {iteration}")  # TODO: change from printing to logging
                evaluated_batches = worker_pool.starmap(
                    GeneticAlgorithm._evaluate_member_batch,
                    evaluation_jobs
                )
                for combination_id, evaluated_members in evaluated_batches:
                    members = rival_members_container[combination_id]
                    for member_index, member in evaluated_members:
                        members[member_index] = member

                """Build rival Generations out of members and compose their respective fitness rankings"""
                self.rival_gen_pool = {}
                for combination_id in operator_combinations_ids:
                    new_rival_generation = Generation(
                        generation_members=rival_members_container[combination_id],
                        num_parents_pairs=self.current_generation.num_parents_pairs,
                        elite_size=self.elite_size
                    )
                    new_rival_generation.create_fitness_ranking()
                    self.rival_gen_pool[combination_id] = new_rival_generation

                """Last stage of each iteration is to choose the next accepted Generation and mutate it:"""
                self._choose_best_rival_generation()
                # print(self.best_solution())
                # print(self.current_generation.members)
                self.mutate(mutation_type=self.args.get('mutation'))
                self.current_generation.fitness_ranking = []
                self.current_generation.create_fitness_ranking()
                self.best_fit_history[-1] = self.current_generation.fitness_ranking[0].get('fitness value')

    def fitness_plot(self):  # TODO: finish with an optional argument for using plotly or matplotlib
        """Method for plotting fitness values history of the best Members from each accepted Generation."""
        pass
