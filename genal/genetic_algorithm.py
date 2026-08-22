"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
Script is distributed under the license: https://github.com/gnypit/pyqkd/blob/main/LICENSE
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import \
    Callable  # https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
from copy import deepcopy
from inspect import signature
from math import ceil
from multiprocessing import Pool, cpu_count
from typing import Literal

import numpy as np

"""Global variable to hold IDs of chromosomes for backtracking"""
identification = 0

"""Read-only configuration installed once in each process-pool worker.

The _worker_operators is a dictionary with combinations of selection and crossover operators as values. Each combination
has it's ID as the key. 
"""
_worker_fitness_function: Callable | None = None
_worker_operators: dict[int, tuple[Callable, Callable]] | None = None
_worker_args: dict | None = None
_worker_custom_mutation_operator: Callable | None = None


def _initialize_parallel_worker(
        fitness_function: Callable,
        operators: dict,
        args: dict,
        custom_mutation_operator: Callable | None
):
    """Cache immutable GA configuration in a worker instead of sending it with every task."""
    global _worker_fitness_function, _worker_operators, _worker_args, _worker_custom_mutation_operator
    _worker_fitness_function = fitness_function
    _worker_operators = operators
    _worker_args = args
    _worker_custom_mutation_operator = custom_mutation_operator


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


def uniform_gene_generator(ga_args: dict, gene_position: int | None = None):
    """Uniformly generate either a complete genome or one gene for mutation.

    Parameters:
        ga_args (dict): This dictionary is stored within the GeneticAlgorithm class and contains info about args to be
            used by either genome generator, crossover operators or selection operators. For the genome generation,
            args are stored under the key ``"genome"`` as ``(gene_space, genome_length)``. ``gene_space`` may be a
            shared sequence for every position or a dictionary mapping each position to its own sequence.
        gene_position (int | None): Position whose replacement gene should be generated. If omitted, a complete genome
            is generated for initial-population creation and whole-member mutation.

    Returns:
        list | object: A complete genome when ``gene_position`` is ``None``; otherwise one gene sampled from the space
            associated with that position.

    Raises:
        IndexError: if ``gene_position`` is outside the configured genome.
    """
    gene_space, length = ga_args["genome"]

    if gene_position is not None:
        if not 0 <= gene_position < length:
            raise IndexError(
                f"gene_position must be between 0 and {length - 1}; got {gene_position}."
            )
        position_space = gene_space[gene_position] if isinstance(gene_space, dict) else gene_space
        return random.choice(position_space)

    if isinstance(gene_space, dict):
        return [random.choice(gene_space[position]) for position in range(length)]
    return list(np.random.choice(gene_space, length))


def uniform_gene_generator_for_position(ga_args: dict, gene_position: int):
    """Uniform gene generator for a single gene mutation operator. Designed for the labyrinth test."""
    gene_space, _ = ga_args["genome"]

    if isinstance(gene_space, dict):
        position_space = gene_space[gene_position]
        return random.choice(position_space)

    return random.choice(gene_space)


class ChromosomeInterface(ABC):
    """Abstract class representing chromosomes, the most fundamental objects in genetic algorithms."""

    @abstractmethod
    def evaluate(self, fitness_function: Callable | None):
        pass

    @abstractmethod
    def change_genes(self, new_genes: list | dict):
        pass


class Chromosome(ChromosomeInterface):
    """Simple implementation of chromosomes, the most fundamental objects in genetic algorithms.

    Apart from genes, in this implementation of the Genetic Algorithm, the Chromosome class also stores the fitness
    function and value. This allows self-evaluation of each chromosome.

    Attributes:
        fit_val (float): Fitness value of the chromosome. None, by default, stores a float number once the chromosome
            is evaluated.
        genome (list | dict): Either a list or a dict with genes of this chromosome.
        fit_fun (Callable | None): Fitness function used for computing fitness value based on chromosome's genes.
    """
    fit_val: float
    genome: list | dict
    fit_fun: Callable | None = None

    def __init__(self, genome: list | dict, fitness_function: Callable | None = None):
        """Constructor of the Chromosome class.

        Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc. However, this class is limited to storage of genes, fitness function and value, and to fitness
        evaluation.

        Parameters:
            genome (list | dict): Either a dict with genes as values and names provided by the User as keys, or simply
                a list of genes.
            fitness_function (Callable | None): Optional; callable fitness function provided by the User, which computes
                fitness value based on genome. Can be passed later, thus it is None by default.
        """
        self.genome = genome
        self.fit_fun = fitness_function  # special variable

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return (f"{type(self).__name__}(genes={self.genome}, fitness function={self.fit_fun}, "
                f"fitness value={self.fit_val})")

    def evaluate(self, fitness_function: Callable | None = None) -> float:
        """Method for applying fitness function to this chromosome (it's genes, to be precise).

        If the fitness function was passed on in the constructor of this class, it doesn't have to be provided as an
        argument of this method. Fitness value returned by this method is also remembered in an attribute of this class.
        If no fitness function is provided, the fitness value assigned by default is 0.0.

        Parameters:
            fitness_function (Callable | None = None): Optional; callable fitness function provided by the User, which 
                computes fitness value based on genome. Could have already been provided in the constructor, thus it is 
                None by default.

        Returns:
            float: Fitness value as a float number.
        """
        try:
            if self.fit_fun is None:
                self.fit_fun = fitness_function

            if self.fit_fun is None:
                print(f"Warning: no fitness function available for {self}")
                self.fit_val = 0.0
            else:
                self.fit_val = _evaluate_genome(self.genome, self.fit_fun)
        except Exception as e:
            print(f"Error evaluating member {self}: {e}")
            self.fit_val = 0.0

        return self.fit_val

    def change_genes(self, new_genes: list | dict):
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

    def __init__(self, genome: list | dict, identification_number: int, fitness_function: Callable | None = None):
        """Apart from what 'Chromosome' class constructor needs, here identification number should be passed.

        Parameters:
            genome (list | dict): Either a dict with genes as values and names provided by the User as keys,
                or simply a list of genes.
            identification_number (int): An ID to be created based on the global variable, for backtracking a
                genealogical tree of all members across different generations in a particular run of the GA.
            fitness_function (Callable | None): Optional; callable fitness function provided by the User, which computes
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
        members (list[Member]): Parent-process-owned list of chromosomes and their parent IDs. Generations are copied
            to pool workers when selection and crossover work is dispatched; no manager proxy is used.
        # TODO: add genome size for per-gene mutation
        num_parents_pairs (int): number of pairs of Members can be parents, e.g., 20 pairs means 40 mating chromosomes.
        elite_size (int): number of Members to be copy-pasted directly into a new Generation.
        size (int): number of Members in the generation.
        fitness_ranking (list[dict]): dicts in this list have the index of a Member in the Generation as keys and its
            fitness value as values.
    """
    members: list[Member]
    genome_size: int
    num_parents_pairs: int
    elite_size: int
    size: int
    fitness_ranking: list[dict]

    def __init__(self, generation_members: list[Member], num_parents_pairs: int, elite_size: int):
        """Constructor for any Generation inside the GeneticAlgorithm.

        Parameters:
            generation_members (list[Member]): parent-process-owned list of Members to put in this Generation.
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
    """Container for a process-pool-based hierarchical parallel genetic algorithm.

    A persistent process pool performs both rival-generation creation and fitness evaluation. Immutable callables and
    operator configuration are installed once in every worker by the pool initialiser. Fitness inputs are dispatched
    in chunks to reduce inter-process communication overhead for inexpensive fitness functions. Workers return ordinary
    Python objects; the parent process owns all accepted and rival Generations.

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
        current_generation (Generation): Members constituting population inside the Genetic Algorithm in a given
            iteration. It is the last accepted Generation from the previous iteration or the initial Generation.
        parallel_workers (int | None): positive maximum number of persistent worker processes. If omitted, the worker
            count is limited by the CPU count and the amount of work.
        rival_gen_pool (dict[int, Generation]): parent-process-owned rival Generations keyed by operator-combination ID.
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
    gene_generator: Callable[[dict, int], object] | None = None
    operators: dict[int, tuple[Callable]]
    no_parents_pairs: int
    mutation_prob: float
    current_generation: Generation
    rival_gen_pool: dict[int, Generation]
    accepted_gen_list: list[Generation]
    best_fit_history: list[float]
    parallel_workers: int | None
    creation_parallelism: Literal["auto", "local", "operators", "parent_pairs"]
    args: dict

    @staticmethod
    def __zip_crossover_selection(selection_operators: list[Callable], crossover_operators: list[Callable]):
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
                 seed=None, parallel_workers: int | None = None,
                 creation_parallelism: Literal["auto", "local", "operators", "parent_pairs"] = "auto",
                 custom_mutation_operator: Callable | None = None):
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
            parallel_workers (int | None): optional; positive maximum number of worker processes reused by the parallel
                run. ``None`` selects up to ``cpu_count()`` workers. Zero and negative values are rejected.
            creation_parallelism (str): strategy for selection, crossover, and child construction. ``"local"`` keeps
                creation in the parent process; ``"operators"`` submits one task per selection/crossover combination;
                ``"parent_pairs"`` selects parents locally and distributes batches of pairs; ``"auto"`` uses operator
                tasks when multiple rival combinations exist and otherwise avoids IPC for lightweight local creation.
            custom_mutation_operator (Callable | None): optional expensive mutation callable accepting ``(genome,
                args)``. It may return a replacement genome or mutate its input in place and return ``None``. Selecting
                mutation type ``"custom"`` runs this operator in pool workers and fuses mutation with evaluation.

        Raises:
            TypeError: if ``parallel_workers`` is not an integer or ``None``.
            ValueError: if ``parallel_workers`` is zero or negative, or the creation strategy is unsupported.
        """
        self.pop_size = initial_pop_size
        self.no_generations = number_of_generations
        self.elite_size = elite_size

        # self.genome_generator_args = args.get('genome')
        self.args = args

        self.fit_fun = fitness_function
        self.mutation_prob = mutation_prob
        self.custom_mutation_operator = custom_mutation_operator
        if parallel_workers is not None:
            if isinstance(parallel_workers, bool) or not isinstance(parallel_workers, int):
                raise TypeError("parallel_workers must be a positive integer or None.")
            if parallel_workers <= 0:
                raise ValueError("parallel_workers must be greater than zero.")
        self.parallel_workers = parallel_workers
        valid_creation_modes = {"auto", "local", "operators", "parent_pairs"}
        if creation_parallelism not in valid_creation_modes:
            raise ValueError(
                f"creation_parallelism must be one of {sorted(valid_creation_modes)}; got {creation_parallelism!r}."
            )
        self.creation_parallelism = creation_parallelism
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
        try:
            signature(genome_generator).bind(self.args, gene_position=0)
            self.genome_generator_supports_single_gene = True
        except (TypeError, ValueError):
            self.genome_generator_supports_single_gene = False

        """Based on lists of (callable) function selected by the User from selection_operators.py 
        and crossover_operators.py, a more general dict is created with all the possible combinations of the operators.
        """
        if type(selection) is not list:
            selection = [selection]
        if type(crossover) is not list:
            crossover = [crossover]

        self.operators = self.__zip_crossover_selection(selection_operators=selection, crossover_operators=crossover)

    def _get_parallel_worker_count(self, no_members: int) -> int:
        """Return the bounded number of persistent worker processes to use during a run."""
        if self.parallel_workers is not None:
            return max(1, min(self.parallel_workers, no_members))

        return max(1, min(cpu_count(), no_members))

    @staticmethod
    def _create_fitness_batches(
            members: dict[int, list | dict], no_workers: int
    ) -> list[list[tuple[int, list | dict]]]:
        """Split fitness work into enough batches for load balancing without per-member IPC calls.

        Returns: An list, which each entry is a list of tuples with members' IDs and their genomes to be evaluated
            in a given batch.
        """
        jobs = list(members.items())
        batch_size = max(1, ceil(len(jobs) / (no_workers * 4)))
        return [jobs[index:index + batch_size] for index in range(0, len(jobs), batch_size)]

    def _resolve_creation_parallelism(self, no_workers: int) -> str:
        """Resolve ``auto`` without assuming that arbitrary user crossover functions are expensive."""
        if self.creation_parallelism != "auto":
            return self.creation_parallelism
        if no_workers > 1 and len(self.operators) > 1:
            return "operators"
        return "local"

    @staticmethod
    def _select_parent_genome_pairs(
            combination_id: int,
            parent_generation: Generation,
            operators: dict[int, tuple[Callable, Callable]],
            args: dict
    ) -> list[tuple[list | dict, list | dict]]:
        """Select parents once and normalize supported selection results to raw-genome pairs."""
        selection, _ = operators[combination_id]
        selection_args = args.get("selection") if isinstance(args, dict) and "selection" in args else args

        try:
            selected_parents = selection(parent_generation, selection_args)
        except TypeError as error:
            member_details = "\n".join(
                f"Parent member {member} has fitness function {member.fit_fun}."
                for member in parent_generation.members
            )
            raise TypeError(
                f"Selection operator failed for operator combination {combination_id}: {error}\n{member_details}"
            ) from error

        if selected_parents and isinstance(selected_parents[0], dict):
            parent_pairs = [
                (parents["parent1"].genome, parents["parent2"].genome)
                for parents in selected_parents
            ]
        else:
            parent_pairs = [
                (selected_parents[2 * index].genome, selected_parents[2 * index + 1].genome)
                for index in range(parent_generation.num_parents_pairs)
            ]

        if len(parent_pairs) < parent_generation.num_parents_pairs:
            raise ValueError(
                f"Selection operator for combination {combination_id} returned {len(parent_pairs)} parent pairs; "
                f"{parent_generation.num_parents_pairs} are required."
            )
        return parent_pairs[:parent_generation.num_parents_pairs]

    @staticmethod
    def _build_members_from_parent_pairs(
            parent_pairs: list[tuple[list | dict, list | dict]],
            crossover: Callable,
            crossover_args,
            fitness_function: Callable,
            first_identification_number: int,
            first_pair_index: int = 0
    ) -> list[Member]:
        """Cross parent genomes and assign stable IDs independent of worker completion order."""
        new_members = []
        for local_pair_index, (parent1_genome, parent2_genome) in enumerate(parent_pairs):
            pair_index = first_pair_index + local_pair_index
            child1_genome, child2_genome = crossover(parent1_genome, parent2_genome, crossover_args)
            child1_id = first_identification_number + 2 * pair_index
            new_members.extend([
                Member(child1_genome, child1_id, fitness_function),
                Member(child2_genome, child1_id + 1, fitness_function),
            ])
            # TODO: Record both selected parent IDs on each child for genealogy tracking.
        return new_members

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
            first_identification_number: int
    ) -> tuple[int, list[Member]]:
        """Create and return one rival generation's members.

        Workers return plain Python objects instead of writing through a multiprocessing.Manager proxy. On Windows,
        high-frequency Manager proxy traffic uses named pipes and can fail with WinError 231 when the pipe server is
        saturated.
        """
        if _worker_fitness_function is None or _worker_operators is None or _worker_args is None:
            raise RuntimeError("Parallel worker was not initialized with GA configuration.")

        _, crossover = _worker_operators[combination_id]
        crossover_args = (_worker_args.get("crossover")
                          if isinstance(_worker_args, dict) and "crossover" in _worker_args else None)
        parent_pairs = GeneticAlgorithm._select_parent_genome_pairs(
            combination_id, parent_generation, _worker_operators, _worker_args
        )
        new_members = GeneticAlgorithm._build_members_from_parent_pairs(
            parent_pairs, crossover, crossover_args, _worker_fitness_function, first_identification_number
        )
        return combination_id, new_members

    @staticmethod
    def _create_member_batch(
            combination_id: int,
            first_pair_index: int,
            parent_pairs: list[tuple[list | dict, list | dict]],
            first_identification_number: int
    ) -> tuple[int, int, list[Member]]:
        """Create one parent-pair batch in a configured pool worker."""
        if _worker_fitness_function is None or _worker_operators is None or _worker_args is None:
            raise RuntimeError("Parallel worker was not initialized with GA configuration.")
        _, crossover = _worker_operators[combination_id]
        crossover_args = (_worker_args.get("crossover")
                          if isinstance(_worker_args, dict) and "crossover" in _worker_args else None)
        members = GeneticAlgorithm._build_members_from_parent_pairs(
            parent_pairs,
            crossover,
            crossover_args,
            _worker_fitness_function,
            first_identification_number,
            first_pair_index,
        )
        return combination_id, first_pair_index, members

    def _create_rival_members(
            self,
            worker_pool: Pool,
            operator_combinations_ids: list[int],
            first_member_id: int,
            members_per_rival: int,
            no_workers: int
    ) -> dict[int, list[Member]]:
        """Create rival members using the configured local or process-pool strategy."""
        creation_mode = self._resolve_creation_parallelism(no_workers)

        if creation_mode == "operators":
            creation_jobs = [
                (combination_id, self.current_generation, first_member_id + offset * members_per_rival)
                for offset, combination_id in enumerate(operator_combinations_ids)
            ]
            return dict(worker_pool.starmap(self._create_members_for_rival_generation, creation_jobs))

        if creation_mode == "local":
            rival_members = {}
            for offset, combination_id in enumerate(operator_combinations_ids):
                parent_pairs = self._select_parent_genome_pairs(
                    combination_id, self.current_generation, self.operators, self.args
                )
                _, crossover = self.operators[combination_id]
                crossover_args = (self.args.get("crossover")
                                  if isinstance(self.args, dict) and "crossover" in self.args else None)
                rival_members[combination_id] = self._build_members_from_parent_pairs(
                    parent_pairs,
                    crossover,
                    crossover_args,
                    self.fit_fun,
                    first_member_id + offset * members_per_rival,
                )
            return rival_members

        creation_jobs = []
        for offset, combination_id in enumerate(operator_combinations_ids):
            parent_pairs = self._select_parent_genome_pairs(
                combination_id, self.current_generation, self.operators, self.args
            )
            batch_size = max(1, ceil(len(parent_pairs) / (no_workers * 4)))
            rival_first_id = first_member_id + offset * members_per_rival
            for first_pair_index in range(0, len(parent_pairs), batch_size):
                creation_jobs.append((
                    combination_id,
                    first_pair_index,
                    parent_pairs[first_pair_index:first_pair_index + batch_size],
                    rival_first_id,
                ))

        created_batches = worker_pool.starmap(self._create_member_batch, creation_jobs)
        batches_by_rival = {combination_id: [] for combination_id in operator_combinations_ids}
        for combination_id, first_pair_index, members in created_batches:
            batches_by_rival[combination_id].append((first_pair_index, members))

        return {
            combination_id: [
                member
                for _, members in sorted(batches, key=lambda batch: batch[0])
                for member in members
            ]
            for combination_id, batches in batches_by_rival.items()
        }

    def best_solution(self):
        """Returns the genome of Member with the highest fitness value with its fitness value,
        from the current Generation.

        Returns:
            tuple[list | dict, float]: tuple of the genome list/dict of the best Member and its float fit. value
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

    def mutate(self, mutation_type: str = "member") -> list[int]:  # TODO: add adaptive mutation
        """Mutate the current generation and return indexes whose fitness values became stale.
        
        Mutation types:
        - "member": Entire genome reset
        - "gene": Individual genes replaced
        - "gaussian": Numeric genes perturbed (Gaussian noise)
        - "swap": Two random genes swap positions
        - "custom": Expensive user operator executed and evaluated in worker batches by ``run()``
        
        Parameters:
            mutation_type (str): Type of mutation to apply

        Returns:
            list[int]: Sorted indexes of members changed by the mutation operator. Evaluation is intentionally handled
                by :meth:`run` so the same process pool can reevaluate affected genomes in batches.
        """
        if self.mutation_prob == 0.0:
            return []

        match mutation_type:
            case "member":
                return self._mutate_members()
            case "gene":
                return self._mutate_genes()
            case "gaussian":
                return self._mutate_gaussian()
            case "swap":
                return self._mutate_swap()
            case "custom":
                raise RuntimeError(
                    "Custom mutation is worker-side only; call run() so it can use the persistent process pool."
                )
            case _:
                print(f"Warning: Unknown mutation type '{mutation_type}'. Using 'member' by default.")
                return self._mutate_members()

    def _mutate_members(self) -> list[int]:
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
        return sorted(indexes)

    def _mutate_genes(self) -> list[int]:
        """Mutates individual genes across members of the current generation.

        Selected positions are grouped by member. Generators supporting ``gene_position`` produce each replacement
        directly; legacy full-genome generators are called once per affected member and provide all its replacements.
        """
        non_elite_members_count = self.current_generation.size - self.elite_size
        total_available_genes = non_elite_members_count * self.current_generation.genome_size

        number_of_mutations = int(np.floor(self.mutation_prob * total_available_genes))

        if number_of_mutations == 0:
            return []

        # Select random gene indexes across all non-elite members and genes
        gene_indexes = random.sample(range(total_available_genes), number_of_mutations)

        mutations_by_member: dict[int, list[int]] = {}
        for gene_index in gene_indexes:
            member_index, gene_position = divmod(gene_index, self.current_generation.genome_size)
            mutations_by_member.setdefault(member_index, []).append(gene_position)

        for member_index, gene_positions in mutations_by_member.items():
            genome = self.current_generation.members[member_index].genome
            if self.genome_generator_supports_single_gene:
                for gene_position in gene_positions:
                    genome[gene_position] = self.genome_generator(
                        self.args, gene_position=gene_position
                    )
            else:
                replacement_genome = self.genome_generator(self.args)
                for gene_position in gene_positions:
                    genome[gene_position] = replacement_genome[gene_position]

        return sorted(mutations_by_member)

    def _mutate_gaussian(self) -> list[int]:
        """Gaussian mutation: adds random noise to numeric genes only."""
        non_elite_members_count = self.current_generation.size - self.elite_size
        total_available_genes = non_elite_members_count * self.current_generation.genome_size

        number_of_mutations = int(np.floor(self.mutation_prob * total_available_genes))

        if number_of_mutations == 0:
            return []

        gene_indexes = random.sample(range(total_available_genes), number_of_mutations)

        genome_spec, _ = self.args["genome"]
        if isinstance(genome_spec, dict):
            spec_list = [genome_spec[position] for position in range(self.current_generation.genome_size)]
        else:
            spec_list = genome_spec

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

        return sorted(affected_members)

    def _mutate_swap(self) -> list[int]:
        """Swap mutation: randomly swaps two genes within a Member's genome."""
        non_elite_members_count = self.current_generation.size - self.elite_size
        number_of_mutations = int(np.floor(self.mutation_prob * non_elite_members_count))

        if number_of_mutations == 0:
            return []

        member_indexes = random.sample(range(non_elite_members_count), number_of_mutations)

        for member_index in member_indexes:
            # Randomly select two gene positions to swap
            pos1, pos2 = random.sample(range(self.current_generation.genome_size), 2)

            # Swap genes
            genome = self.current_generation.members[member_index].genome
            genome[pos1], genome[pos2] = genome[pos2], genome[pos1]

        return sorted(member_indexes)

    def _evaluate_member_indexes(
            self,
            worker_pool: Pool,
            member_indexes: list[int],
            no_workers: int
    ) -> None:
        """Reevaluate selected current-generation members using the persistent worker pool."""
        if not member_indexes:
            return

        members_to_evaluate = {
            self.current_generation.members[index].id: self.current_generation.members[index].genome
            for index in member_indexes
        }
        fitness_batches = self._create_fitness_batches(members_to_evaluate, no_workers)
        evaluated_batches = worker_pool.map(_evaluate_fitness_batch, fitness_batches, chunksize=1)
        evaluated_members = dict(
            evaluated_member
            for batch in evaluated_batches
            for evaluated_member in batch
        )
        for index in member_indexes:
            member = self.current_generation.members[index]
            member.fit_val = evaluated_members[member.id]

    def _mutate_custom_in_workers(
            self,
            worker_pool: Pool,
            no_workers: int,
            evaluate_all_members: bool
    ) -> list[int]:
        """Run an opt-in expensive custom mutation and fitness evaluation in the same worker batches.

        When a sole, not-yet-evaluated rival is accepted, ``evaluate_all_members`` includes unchanged members so the
        fused pass initializes every fitness value. For an already-evaluated winner among multiple rivals, only mutated
        members are sent back through the pool.
        """
        if self.custom_mutation_operator is None:
            raise ValueError(
                "Mutation type 'custom' requires custom_mutation_operator to be passed to GeneticAlgorithm."
            )

        non_elite_members_count = self.current_generation.size - self.elite_size
        number_of_mutations = int(np.floor(self.mutation_prob * non_elite_members_count))
        mutated_indexes = set(random.sample(range(non_elite_members_count), number_of_mutations))
        indexes_to_process = (
            list(range(self.current_generation.size))
            if evaluate_all_members
            else sorted(mutated_indexes)
        )
        if not indexes_to_process:
            return []

        jobs = [
            (
                index,
                self.current_generation.members[index].id,
                self.current_generation.members[index].genome,
                index in mutated_indexes,
            )
            for index in indexes_to_process
        ]
        batch_size = max(1, ceil(len(jobs) / (no_workers * 4)))
        batches = [jobs[index:index + batch_size] for index in range(0, len(jobs), batch_size)]
        evaluated_batches = worker_pool.map(_mutate_and_evaluate_batch, batches, chunksize=1)

        for member_index, member_id, genome, fitness in (
                result for batch in evaluated_batches for result in batch
        ):
            member = self.current_generation.members[member_index]
            if member.id != member_id:
                raise RuntimeError(
                    f"Custom mutation result ID {member_id} does not match member {member.id} at index {member_index}."
                )
            member.genome = genome
            member.fit_val = fitness

        return sorted(mutated_indexes)

    def run(self):
        """Run the GA with one persistent, initialised process pool.

        Rival generations are created in parallel. Their genomes are then evaluated in batches, with approximately four
        batches per worker to balance uneven fitness costs while avoiding one IPC task per member. Fitness-function
        exceptions have the same semantics as serial ``Member.evaluate``: the affected member receives fitness ``0.0``.
        When only one rival exists, it is accepted and mutated before evaluation, avoiding a redundant pre-mutation
        fitness pass. Multiple rivals are evaluated first because their fitness values determine which one is accepted.
        Expensive custom mutation is fused with worker-side evaluation; built-in lightweight mutation remains local.
        """
        global identification
        # print(f"Creating the initial population.")
        self._create_initial_generation()

        operator_combinations_ids = list(self.operators.keys())
        members_per_rival = 2 * self.no_parents_pairs
        no_members = members_per_rival * len(operator_combinations_ids)
        no_workers = self._get_parallel_worker_count(no_members=no_members)

        with Pool(
                processes=no_workers,
                initializer=_initialize_parallel_worker,
                initargs=(self.fit_fun, self.operators, self.args, self.custom_mutation_operator)
        ) as worker_pool:
            for _ in range(self.no_generations):
                """Rival generations are created based on accessible combinations of selection and crossover
                operators with different processes in parallel:"""
                first_member_id = identification
                rival_members_container = self._create_rival_members(
                    worker_pool,
                    operator_combinations_ids,
                    first_member_id,
                    members_per_rival,
                    no_workers,
                )
                identification += no_members

                """Build rival Generations out of members and compose their respective fitness rankings"""
                self.rival_gen_pool = {}
                for combination_id in operator_combinations_ids:
                    # TODO: Copy the configured elite into each rival generation before accepting one.
                    # TODO: Preserve pop_size when no_parents_pairs is smaller than initial_pop_size // 2.
                    new_rival_generation = Generation(
                        generation_members=rival_members_container[combination_id],
                        num_parents_pairs=self.current_generation.num_parents_pairs,
                        elite_size=self.elite_size
                    )
                    self.rival_gen_pool[combination_id] = new_rival_generation

                if len(self.rival_gen_pool) == 1:
                    """With no rival choice to make, mutate first and evaluate the resulting generation only once."""
                    self.current_generation = next(iter(self.rival_gen_pool.values()))
                    self.accepted_gen_list.append(self.current_generation)
                    if self.args.get('mutation') == "custom":
                        self._mutate_custom_in_workers(worker_pool, no_workers, evaluate_all_members=True)
                    else:
                        self.mutate(mutation_type=self.args.get('mutation'))
                        self._evaluate_member_indexes(
                            worker_pool,
                            list(range(self.current_generation.size)),
                            no_workers,
                        )
                    self.current_generation.fitness_ranking = []
                    self.current_generation.create_fitness_ranking()
                    self.best_fit_history.append(
                        self.current_generation.fitness_ranking[0].get('fitness value')
                    )
                    continue

                """Multiple rivals must be evaluated before the best one can be selected."""
                all_members = {
                    member.id: member.genome
                    for generation in self.rival_gen_pool.values()
                    for member in generation.members
                }

                fitness_batches = self._create_fitness_batches(all_members, no_workers)
                evaluated_batches = worker_pool.map(_evaluate_fitness_batch, fitness_batches, chunksize=1)
                evaluated_members = dict(
                    evaluated_member
                    for batch in evaluated_batches
                    for evaluated_member in batch
                )

                for generation in self.rival_gen_pool.values():
                    for member in generation.members:
                        member.fit_val = evaluated_members[member.id]

                    generation.fitness_ranking = []
                    generation.create_fitness_ranking()

                """Choose the best evaluated rival, mutate it, and reevaluate only changed members."""
                self._choose_best_rival_generation()
                # print(self.best_solution())
                # print(self.current_generation.members)
                if self.args.get('mutation') == "custom":
                    self._mutate_custom_in_workers(worker_pool, no_workers, evaluate_all_members=False)
                else:
                    affected_member_indexes = self.mutate(mutation_type=self.args.get('mutation'))
                    self._evaluate_member_indexes(worker_pool, affected_member_indexes, no_workers)
                self.current_generation.fitness_ranking = []
                self.current_generation.create_fitness_ranking()
                self.best_fit_history[-1] = self.current_generation.fitness_ranking[0].get('fitness value')

    def fitness_plot(self):  # TODO: finish with an optional argument for using plotly or matplotlib
        """Method for plotting fitness values history of the best Members from each accepted Generation."""
        pass
