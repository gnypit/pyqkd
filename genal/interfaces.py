from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Literal


class GAArgumentsInterface(Mapping[str, object], ABC):
    """Common configuration contract received by every GA operator.

    Implementations must support process-pool pickling. Sections are read-only
    through this API; mutable objects within sections are not deeply frozen.
    Operator-specific validation belongs to the operator using a section.
    """

    @property
    @abstractmethod
    def genome(self) -> tuple[object, int]:
        """Gene space and genome length; required by built-in generators."""
        ...

    @property
    @abstractmethod
    def selection(self) -> Mapping[str, object]:
        ...

    @property
    @abstractmethod
    def crossover(self) -> Mapping[str, object] | Sequence | None:
        ...

    @property
    @abstractmethod
    def mutation(self) -> Mapping[str, object]:
        """Mutation settings, including the optional ``type`` key."""
        ...

    @property
    def mutation_type(self) -> str:
        return self.mutation.get("type", "member")

    def crossover_parameter(self, name: str, position: int, default: object) -> object:
        """Read named settings, with support for legacy positional sequences."""
        section = self.crossover
        if section is None:
            return default
        if isinstance(section, Mapping):
            return section.get(name, default)
        return section[position]


class ChromosomeInterface(ABC):
    """Abstract class defining a chromosome, representing a possible solution to an optimization problem.

    Attributes:
        fit_val (float): Fitness value of the chromosome.
        genome (list | dict): Either a list or a dict with genes of this chromosome.
        fit_fun (Callable | None): Fitness function used for computing fitness value based on chromosome's genes.
    """
    fit_val: float | int
    genome: list | dict
    fit_fun: Callable | None = None

    @abstractmethod
    def evaluate(self, fitness_function: Callable | None):
        pass

    @abstractmethod
    def change_genes(self, new_genes: list | dict):
        pass


class GenerationInterface(ABC):
    """Interface defining a single generation in a genetic algorithm.

    Attributes:
        members (list[ChromosomeInterface]): List of chromosomes.
        genome_size (int): a constant size of this generation's members' genome, i.e., number of params specifying a
            potential solution to a problem being optimised by the genetic algorithm.
        num_parents_pairs (int): number of pairs of members can be parents, e.g., 20 pairs means 40 mating chromosomes.
        fitness_ranking (list[dict[Literal['index', 'fitness value'], int | float]]): dicts in this list have the index
            of a member and its fitness value as values.
    """
    members: list[ChromosomeInterface]
    genome_size: int
    num_parents_pairs: int
    fitness_ranking: list[dict[Literal['index', 'fitness value'], int | float]]

    @abstractmethod
    def evaluate(self):
        """Abstract method meant for calculating fitness values of all generation's members."""
        pass

    @abstractmethod
    def create_fitness_ranking(self, reverse=True):
        """Abstract method meant for constructing a fitness ranking of this generation's members.

        Parameters:
            reverse (Bool=True, optional): parameter which decided whether the fitness ranking should be sorted in
                ascending order of fitness values (reverse=False) or in descending order (reverse=True), which is
                the default.
        """
        pass

    @abstractmethod
    def mutate_members(self, prob: float):
        pass


class GeneticAlgorithmInterface(ABC):
    """Interface defining a genetic algorithm."""
