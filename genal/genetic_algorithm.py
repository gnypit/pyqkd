"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import floor
import crossover_operators
import selection_operators

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

    def __init__(self, genes: type[list | dict], fitness_function=None):
        """Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc.

        Here, `genome` is either a dict with genes as values and names provided by the User as keys, or simply a list.

        For computational purposes of parallel programming, the fitness function can be passed to
        the Chromosome on its initiation/construction.
        """
        self.genome = genes
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

    def __init__(self, genes: type[list | dict], identification_number: int, fitness_function=None):
        """Apart from what 'Chromosome' class constructor needs, here identification number should be passed."""
        super().__init__(genes=genes, fitness_function=fitness_function)
        self.id = identification_number

    def add_parents_id(self, parents_id: list):
        """This method is meant for 'genealogical tree' tracking;
        it assigns to the current member IDs of its parents.
        """
        self.parents_id = parents_id

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return f"{type(self).__name__}(genes={self.genome}, id={self.id}, parents_id={self.parents_id})"


class Generation:
    """This class is meant to represent a single (rival) generation in a genetic algorithm, with its members and
    characteristic info: current fitness ranking of the members, elite size, number of parents' pairs mating
    and operator-specific arguments, if necessary.
    """
    members: list[Member]  # list of Member class instances - chromosomes of the generation with their and parents IDS
    size: int  # number of members in this generation
    num_parents_mating: int  # number of parent paris mating must be positive and equal to or smaller than the size
    elite_size: int  # number of members to be copy-pasted directly into a new generation
    fitness_ranking: list[dict]  # dicts in this list have the index of a member in the generation and its fitness value

    def __init__(self, generation_members, number_of_parents_pairs_mating, elite_size, fitness_ranking):
        self.members = generation_members
        self.size = len(generation_members)
        self.num_parents_mating = number_of_parents_pairs_mating
        self.elite_size = elite_size
        self.fitness_ranking = fitness_ranking

    def add_member(self, genome, parents_id=None):
        """Method for manual creation of new members"""

        global identification
        new_member = Member(genes=genome, identification_number=identification)

        if parents_id is not None:
            new_member.add_parents_id(parents_id=parents_id)

        self.members.append(new_member)
        identification += 1

    def evaluate(self, reverse=True):
        """This method uses the fitness function stored in members of the generation to create and then sort the fitness
        ranking by the computed fitness values; 'reverse' means sorting will be performed from max fitness value to min.
        """
        self.fitness_ranking = []

        for i in range(self.size):
            self.fitness_ranking.append(
                {'index': i, 'fitness value': self.members[i].evaluate()}
            )

        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)


class GeneticAlgorithm:
    
    def __zip_crossover_selection(self,selection_operator,crossover_operator):
        # Creates a list that combines pairs of elements from 'selection_operator' 
        # and 'crossover_operator'. For each index 'i', it adds a tuple containing 
        # 'selection_operator[i]' and 'crossover_operator[i]' to the 'listoperator' list.
        listoperator=[]        
        for i in range(len(selection_operator)):
            listoperator.append((selection_operator[i],crossover_operator[i]))
        return listoperator
    
    def __init__(self, initial_pop_size, fit_fun, genome_generator, elite_size, selection_operator:list, crossover_operator:list,
                 number_of_generations, args: dict, no_parents_pairs=None, mutation_prob=0.0, seed=None):
        """initial_pop_size is the size of an initial population, fit_fun is a chosen fitness function to be used in a
        genetic algorithm, genom_generator is the function that creates genomes for the initial generation
        of population members, args are arguments to be used in genome_generator & selection/crossover operators,
        mutation_prob is a probability of a single member's genome being initialised from scratch,
        seed is an optional argument useful for comparison of pseudo-random number generation

        no_parents_pairs is the designated number of parent pairs for future generations,
        e.g., if the initial population size is 1000 and no_parents_pairs = 200, there will be 2 * 200 = 400 children
        in the next generation, which becomes a constant population size. Additionally, the elite_size number of
        Members is copied from an i-th generation to the (i+1)-th generation.
        """

        if seed is not None:
            random.seed(a=seed)  # useful for debugging

        self.pop_size = initial_pop_size
        self.fit_fun = fit_fun
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.no_generations = number_of_generations

        """For now remembering a single operator for selection and a single for crossover:"""  # TODO: to be changed in the parallel version
        self.operator=self.__zip_crossover_selection(selection_operator=selection_operator,crossover_operator=crossover_operator)

        """If the provided number of parents pairs would require more Members than the current (initial) generation has,
        it'll be limited to the maximum possible number. Also, if no specific number of parent pairs is provided,
        the initial population size is assumed to be a constant throughout the whole algorithm."""
        if no_parents_pairs is None or no_parents_pairs > initial_pop_size // 2:
            self.no_parents_pairs = initial_pop_size // 2
        else:
            self.no_parents_pairs = no_parents_pairs

        """Even though for the initial population we can pass the genome generator with it's arguments
        directly to the __init__ method within the Generation class, we need to memorise these two variables
        for mutation later on."""
        self.genome_generator = genome_generator
        self.genome_generator_args = args.get('genome')
        self.selection_args = args.get('selection')
        self.crossover_args = args.get('crossover')

        """Creating the first - initial - generation in this population and lists to handle future generations"""
        self.current_generation = Generation(
            size=initial_pop_size,
            genome_generator=genome_generator,
            genome_args=self.genome_generator_args,
            fitness_function=fit_fun
        )
        self.generations = [self.current_generation]

        if self.genome_generator is not None:  # ONLY for the initial generation within the population -> should it be in the GeneticAlgorithm class, or do we need it for mutation too?
            for index in range(self.size):
                genes = self.genome_generator(self.genome_generator_args)
                new_member = Member(
                    genes=genes,
                    identification_number=identification,
                    fitness_function=fitness_function
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

    def create_new_generation(self):  # First to be parallelled & TODO: change the way the selection operators are handled
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
            new_generation.add_member(genome=pair[0])
            new_generation.add_member(genome=pair[1])

        """Thirdly, we add the elite - it doesn't matter that it's at the end of the new generation, because it'll be
        sorted anyway after new Members evaluation."""
        index = 0
        while index < self.elite_size:
            new_generation.add_member(
                genome=self.current_generation.members[self.current_fitness_ranking[index].get('index')].genome
            )
            new_generation.add_member(
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

    def change_population_size(self, pop_size):  # TODO isin't it in a conflict with the change to initial size and parent pairs number?
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
