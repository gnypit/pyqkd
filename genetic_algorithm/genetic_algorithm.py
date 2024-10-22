"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import floor
from crossover_operators import uniform_crossover, single_point_crossover, plco

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
    def __init__(self, genes, fitness_function=None):
        """Each chromosome represents a possible solution to a given problem. Parameters characterising these solutions
        are called genes; their set is sometimes referred to as 'genome'. They are supposed to be evaluated by the
        fitness function. Then, based on the fitness (function's) values, they are compared, sorted, selected for
        crossover, etc.

        For computational purposes of parallel programming, the fitness function can be passed to
        the Chromosome on its initiation/construction.
        """
        self.genes = genes  # a dictionary
        self.fit_fun = fitness_function
        self.fit_val = None

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return (f"{type(self).__name__}(genes={self.genes}, fitness function={self.fit_fun}, "
                f"fitness value={self.fit_val})")

    def change_genes(self, genes):
        """Method meant to be used when mutation occurs, to modify the genes in an already created chromosome.
        Can be called upon manually."""
        self.genes = genes

    def evaluate(self, fitness_function=None):
        """Method for applying fitness function to this chromosome (it's genes, to be precise).
        If the fitness function was passed on in the constructor of this class, it has to be provided as an argument of
        this method. Fitness value is remembered in a field of this classed and returned on output. If no fitness
        function is provided, the assigned fitness value is 0."""
        if fitness_function is None:
            self.fit_fun = fitness_function
        elif self.fit_fun is not None:
            self.fit_val = self.fit_fun(self.genes)
        else:
            self.fit_val = 0

        return self.fit_val

    def __iter__(self):  # might be redundant
        return self


class Member(Chromosome):
    """This class is a child of the 'Chromosome' class and is designated to store a unique ID, enabling tracking a
    genealogical tree of chromosomes in a population of a genetic algorithm.
    """
    def __init__(self, genes, identification_number, fitness_function=None):
        """Apart from what 'Chromosome' class' constructor needs, here identification number should be passed."""
        super().__init__(genes=genes, fitness_function=fitness_function)
        self.id = identification_number
        self.parents_id = None

    def add_parents_id(self, parents_id):
        """This method is meant for 'genealogical tree' tracking;
        it assigns to the current member IDs of its parents.
        """
        self.parents_id = parents_id  # it's a list with IDs of the parents

    def __repr__(self) -> str:
        """Default method for self-representing objects of this class."""
        return f"{type(self).__name__}(genes={self.genes}, id={self.id}, parents_id={self.parents_id})"


class Generation:
    def __init__(self, size, fitness_function, genome_generator=None, genome_args=None):
        """genome_generator is the function that creates genomes for the initial generation
        of population members, genome_args are arguments to be used in genome_generator;
        this method uses a global variable identification for creating unique IDs for created members"""

        global identification
        self.size = size
        self.fitness_function = fitness_function
        self.members = []
        self.genome_generator = genome_generator
        self.genome_generator_args = genome_args

        if self.genome_generator is not None:  # ONLY for the initial generation within the population
            for index in range(self.size):
                new_member = Member(
                    genes=self.genome_generator(self.genome_generator_args),
                    identification_number=identification,
                    fitness_function=fitness_function
                )
                identification += 1
                self.members.append(new_member)

        """With parallel computation in mind, with possible multiple rival generations being created, 
        a more transparent and easier to control approach is to store the current fitness ranking of a given generation
        inside the Generation class. Besides, the fitness ranking is a property of a generation, 
        not the whole population.
        """
        self.fitness_ranking = []

    def add_member(self, genome, parents_id=None):
        """Method for manual creation of new members"""

        global identification
        new_member = Member(genes=genome, identification_number=identification)

        if parents_id is not None:
            new_member.add_parents_id(parents_id=parents_id)

        self.members.append(new_member)
        identification += 1

    def evaluate_all_members(self, reverse=True, fitness_function=None):
        """This method applies the fitness function to the generation and sorts the fitness ranking by
        the fitness values of generation's members - 'reverse' means sorting will be performed
        from maximum fitness to the minimum.

        If 'fitness_function' is provided, it overrides the one given in the constructor."""
        self.fitness_ranking = []

        for i in range(len(self.members)):
            self.fitness_ranking.append(
                {'index': i, 'fitness value': self.members[i].evaluate()}
            )

        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=reverse)


class Population:
    def __init__(self, pop_size, fit_fun, genome_generator, elite_size, args: dict, mutation_prob=0.0, seed=None):
        """pop_size is a constant size of the population, fit_fun is a chosen fitness function to be used in a
        genetic algorithm, genom_generator is the function that creates genomes for the initial generation
        of population members, args are arguments to be used in genome_generator & selection/crossover operators,
        mutation_prob is a probability of a single member's genome being initialised from scratch,
        seed is for pseud-random number generation."""

        # TODO differentiate between initial population size and the population size later on

        if seed is not None:
            random.seed(a=seed)  # temporary, for debugging

        self.pop_size = pop_size
        self.fit_fun = fit_fun
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob

        """Even though for the initial population we can pass the genome generator with it's arguments
        directly to the __init__ method within the Generation class, we need to memorise these two variables
        for mutation later on."""
        self.genome_generator = genome_generator
        self.genome_generator_args = args.get('genome')  # TODO we need to verify it and document outside of code...
        # TODO ...as there are more and more args everywhere and I don't know how to handle it better
        self.selection_args = args.get('selection')
        self.crossover_args = args.get('crossover')

        """Creating the first - initial - generation in this population and lists to handle future generations"""
        self.current_generation = Generation(
            size=pop_size,
            genome_generator=genome_generator,
            genome_args=self.genome_generator_args
        )
        self.generations = [self.current_generation]
        self.current_parents = []
        self.current_children = []

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
        bf = [self.current_generation.members[self.current_fitness_ranking[0].get('index')].genes,
              self.current_fitness_ranking[0].get('fitness value')]
        return bf

    def ranking_selection(self):  # deterministic
        """We use 'member_counter' as an index for the fitness ranking. While it's smaller than the elite
        size for the whole population, we simply copy-paste the old members into the new generation. That's why
        we automatically calculate it and focus on selecting pairs of members from the current generation as parents
        for crossover performed later. Only then will the 'elite' members be copy-pasted."""
        member_counter = 0

        """As every other one, this selection operator creates his own list of candidates for parents of the future
        generation from the current generation and appends it to the 'parents' field in this class:"""
        parents_candidates = []

        """Because I decided to not only preserve the elite, but also perform crossover on it, I'll disregard
        a part of current generation's members with worst fitness, so that the size os population is constant.
        
        We'll have elite_size number of elite Members copied, elite_size number of Members being the children of the 
        elite, and that leaves us with (pop_size - 2 * elite_size) number of places in the generation. Since the
        elite-parents will be added now, we have to subtract the 'other' elite_size number of Members from the
        loop limit to preserve the right size of generation - for when the elite will be copied directly
        into children's list:"""
        while member_counter < self.current_generation.size - self.elite_size:
            parent1 = self.current_generation.members[self.current_fitness_ranking[member_counter].get('index')]
            parent2 = self.current_generation.members[self.current_fitness_ranking[member_counter + 1].get('index')]
            parents_candidates.append({'parent1': parent1, 'parent2': parent2})
            member_counter += 2

        self.current_parents.append({'ranking': parents_candidates})
        # TODO: why where there three lists from ranking selection alone??? -> probably resolved by now 10/11/2023

    def roulette_wheel_selection(self):  # probability-based
        """The conspicuous characteristic of this selection method is the fact that it gives to
        each member i of the current generation a probability p(i) of being selected,
        proportional to its fitness value f(i).

        Each member in the population occupies an area on the roulette wheel proportional to its
        fitness. Then, conceptually, the roulette wheel is spun as many times as the population size, each time
        selecting a member marked by the roulette-wheel pointer. Since the members are marked proportionally
        to their fitness, a member with a higher fitness is likely to receive more copies than a solution with a
        low fitness.

        For more details please refer to 'Introduction to Evolutionary Computing', sect. 5.2.3 'Implementing Selection
        Probabilities'. [DOI 10.1007/978-3-662-44874-8]"""
        fit_total = 0

        """In order to select members with regard to their fitness value compared to all of the values,
        we calculate the total sum of fitness values of all members in the current generation:"""
        for i in range(self.pop_size):
            fit_total += self.current_fitness_ranking[i].get('fitness value')

        """We use 'member_counter' as an index for the fitness ranking. While it's smaller than the elite
        size for the whole population, we simply copy-paste the old members into the new generation. That's why
        we automatically calculate it and focus on selecting pairs of members from the current generation as parents
        for crossover performed later. Only then will the 'elite' members be copy-pasted."""
        member_counter = 0

        """As every other one, this selection operator creates his own list of candidates for parents of the future
        generation from the current generation and appends it to the 'parents' field in this class:"""
        parents_candidates = []

        """Because I decided to not only preserve the elite, but also perform crossover on it, I'll disregard
        a part of current generation's members with worst fitness, so that the size os population is constant.

        We'll have elite_size number of elite Members copied, elite_size number of Members being the children of the 
        elite, and that leaves us with (pop_size - 2 * elite_size) number of places in the generation. Since the
        elite-parents will be added now, we have to subtract the 'other' elite_size number of Members from the
        loop limit to preserve the right size of generation - for when the elite will be copied directly
        into children's list:"""
        while member_counter < self.current_generation.size - self.elite_size:
            """In each iteration we add two parents who will result in two children,
            which is why we use a while loop and 'jump' 2 population members in each iteration."""
            param = random.uniform(0, fit_total)
            fit_sum = 0
            index = 0

            while fit_sum < param and index < self.pop_size:
                fit_sum += self.current_fitness_ranking[index].get('fitness value')
                index += 1
            parent1 = self.current_generation.members[self.current_fitness_ranking[index].get('index')]

            param = random.uniform(0, fit_total)
            fit_sum = 0
            index = 0
            while fit_sum < param and index < self.pop_size:
                fit_sum += self.current_fitness_ranking[index].get('fitness value')
                index += 1
            parent2 = self.current_generation.members[self.current_fitness_ranking[index].get('index')]

            parents_candidates.append({'parent1': parent1, 'parent2': parent2})
            member_counter += 2

        self.current_parents.append({'roulette wheel': parents_candidates})

    def stochastic_universal_sampling(self):  # probability-based
        """It is an improved version of the roulette wheel selection operator, as described in 'Introduction to
        Evolutionary Computing' in sect. 5.2.3. We begin similarly to the roulette wheel, because we still base
        the probabilities of selection on the cumulative probability distribution, associated with the fitness values.
        """
        fit_total = 0
        cumulative_prob_distribution = []
        for i in range(self.pop_size):
            fit_total += self.current_fitness_ranking[i].get('fitness value')
            cumulative_prob_distribution.append(self.current_fitness_ranking[i].get('fitness value'))
        cumulative_prob_distribution = [fit / fit_total for fit in cumulative_prob_distribution]

        random_value = random.uniform(0, 1 / self.current_generation.size)
        parents = []
        index = 0

        """We use 'member_counter' as an index for the fitness ranking. While it's smaller than the elite
        size for the whole population, we simply copy-paste the old members into the new generation. That's why
        we automatically calculate it and focus on selecting pairs of members from the current generation as parents
        for crossover performed later. Only then will the 'elite' members be copy-pasted."""
        member_counter = 0

        """As every other one, this selection operator creates his own list of candidates for parents of the future
        generation from the current generation and appends it to the 'parents' field in this class:"""
        parents_candidates = []

        # TODO: optimise loops; verify that it's correct

        """Because I decided to not only preserve the elite, but also perform crossover on it, I'll disregard
        a part of current generation's members with worst fitness, so that the size os population is constant.

        We'll have elite_size number of elite Members copied, elite_size number of Members being the children of the 
        elite, and that leaves us with (pop_size - 2 * elite_size) number of places in the generation. Since the
        elite-parents will be added now, we have to subtract the 'other' elite_size number of Members from the
        loop limit to preserve the right size of generation - for when the elite will be copied directly
        into children's list:"""
        while member_counter < self.current_generation.size - self.elite_size:
            while random_value <= cumulative_prob_distribution[index]:  # TODO IndexError: list index out of range
                """We keep assigning given member as a new parent until the condition is met. This way we reflect
                the probability of it's selection."""
                new_parent = self.current_generation.members[self.current_fitness_ranking[index].get('index')]
                random_value += 1 / self.current_generation.size
                member_counter += 1
                parents.append(new_parent)
            index += 1

        """Now I rewrite parents into an organised list of pairs as dictionaries..."""
        index = 0
        while index < self.current_generation.size:
            parents_candidates.append({'parent1': parents[index], 'parent2': parents[index + 1]})
            index += 2

        """...and append it to the list of candidate parents lists:"""
        self.current_parents.append({'sus': parents_candidates})

    def perform_crossover(self, crossover_operator, selection_operator_name):
        """Let's try passing the selection operator info into the crossover operator, so that instead of forcing
        taking a list of dict_values we simply call the value with a key.

        We need to pass a list of arguments for the crossover operator, if necessary. For args = None crossover
        operators use default settings/parameters.
        """
        children_candidates = []
        for parents_candidates in self.current_parents:
            # list_of_parents_pairs = list(parents_candidates.values())  # I'm forcing it to be a list object
            list_of_parents_pairs = parents_candidates.get(selection_operator_name)
            for parents_pair in list_of_parents_pairs:  # this loop end way to soon, I think
                children_candidates.append(
                    crossover_operator(
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

    def create_new_generation(self, selection_operator, crossover_operator):
        """A method for combining selection and crossover operators over the current population to create a new one.
        For the moment, we are assuming that there will be a single list of children candidates.
        Firstly, we have to match the selection operator; then in each case we have to match the crossover operator.

        In each of the selection-oriented cases we feed the selection operator name to the crossover operator
        method, so that it takes the parents lists designated for a given new generation creation, i.e., to
        always connect the chosen crossover to chosen selection and yet keep all probable parents lists
        from different selection processes in one object for multiple processes to access.

        Selection_operator is a function passed to this method for parents selection
        crossover_operator is a function passed to this method for the crossover of the parents
        """

        match str(selection_operator):
            case 'ranking':
                self.ranking_selection()
                match crossover_operator:
                    case 'single point':
                        self.perform_crossover(
                            crossover_operator=single_point_crossover,
                            selection_operator_name='ranking'
                        )
                    case 'uniform':
                        self.perform_crossover(
                            crossover_operator=uniform_crossover,
                            selection_operator_name='ranking'
                        )
                    case 'plco':
                        self.perform_crossover(
                            crossover_operator=plco,
                            selection_operator_name='ranking'
                        )
            case 'roulette wheel':
                self.roulette_wheel_selection()
                match crossover_operator:
                    case 'single point':
                        self.perform_crossover(
                            crossover_operator=single_point_crossover,
                            selection_operator_name='roulette wheel'
                        )
                    case 'uniform':
                        self.perform_crossover(
                            crossover_operator=uniform_crossover,
                            selection_operator_name='roulette wheel'
                        )
                    case 'plco':
                        self.perform_crossover(
                            crossover_operator=plco,
                            selection_operator_name='roulette wheel'
                        )
            case 'sus':  # abbreviation for stochastic universal sampling
                self.stochastic_universal_sampling()
                match crossover_operator:
                    case 'single point':
                        self.perform_crossover(
                            crossover_operator=single_point_crossover,
                            selection_operator_name='sus'
                        )
                    case 'uniform':
                        self.perform_crossover(
                            crossover_operator=uniform_crossover,
                            selection_operator_name='sus'
                        )
                    case 'plco':
                        self.perform_crossover(
                            crossover_operator=plco,
                            selection_operator_name='sus'
                        )

        """Secondly, we create the new generation with children being a result od selection and crossover operators
        on the current population:"""
        new_generation = Generation(size=self.pop_size)

        for pair in self.current_children[0].get('children'):
            new_generation.add_member(genome=pair[0])
            new_generation.add_member(genome=pair[1])

        """Thirdly, we add the elite - it doesn't matter that it's at the end of the new generation, because it'll be
        sorted anyway after new Members evaluation."""
        index = 0
        while index < self.elite_size:
            new_generation.add_member(
                genome=self.current_generation.members[self.current_fitness_ranking[index].get('index')].genes
            )
            new_generation.add_member(
                genome=self.current_generation.members[self.current_fitness_ranking[index + 1].get('index')].genes
            )
            index += 2

        """Finally, we overwrite the current generation with the new one:"""
        self.current_generation = new_generation

    def mutate(self):
        """Mutation probability is the probability of 'resetting' a member of the current generation, i.e. changing
        it genome randomly. For optimisation purposes instead of a loop over the whole generation, I calculate the
        number of members to be mutated and then generate pseudo-randomly a list of member indexes in the current
        generation to be mutated.
        """
        number_of_mutations = floor(self.mutation_prob * self.current_generation.size)

        """Size of generation is a constant, it has to be adjusted to the lack of elite; 
        after all we want to mutate all but the elite members
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

    def reset_parents(self):
        self.current_parents = []

    def reset_children(self):
        self.current_children = []

    def change_population_size(self, pop_size):
        self.pop_size = pop_size

    def fitness_plot(self):
        historic_best_fits = []
        for old_fitness_ranking in self.fitness_rankings:
            historic_best_fits.append(old_fitness_ranking[0].get('fitness value'))

        generation_indexes = np.arange(start=0, stop=len(historic_best_fits), step=1)

        plt.plot(generation_indexes, historic_best_fits)
        plt.show()
