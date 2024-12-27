"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
Selection operators in this script take a parent generation from a genetic algorithm
and return a child (rival) generation.
"""
import random
from genetic_algorithm import Generation  # TODO: make sure there are no cycling imports


def ranking_selection(parent_generation: Generation):  # deterministic
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
    while member_counter < parent_generation.size - parent_generation.elite_size:
        parent1 = parent_generation.members[parent_generation.fitness_ranking[member_counter].get('index')]
        parent2 = parent_generation.members[parent_generation.fitness_ranking[member_counter + 1].get('index')]
        parents_candidates.append({'parent1': parent1, 'parent2': parent2})
        member_counter += 2

    return parents_candidates


def roulette_wheel_selection(self):  # probability-based
    """The conspicuous characteristic of this selection method is the fact that it gives to each member i of the current
    generation a probability p(i) of being selected, proportional to its fitness value f(i). Each member in the
    generation occupies an area on the roulette wheel proportional to its fitness. Then, conceptually, the roulette
    wheel is spun as many times as the population size, each time selecting a member marked by the roulette-wheel
    pointer. Since the members are marked proportionally to their fitness, a member with a higher fitness is likely to
    receive more copies than a solution with a low fitness.

    For more details, please refer to 'Introduction to Evolutionary Computing', sect. 5.2.3 'Implementing Selection
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
    """It is an improved version of the roulette-wheel selection operator, as described in 'Introduction to
    Evolutionary Computing' in section 5.2.3. We begin similarly to the roulette wheel, because we still base
    the probabilities of selection on the cumulative probability distribution, associated with the fitness values.
    """
    fit_total = 0  # TODO: is it a problem when some fitness values are negative?
    cumulative_prob_distribution = []

    for i in range(self.pop_size):
        fit_total += self.current_fitness_ranking[i].get('fitness value')
        cumulative_prob_distribution.append(self.current_fitness_ranking[i].get('fitness value'))

    cumulative_prob_distribution = [fit / fit_total for fit in cumulative_prob_distribution]

    random_value = random.uniform(0, 1 / self.current_generation.size)  # that's too big a value...
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
        try:
            while random_value <= cumulative_prob_distribution[index]:  # TODO IndexError: list index out of range
                """We keep assigning given member as a new parent until the condition is met. This way we reflect
                the probability of it's selection."""
                new_parent = self.current_generation.members[self.current_fitness_ranking[index].get('index')]
                random_value += 1 / self.current_generation.size
                member_counter += 1
                parents.append(new_parent)
            index += 1
        except IndexError:
            """We've increased the index over the length of the current generation"""
            break

    """Now I rewrite parents into an organised list of pairs as dictionaries..."""
    index = 0
    while index < self.current_generation.size:
        parents_candidates.append({'parent1': parents[index], 'parent2': parents[index + 1]})
        index += 2

    """...and append it to the list of candidate parents lists:"""
    self.current_parents.append({'sus': parents_candidates})
