import numpy as np
import genetic_algorithm
import random


def fitness_fun(chromosome, route_idx=None):
    route = chromosome.genes

    # Zaczynamy w (1, 1)
    i, j = 1, 1
    position = [0]
    is_problem = 0
    zero_no = 0
    bonus = 0

    for move in route:  # Zmieniamy położenie w zależności od wykonanego ruchu

        if position[-1] == [10, 10] and move == 0:  # bonus za pozostanie w mecie
            bonus += 1
            continue

        if move == 1:
            new_j, new_i = j - 1, i
        elif move == 2:
            new_j, new_i = j + 1, i
        elif move == 3:
            new_i, new_j = i - 1, j
        elif move == 4:
            new_i, new_j = i + 1, j
        else:
            zero_no += 1
            new_i, new_j = i, j

        if 0 <= new_i <= 11 and 0 <= new_j <= 11:
            if labyrinth[new_i, new_j] == 0:
                i, j = new_i, new_j
                position.append([i, j])
                if position.count([i, j]) > 1:  # kara za powtórzenie pozycji
                    is_problem += 1
            else:
                is_problem += 1.25  # kara za zmarnowanie ruchu na odbicie się od ściany

    fitness = ((20 - abs(10 - i) - abs(10 - j)) * 2 - is_problem - zero_no + bonus * 2) / 56

    return fitness


def genome_generator(args):
    """Simple function for generating a given number of randomly sequenced genes"""
    space, amount = args[0], args[1]
    genome = random.choices(population=space, k=amount)
    return genome


labyrinth = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

gene_space = [0, 1, 2, 3, 4]
num_generations = 3000
sol_per_pop = 500
num_parents_mating = 250
elite_size = 20
num_genes = 30

population = genetic_algorithm.Population(
    pop_size=500,
    fit_fun=fitness_fun,
    genome_generator=genome_generator,
    args=[gene_space, num_genes],
    elite_size=elite_size,
    mutation_prob=0.25
)

solutions = []
index = 0

for _ in range(num_generations):
    print("Generation number: {}".format(index))
    index += 1
    population.evaluate_generation()
    print("Population evaluated")
    current_solution = population.best_fit()
    print("Best fit found")
    solutions.append(current_solution)
    population.create_new_generation(selection_operator='ranking', crossover_operator='uniform')
    print("New generation created")
    population.mutate()
    print("New generation mutated")

    population.reset_parents()
    population.reset_children()