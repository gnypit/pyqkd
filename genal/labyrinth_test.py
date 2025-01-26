"""Jakub T. Gnyp, gnyp.jakub@gmail.com
Labyrinth test case for the Genetic Algorithm - script with functions for visualizations.
Required program: ImageMagick https://www.techspot.com/downloads/5515-imagemagick.html
Alternatively one can use Pillow instead.
"""
import genetic_algorithm
import selection_operators
import crossover_operators
import pygad
import random
import tqdm
import copy
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from time import time

"""The labyrinth is encoded using a matrix with elements 0 and 1. A zero represents an allowed field that can be 
traversed, while a black field represents a wall; these fields cannot be entered. Including the walls on the boundary of 
the labyrinth, its dimensions are 12x12, so the coordinates in Python will be numbered from 0 to 11. The entrance is 
located at field (1,1), while the exit is at field (10,10)."""

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

"""The movements through the labyrinth are encoded as follows:  
0 - no movement;  
1 - move left;  
2 - move right;  
3 - move up;  
4 - move down.  

Additionally, a dictionary has been created to map this encoding to the corresponding coordinate changes:  
`moves_mapping = {movement: (change in y, change in x), ...}`  
`y` is the first coordinate in the matrix, as it represents the row number!"""
gene_space = [0, 1, 2, 3, 4]
moves_mapping = {
    0: (0, 0),  # bez ruchu
    1: (0, -1),  # w lewo
    2: (0, 1),  # w prawo
    3: (-1, 0),  # w górę
    4: (1, 0)  # w dół
}


def draw_labyrinth(plot_object, labyrinth: np.ndarray):
    """Function draws an empty labyrinth (no path) as dictated by 'labyrinth' matrix, on a provided graphical object
    'plot_object'."""
    for i in range(len(labyrinth)):
        """Iterating over rows"""
        for j in range(len(labyrinth[i])):
            """Iterating over columns"""
            if labyrinth[i, j] == 1:
                """Drawing walls"""
                rect = patches.Rectangle(
                    (j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='black'
                )
            else:
                """Drawing empty fields"""
                rect = patches.Rectangle(
                    (j, i), 1, 1, linewidth=1, edgecolor='grey', facecolor='white'
                )
            plot_object.add_patch(rect)

    """Some final settings"""
    plot_object.set_xlim(0, 12)
    plot_object.set_ylim(0, 12)
    plot_object.invert_yaxis()
    plot_object.set_aspect('equal')


def see_route(labyrinth: np.ndarray, moves_mapping: dict, steps: list,
              gif_filename='labirynt.gif', summary_filename='labirynt_summary.png'):
    """Function takes on input a matrix representing the labyrinth, dict mapping moves inside the labyrinth to changes
    of coordinates — in matrices from numpy we first have the row (y) coordinate, and then the column coordinate (x)!

    Results of this function are a GIF with a step-by-step animation of a path in the labyrinth
    and a PNG with the whole path."""
    start_pos = (1, 1)
    fig, ax = plt.subplots()

    """Drawing en empty labyrinth"""
    draw_labyrinth(plot_object=ax, labyrinth=labyrinth)

    """Creating a path based on the following steps"""
    path_x, path_y = [start_pos[1] + 0.5], [start_pos[0] + 0.5]  # path starts in the middle of a field, hence 0.5
    pos = list(start_pos)
    for step in steps:
        move = moves_mapping[int(step)]
        pos[0] += move[0]
        pos[1] += move[1]
        path_x.append(pos[1] + 0.5)  # 0.5 shift, so that lines are "centered", not along borders of fields
        path_y.append(pos[0] + 0.5)  # 0.5 shift, so that lines are "centered", not along borders of fields

    def update(num, path_x, path_y, line):
        """Function for updating the animation."""
        line.set_data(path_x[:num], path_y[:num])
        return line,

    """Initiating a line of the path"""
    line, = ax.plot([], [], lw=2, color='red')

    """Animation of steps with a time interval, specifying FPS in the final GIF."""
    ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=[path_x, path_y, line], interval=200,
                                  blit=True)

    """Axis settings"""
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.gca().invert_yaxis()

    """Saving animation as a GIF"""
    ani.save(gif_filename, writer='imagemagick')

    """Drawing the whole path on a PNG"""
    fig, ax = plt.subplots()  # again using local variables
    draw_labyrinth(plot_object=ax, labyrinth=labyrinth)  # drawing an empty labyrinth anew
    ax.plot(path_x, path_y, lw=2, color='red')  # adding the whole generated path

    """Axis settings"""
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.gca().invert_yaxis()

    """Saving the PNG"""
    plt.savefig(summary_filename)
    plt.close()

def example():
    """Example of applying the functions for labyrinth visualisation"""
    steps = [2, 2, 4, 4, 4, 1, 4, 2, 2, 1, 4, 4, 1, 1, 1, 4, 2, 2, 4, 2, 4, 2]
    see_route(steps=steps, labyrinth=labyrinth, moves_mapping=moves_mapping)
    print("see_route executed")

    # Stworzę jeszcze pusty labirynt do analizy
    fig, ax = plt.subplots()
    draw_labyrinth(plot_object=ax, labyrinth=labyrinth)
    plt.savefig("empty_labyrinth.png")
    print("Empty labyrinth created.")

"""Settings for the genetic algorithm (pygad approach)"""
exit_labyrinth = {'y': 10, 'x': 10}  # target/exit coordinates of the labyrinth
num_generations = 4000
sol_per_pop = 500
num_parents_mating = 250
num_genes = 30  # number of steps allowed to take in the labyrinth to find the exit (20 is the shortest path)
selection = "tournament"
mutation = "random"
mutation_prob = 0.15
k_tournament = 10
stop_criteria = "reach_1"

"""Weights of bonuses and punishments:"""
bonus_point = 2
pos_repeat_point = 1  # punishment for repeating a position
hitting_a_wall_point = 1.25  # punishment for wasting a move to "bounce back" from a wall
max_bonus = 10 * bonus_point  # if the GA follows the shortest path, it can wait up to 10 moves at the exit


def fitness_fun_pygad(genetic_algorithm_instance, route, route_idx):
    """We're using the Taxi metric to evaluate path through the labyrinth. Additionally, we give bonuses
    and punishments for specific behaviours, so that path suggested by chromosomes were as close to actually possible
    ones as possible, taking into account "bouncing back" from walls.
    """
    position = {'y': 1, 'x': 1}  # we're starting in (1,1)

    """Not to encounter the 'mutable' attribute of dicts, in the position history list we save copies of positions
    after each step:"""
    history = [copy.deepcopy(position)]
    is_probem = 0  # starting number of unwanted behaviours
    bonus = 0  # starting bonus value

    for move in route:  # changing position based on moves

        if position.get('x') == exit_labyrinth.get('x') and position.get('y') == exit_labyrinth.get('y') and move == 0:
            bonus += bonus_point  # bonus for staying at the exit
            continue

        new_y, new_x = position.get('y') + moves_mapping.get(move)[0], position.get('x') + moves_mapping.get(move)[1]

        """Checkin, if the new position is an allowed field:"""
        if labyrinth[new_y, new_x] == 0:
            position['x'], position['y'] = new_x, new_y
            history.append(copy.deepcopy(position))

            """Checking, if a punishment for repeating the position is required:"""
            if history.count(position) > 1:
                is_probem += pos_repeat_point
        else:  # field, onto which the chromosome want to step, is not allowed!
            is_probem += hitting_a_wall_point

    """For the final fitness value, first we calculate some additional variables:"""
    x_distance = abs(exit_labyrinth.get('x') - position.get('x'))
    y_distance = abs(exit_labyrinth.get('y') - position.get('y'))
    sum_exit_coordinates = exit_labyrinth.get('x') + exit_labyrinth.get('y')

    """Actual fitness value, max 1:"""
    fitness_val = (sum_exit_coordinates - x_distance - y_distance) * 2  # using Taxi metric
    fitness_val += bonus  # adding bonus for waiting at the end
    fitness_val -= is_probem  # subtracting punishments
    fitness_val = fitness_val / (sum_exit_coordinates * 2 + max_bonus)

    return fitness_val


def fitness_fun_pyqkd(route):
    """We're using the Taxi metric to evaluate path through the labyrinth. Additionally, we give bonuses
    and punishments for specific behaviours, so that path suggested by chromosomes were as close to actually possible
    ones as possible, taking into account "bouncing back" from walls.
    """
    position = {'y': 1, 'x': 1}  # we're starting in (1,1)

    """Not to encounter the 'mutable' attribute of dicts, in the position history list we save copies of positions
    after each step:"""
    history = [copy.deepcopy(position)]

    for move in route:  # changing position based on moves
        new_y, new_x = position.get('y') + moves_mapping.get(move)[0], position.get('x') + moves_mapping.get(move)[1]

        """Checking, if the new position is inside the labyrinth:"""
        if 1 <= new_x <= 10 and 1 <= new_y <= 10:
            position['x'], position['y'] = new_x, new_y
            history.append(copy.deepcopy(position))

    """For the final fitness value, first we calculate some additional variables:"""
    x_distance = abs(exit_labyrinth.get('x') - position.get('x'))
    y_distance = abs(exit_labyrinth.get('y') - position.get('y'))
    sum_exit_coordinates = exit_labyrinth.get('x') + exit_labyrinth.get('y')

    """Actual fitness value, max 1:"""
    fitness_val = sum_exit_coordinates - x_distance - y_distance # using Taxi metric
    fitness_val = fitness_val / sum_exit_coordinates

    return fitness_val


def generator(args):
    genome, length = args[0], args[1]
    return random.choices(genome, k=length)


def main_pygad():
    """PyGAD approach to solving the labyrinth"""
    fitness_list = []
    times = []
    output_list = []
    generations_no = []  # number of generations in which a single GA found it's proposition for a solution

    for i in tqdm.tqdm(range(10)):
        start = time()

        ga_instance = pygad.GA(
            gene_space=gene_space,
            num_genes=num_genes,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_fun_pygad,
            sol_per_pop=sol_per_pop,
            parent_selection_type=selection,
            mutation_type=mutation,
            mutation_probability=mutation_prob,
            stop_criteria=stop_criteria,
            suppress_warnings=True,
            K_tournament=k_tournament
        )

        ga_instance.run()
        end = time()
        times.append(end - start)

        """Manually visualising fitness history through generations."""
        fig, ax = plt.subplots()
        fitness = ga_instance.best_solutions_fitness
        generations = list(range(len(fitness)))

        ax.plot(generations, fitness, color="lime", linewidth=4, drawstyle='steps-post', label='Fitness')

        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title("PyGAD - Generations vs. Fitness")
        ax.legend()  # żeby mieć pewność, że legenda się wyświetli
        ax.grid(True)
        plt.show()

        """Remembering the solution:"""
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        generations_no.append(ga_instance.best_solution_generation)
        fitness_list.append(solution_fitness)
        output_list.append(solution)

        """Visualising results (routes, which were tried by chromosomes)"""
        gif_filename = 'chromosome_animation' + str(i) + '.gif'
        picture_filename = 'chromosome_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=output_list[-1],
                  gif_filename=gif_filename, summary_filename=picture_filename)

        """Visualising the actual routes, not allowing to walk through the walls"""
        x, y = 1, 1
        history = []

        for step in output_list[-1]:
            new_y, new_x = y + moves_mapping.get(step)[0], x + moves_mapping.get(step)[1]
            if 0 <= new_y <= 11 and 0 <= new_x <= 11:
                """After verifying that new coordinates are inside the labyrinth (in the matrix at all),
                we check if they are an allowed field:"""
                if labyrinth[new_y, new_x] == 0:
                    x, y = new_x, new_y
                    history.append(step)
                else:
                    history.append(0)
            else:
                print(f"We got coordinates x={new_x} and y={new_y} from outside the labyrinth (when drawing).")

        gif_filename = 'actual_route_animation' + str(i) + '.gif'
        picture_filename = 'actual_route_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=history,
                  gif_filename=gif_filename, summary_filename=picture_filename)

    print(f"Mean time of PyGAD's GA running: {np.mean(times)}")
    print(f"Mean fitness value of the PyGAD's GA best solutions: {np.mean(fitness_list)}")
    print(f"Mean number of generations in the PyGAD's GA to get the best solution: {np.mean(generations_no)}")

    print(f"Results history: \n")
    for j in range(len(output_list)):
        print(output_list[j])


def main_pyqkd():
    """pyqkd.genal approach to solving the labyrinth"""
    fitness_list = []
    times = []
    output_list = []
    generations_no = []  # number of generations in which a single GA found it's proposition for a solution

    for i in tqdm.tqdm(range(10)):
        start = time()

        ga_instance = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=16,
            number_of_generations=10,
            elite_size=2,
            args={
                'genome': (gene_space, num_genes),
                'selection': 4,
                'crossover': None
            },
            fitness_function=fitness_fun_pyqkd,
            genome_generator=genetic_algorithm.uniform_gene_generator,
            selection=selection_operators.tournament_selection,
            crossover=crossover_operators.single_point_crossover,
            pool_size=4,
            no_parents_pairs=8,
            mutation_prob=float(1/16)
        )

        ga_instance.run()
        end = time()
        times.append(end - start)

        """Manually visualising fitness history through generations."""
        fig, ax = plt.subplots()
        fitness = ga_instance.best_fit_history
        generations = list(range(len(fitness)))

        ax.plot(generations, fitness, color="lime", linewidth=4, drawstyle='steps-post', label='Fitness')

        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title("pyqkd - Generations vs. Fitness")
        ax.legend()  # żeby mieć pewność, że legenda się wyświetli
        ax.grid(True)
        plt.show()

        """Remembering the solution:"""
        solution, solution_fitness = ga_instance.best_solution()
        # generations_no.append(ga_instance.best_solution_generation)
        fitness_list.append(solution_fitness)
        output_list.append(solution)

        """Visualising results (routes, which were tried by chromosomes)"""
        gif_filename = 'pyqkd_chromosome_animation' + str(i) + '.gif'
        picture_filename = 'pyqkd_chromosome_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=output_list[-1],
                  gif_filename=gif_filename, summary_filename=picture_filename)

        """Visualising the actual routes, not allowing to walk through the walls"""
        x, y = 1, 1
        history = []

        for step in output_list[-1]:
            new_y, new_x = y + moves_mapping.get(step)[0], x + moves_mapping.get(step)[1]
            if 0 <= new_y <= 11 and 0 <= new_x <= 11:
                """After verifying that new coordinates are inside the labyrinth (in the matrix at all),
                we check if they are an allowed field:"""
                if labyrinth[new_y, new_x] == 0:
                    x, y = new_x, new_y
                    history.append(step)
                else:
                    history.append(0)
            else:
                print(f"We got coordinates x={new_x} and y={new_y} from outside the labyrinth (when drawing).")

        gif_filename = 'pyqkd_actual_route_animation' + str(i) + '.gif'
        picture_filename = 'pyqkd_actual_route_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=history,
                  gif_filename=gif_filename, summary_filename=picture_filename)

    print(f"Mean time of pyqkd's GA running: {np.mean(times)}")
    print(f"Mean fitness value of the pyqkd's GA best solutions: {np.mean(fitness_list)}")
    # print(f"Mean number of generations in the PyGAD's GA to get the best solution: {np.mean(generations_no)}")

    print(f"Results history: \n")
    for j in range(len(output_list)):
        print(output_list[j])
        print(fitness_list[j])


if __name__ == '__main__':
    # main_pygad()
    main_pyqkd()
    # fitness_fun_pyqkd([1, 1, 4, 3, 4, 3, 3, 2, 4, 0, 0, 1, 1, 1, 1, 2, 0, 4, 0, 2, 2, 2, 1, 2, 2, 3, 4, 3, 2, 4])
