import genetic_algorithm
import selection_operators
import crossover_operators
import pygad
import random
import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from time import time

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


def draw_labyrinth(plot_object, labyrinth: np.ndarray):
    """Funkcja rysująca pusty labirynt (bez trasy) zgodnie z macierzą 'labyrinth', na podanym obiekcie
    graficznym 'plot_object'.
    """
    for i in range(len(labyrinth)):
        """Iteracja po kolejnych wierszach"""
        for j in range(len(labyrinth[i])):
            """Iteracja po kolejnych kolumnach"""
            if labyrinth[i, j] == 1:
                """Rysowanie ściany"""
                rect = patches.Rectangle(
                    (j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='black'
                )
            else:
                """Rysowanie pustego pola"""
                rect = patches.Rectangle(
                    (j, i), 1, 1, linewidth=1, edgecolor='grey', facecolor='white'
                )
            plot_object.add_patch(rect)

    """Końcowe ustawienia"""
    plot_object.set_xlim(0, 12)
    plot_object.set_ylim(0, 12)
    plot_object.invert_yaxis()
    plot_object.set_aspect('equal')


def see_route(labyrinth: np.ndarray, moves_mapping: dict, steps: list,
              gif_filename='labirynt.gif', summary_filename='labirynt_summary.png'):
    """Funkcja przyjmująca na wejściu macierz reprezentującą labirynt (labyrinth), słownik dopasowujący kod ruchu
    do zmiany odpowiednich współrzędnych — w macierzy z biblioteki numpy najpierw jest wsp. wiersza, a następnie wsp.
    kolumny!

    Wynikiem funkcji jest animacja GIF danej trasy przez labirynt oraz grafika PNG z podsumowaniem całej trasy.
    """
    start_pos = (1, 1)
    fig, ax = plt.subplots()

    """Rysowanie pustego labiryntu"""
    draw_labyrinth(plot_object=ax, labyrinth=labyrinth)

    """Tworzenie trasy na podstawie kroków"""
    path_x, path_y = [start_pos[1] + 0.5], [start_pos[0] + 0.5]  # ścieżka zaczyna się w środku pola wejścia, stąd 0.5
    pos = list(start_pos)
    for step in steps:
        move = moves_mapping[int(step)]
        pos[0] += move[0]
        pos[1] += move[1]
        path_x.append(pos[1] + 0.5)  # przesunięcie o 0.5, żeby linie były "wycentrowane", a nie wzdłuż krawędzi pól
        path_y.append(pos[0] + 0.5)  # przesunięcie o 0.5, żeby linie były "wycentrowane", a nie wzdłuż krawędzi pól

    def update(num, path_x, path_y, line):  # wewnątrz funkcji można tworzyć "lokalną" funkcję
        """Funkcja do aktualizacji animacji"""
        line.set_data(path_x[:num], path_y[:num])
        return line,

    """Inicjalizacja linii trasy"""
    line, = ax.plot([], [], lw=2, color='red')

    """Animacja trasy wzdłuż odpowiednich współrzędnych z zadanym interwałem, określającym jak szybko zmieniają się
    klatki w końcowym GIFie."""
    ani = animation.FuncAnimation(fig, update, frames=len(path_x), fargs=[path_x, path_y, line], interval=200,
                                  blit=True)

    """Ustawienia osi"""
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.gca().invert_yaxis()

    """Zapisanie animacji jako GIF"""
    ani.save(gif_filename, writer='imagemagick')
    # print(f"Wygenerowano animację trasy przez labirynt")

    """Rysowanie całej trasy na grafice PNG"""
    fig, ax = plt.subplots()  # wykorzystuję ponownie zmienne lokalne
    draw_labyrinth(plot_object=ax, labyrinth=labyrinth)  # rysujemy pusty labirynt od nowa
    ax.plot(path_x, path_y, lw=2, color='red')  # dodajemy całą wygenerowaną trasę

    """Ustawienia osi"""
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.gca().invert_yaxis()

    """Zapisanie obrazu jako PNG"""
    plt.savefig(summary_filename)
    plt.close()
    # print(f"Wykonano grafikę z całą trasą przez labirynt")


"""Ruchy przez labirynt zakodowano w następujący sposób:
0 - bez ruchu;
1 - w lewo;
2 - w prawo;
3 - w górę;
4 - w dół.

Dodatkowo, stworzono słownik mapujący to kodowanie do zmian odpowiednich współrzędnych.
moves_mapping = {ruch: (zmiana y, zmiana x), ...}
y jest pierwszą wsp. w macierzy, ponieważ to nr wiersza!
"""
gene_space = [0, 1, 2, 3, 4]
moves_mapping = {
    0: (0, 0),  # bez ruchu
    1: (0, -1),  # w lewo
    2: (0, 1),  # w prawo
    3: (-1, 0),  # w górę
    4: (1, 0)  # w dół
}

"""Ustawienia algorytmu genetycznego"""
exit_labyrinth = {'y': 10, 'x': 10}  # współrzędne "wyjścia" z labiryntu
num_generations = 4000
sol_per_pop = 500
num_parents_mating = 250
num_genes = 30
selection = "tournament"
mutation = "random"
mutation_prob = 0.15
k_tournament = 10
stop_criteria = "reach_1"

"""Wagi punktów nagród & kar:"""
bonus_point = 2  # do nagród
pos_repeat_point = 1  # do kary za powtórzenie pozycji
hitting_a_wall_point = 1.25  # do kary za zmarnowanie ruchu na odbicie się od ściany
max_bonus = 10 * bonus_point  # maksymalnie 10 kroków czekamy w mecie, do której można dotrzeć w 20 z 30 kroków


def fitness_fun_pygad(genetic_algorithm_instance, route, route_idx):
    """Używamy metryki Taxi do ewaluacji tras przez labirynt. Dodatkowo, przydzielamy kary i nagrody za poszczególne
    zachowania, aby trasy proponowane przez chromosomy były jak najbliższe tym faktycznym, po uwzględnieniu
    "odbijania się" od ścian.
    """
    position = {'y': 1, 'x': 1}  # zaczynamy w (1,1)

    """Aby uniknąć kłopotu z cechą 'mutable' słowników, zapamiętujemy w liście historii położeń kopię
    początkowego stanu słownika położeń, zamiast przypisywać do listy dynamiczną strukturę danych.
    """
    history = [copy.deepcopy(position)]
    is_probem = 0  # początkowa wartość licznika problemów, do którego przydzielamy punkty kar
    bonus = 0  # początkowa wartość bonusu, do której dodajemy punkty nagród

    for move in route:  # zmieniamy położenie w zależności od wykonanego ruchu

        if position.get('x') == exit_labyrinth.get('x') and position.get('y') == exit_labyrinth.get('y') and move == 0:
            bonus += bonus_point  # bonus za pozostanie w mecie
            continue

        new_y, new_x = position.get('y') + moves_mapping.get(move)[0], position.get('x') + moves_mapping.get(move)[1]

        """Sprawdzamy, czy nowe współrzędne wskazują na dozwolone pole:"""
        if labyrinth[new_y, new_x] == 0:
            position['x'], position['y'] = new_x, new_y
            history.append(copy.deepcopy(position))

            """Sprawdzamy, czy trzeba przydzielić karę za powtórzenie pozycji:"""
            if history.count(position) > 1:
                is_probem += pos_repeat_point
        else:  # pole, na które chce wejść chromosom, nie jest dozwolone!
            is_probem += hitting_a_wall_point

    """Najpierw obliczamy pomocnicze zmienne, dla czytelności:"""
    x_distance = abs(exit_labyrinth.get('x') - position.get('x'))
    y_distance = abs(exit_labyrinth.get('y') - position.get('y'))
    sum_exit_coordinates = exit_labyrinth.get('x') + exit_labyrinth.get('y')

    """Faktyczna wartość fitnessu, maksymalnie 1:"""
    fitness_val = (sum_exit_coordinates - x_distance - y_distance) * 2  # użycie metryki taxi
    fitness_val += bonus  # dodajemy punkty nagród za czekanie w mecie "do końca"
    fitness_val -= is_probem  # odejmujemy punkty kar
    fitness_val = fitness_val / (sum_exit_coordinates * 2 + max_bonus)

    return fitness_val


def fitness_fun_pyqkd(generation_member: genetic_algorithm.Member):
    """Używamy metryki Taxi do ewaluacji tras przez labirynt. Dodatkowo, przydzielamy kary i nagrody za poszczególne
    zachowania, aby trasy proponowane przez chromosomy były jak najbliższe tym faktycznym, po uwzględnieniu
    "odbijania się" od ścian.
    """
    route = generation_member.genome
    position = {'y': 1, 'x': 1}  # zaczynamy w (1,1)

    """Aby uniknąć kłopotu z cechą 'mutable' słowników, zapamiętujemy w liście historii położeń kopię
    początkowego stanu słownika położeń, zamiast przypisywać do listy dynamiczną strukturę danych.
    """
    history = [copy.deepcopy(position)]
    is_probem = 0  # początkowa wartość licznika problemów, do którego przydzielamy punkty kar
    bonus = 0  # początkowa wartość bonusu, do której dodajemy punkty nagród

    for move in route:  # zmieniamy położenie w zależności od wykonanego ruchu

        if position.get('x') == exit_labyrinth.get('x') and position.get('y') == exit_labyrinth.get('y') and move == 0:
            bonus += bonus_point  # bonus za pozostanie w mecie
            continue

        new_y, new_x = position.get('y') + moves_mapping.get(move)[0], position.get('x') + moves_mapping.get(move)[1]

        """Sprawdzamy, czy nowe współrzędne wskazują na dozwolone pole:"""
        if labyrinth[new_y, new_x] == 0:
            position['x'], position['y'] = new_x, new_y
            history.append(copy.deepcopy(position))

            """Sprawdzamy, czy trzeba przydzielić karę za powtórzenie pozycji:"""
            if history.count(position) > 1:
                is_probem += pos_repeat_point
        else:  # pole, na które chce wejść chromosom, nie jest dozwolone!
            is_probem += hitting_a_wall_point

    """Najpierw obliczamy pomocnicze zmienne, dla czytelności:"""
    x_distance = abs(exit_labyrinth.get('x') - position.get('x'))
    y_distance = abs(exit_labyrinth.get('y') - position.get('y'))
    sum_exit_coordinates = exit_labyrinth.get('x') + exit_labyrinth.get('y')

    """Faktyczna wartość fitnessu, maksymalnie 1:"""
    fitness_val = (sum_exit_coordinates - x_distance - y_distance) * 2  # użycie metryki taxi
    fitness_val += bonus  # dodajemy punkty nagród za czekanie w mecie "do końca"
    fitness_val -= is_probem  # odejmujemy punkty kar
    fitness_val = fitness_val / (sum_exit_coordinates * 2 + max_bonus)

    return fitness_val


def generator(args):
    genome, length = args[0], args[1]
    return random.choices(genome, k=length)


def main_pygad():
    """Główna funkcja wykonująca, aplikująca algorytm genetyczny do labiryntu zgodnie z ustawieniami w zmiennych
    globalnych oraz z wykorzystaniem zdefiniowanych w skrypcie `projekt01_labirynt_wizualizacje.py` funkcji
    do wizualizacji.
    """
    fitness_list = []
    times = []
    output_list = []
    generations_no = []  # nr generacji w danej iteracji, w której osiągnięto najlepsze rozwiązanie

    for i in tqdm.tqdm(range(10)):
        start = time()  # sprawdzamy czas na starcie

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

        ga_instance.run()  # uruchamiamy algorytm genetyczny
        end = time()  # mierzymy czas na koniec
        times.append(end - start)

        """Ręcznie wizualizujemy historię fitnessu na przestrzeni generacji."""
        fig, ax = plt.subplots()  # tworzymy osobną figurę na wykresy historii fitnessu!
        fitness = ga_instance.best_solutions_fitness  # wartości na oś 0y
        generations = list(range(len(fitness)))  # wartości na oś 0x

        ax.plot(generations, fitness, color="lime", linewidth=4, drawstyle='steps-post', label='Fitness')

        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title("PyGAD - Generations vs. Fitness")
        ax.legend()  # żeby mieć pewność, że legenda się wyświetli
        ax.grid(True)
        plt.show()

        """Zapamiętujemy parametry rozwiązania:"""
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        generations_no.append(ga_instance.best_solution_generation)
        fitness_list.append(solution_fitness)
        output_list.append(solution)

        """Wizualizujemy wyniki (trasy, które chromosomy chciały przejść)"""
        gif_filename = 'chromosome_animation' + str(i) + '.gif'
        picture_filename = 'chromosome_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=output_list[-1],
                  gif_filename=gif_filename, summary_filename=picture_filename)

        """Wizualizujemy faktyczną trasę, z pominięciem kroków polegających na wejściu na pole niedozwolone"""
        x, y = 1, 1
        history = []

        for step in output_list[-1]:
            new_y, new_x = y + moves_mapping.get(step)[0], x + moves_mapping.get(step)[1]
            if 0 <= new_y <= 11 and 0 <= new_x <= 11:
                """Po zweryfikowaniu, że nowe współrzędne są wewnątrz labiryntu (tzn. mieszczą się w macierzy),
                sprawdzamy, czy reprezentują dozwolone pole:
                """
                if labyrinth[new_y, new_x] == 0:
                    x, y = new_x, new_y
                    history.append(step)
                else:
                    history.append(0)
            else:
                print(f"Dostaliśmy współrzędne x={new_x} oraz y={new_y} poza labiryntem (przy wizualizacji).")

        gif_filename = 'actual_route_animation' + str(i) + '.gif'
        picture_filename = 'actual_route_picture' + str(i) + '.png'
        see_route(labyrinth=labyrinth, moves_mapping=moves_mapping, steps=history,
                  gif_filename=gif_filename, summary_filename=picture_filename)

    print(f"Średni czas działania algorytmu genetycznego: {np.mean(times)}")
    print(f"Średnia wartość f. fitnessu najlepszego rozwiązania: {np.mean(fitness_list)}")
    print(f"Średnia liczba generacji do otrzymania najlepszego rozwiązania: {np.mean(generations_no)}")

    print(f"Historia wyników: \n")
    for j in range(len(output_list)):
        print(output_list[j])


def main_pyqkd():
    test = genetic_algorithm.Population(
        initial_pop_size=4,
        fit_fun=fitness_fun_pyqkd,
        genome_generator=generator,
        elite_size=0,
        args={
            'genome': (gene_space, num_genes),
            'selection': None,
            'crossover': None
        },
        selection_operator=selection_operators.ranking_selection,
        crossover_operator=crossover_operators.uniform_crossover,
        no_parents_pairs=2,
        mutation_prob=mutation_prob,
        number_of_generations=10
    )
    test.run()


if __name__ == '__main__':
    main_pyqkd()
