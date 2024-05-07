"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import multiprocessing
import random
from multiprocessing import Process

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import binomial as np_binom
from genalqkd import GeneticAlgorithm
from scipy.special import binom as newton_binom  # wyrażenie Newtona (n nad k)
from bb84 import simulation_bb84


def run_simulations(process_id, generation, flag, barrier_beginning, barrier_end):
    chromosome_number = 0
    while chromosome_number <= len(generation):
        """Pracujemy na chromosomach które mają indeksy chromosome_number + number_of_processes"""


if __name__ == '__main__':
    """Since performing simulations of QKD protocols is quite heavy computing wise, within the genetic algorithm
    every population of chromosomes (potential solutions to our optimisation problem) will be divided into subsets.
    Each subset of chromosomes will be simulated as a separate process, on a separate processor core, in parallel.
    
    For this purpose we start with assesing number of available cores and setting starting method of the processes.
    """
    number_of_processes = multiprocessing.cpu_count()  # each core will perform one of the processes
    multiprocessing.set_start_method('spawn', force=True)

    """Processes have to be initiatied, started from a barrier for a single population, halted at the barrier, etc.,
    until the genetic algorithm converges to an accepted solution of fulfils stopping criteria of maximum number
    of either epochs in general or consecutive epochs with the same maximal fitness value. Then they have to be joined
    and shut down.
    
    Manager helps us monitoring progress and support us in joining and shutting the processes down in a tidy manner. 
    """
    with multiprocessing.Manager() as manager:
        """Spawning processes with manager allows automated joining once they are finished. We monitor that with
        a flag and proper barriers."""
        continue_flag = manager.Value('b', True)
        work_start = multiprocessing.Barrier(number_of_processes + 1)  # no. processes + the main process
        work_complete = multiprocessing.Barrier(number_of_processes + 1)
        processes = []  # a list to hold the processes

    for gain in [0.9, 0.95, 1.0]:
        for dist_prob in [0.01, 0.05, 0.10, 0.15]:
            for i in range(10):  # small for debugging & testing
                gen_al = GeneticAlgorithm(pop_size=20,
                                          max_basis_length=4000,
                                          channel_gain=gain,
                                          disturbance_prob=dist_prob,
                                          fit_fun='factored_fit',
                                          elite_size=2,
                                          max_cascade_pass=8,
                                          seed=261135)
                epoch = 1
                max_iter = 50
                best_fit = -10.0  # big, negative number, surely to be replaced during 1st generation
                best_fit_genes_sequence = []  # the actual estimated solution to the problem
                fit_history = []
                fit_plot = []
                while epoch <= max_iter:
                    """First step in each iteration is evaluating fitness, which comes down to performing simulations
                    of QKD protocol on chromosomes in a current population:"""
                    for p in range(number_of_processes):
                        proc = Process(
                            target=run_simulations,
                            args=[p, gen_al.generation]
                        )
                        proc.start()
                        processes.append(proc)

                    """We find genes of the chromosome with best fit val."""
                    current_solution = gen_al.best_fit()
                    current_genes = current_solution[0]
                    current_fit_val = current_solution[1]  # we also want to know its fit value

                    """We want to plot the fit values and save to a csv file both the genes and the fit val."""
                    fit_history.append(str(current_genes) + ', ' + str(current_fit_val))
                    fit_plot.append(current_fit_val)

                    """Now for the global solution search:"""
                    if best_fit < current_fit_val:
                        best_fit = current_fit_val
                        best_fit_genes_sequence = current_genes

                    """Checking for stopping criteria:"""
                    stop = False
                    for i in range(5):
                        if best_fit == fit_plot[-1 * (i + 1)]:
                            stop = True
                        else:
                            stop = False
                            break
                    if stop:
                        break

                    """Next generation creation, mutation & evaluation:"""
                    gen_al.ranking_selection(
                        crossover_operator=gen_al.single_point_crossover  # multiprocessing nr 2 tutaj?
                    )
                    gen_al.mutation()
                    epoch += 1

                evaluation_data = pd.DataFrame(fit_history)
                file_name = 'evaluation_data' + str(gain) + '_' + str(dist_prob) + '_' + str(i) + '.csv'
                evaluation_data.to_csv(file_name)

                print(best_fit)
                print(best_fit_genes_sequence)

                plt.figure()
                y = fit_plot
                x = np.linspace(1, max_iter, max_iter)

                plt.xlabel('epoch')
                plt.ylabel('fit value')
                plt.title('gain = ' + str(gain) + '; disturbance prob. = ' + str(dist_prob))
                plt.plot(x, y)
                plt.savefig('gain = ' + str(gain) + '; disturbance prob. = ' + str(dist_prob) + '.jpg')
                plt.close()
