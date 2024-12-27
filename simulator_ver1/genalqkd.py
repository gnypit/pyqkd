"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""
import random
import tqdm
import numpy as np

from numpy.random import binomial as np_binom
from scipy.special import binom as newton_binom  # wyrażenie Newtona (n nad k)
from bb84 import simulation_bb84
from fitness_functions import factored_fit, fitness_negative_time, fitness_inv, evaluation


def sort_dict_by_fit(dictionary):
    return dictionary['fitness value']


def run_simulation(process_id, chromosome, protocol, quantum_gain, disturbance_prob, work_begin, work_end, flag):
    protocol_results = simulation_bb84(gain=quantum_gain,
                                       alice_basis_length=chromosome.genome.get('length'),
                                       rectilinear_basis_prob=chromosome.genome.get('rect_basis_prob'),
                                       disturbance_probability=disturbance_prob,
                                       publication_probability_rectilinear=chromosome.genome.get('pub_prob_rect'),
                                       publication_probability_diagonal=chromosome.genome.get('pub_prob_diag'),
                                       cascade_n_passes=chromosome.genome.get('no_pass'))
    chromosome.set_results(protocol_results)


class Chromosome:
    def __init__(self, length, rect_basis_prob, publication_prob_rectilinear, publication_prob_diagonal, no_pass,
                 fitness_function):
        """Genes = [number of Alice's basis choices (and so, qubits sent to Bob);
                    the probability of choosing a rectilinear basis - for both Alice and Bob, as in Lo, Chau, Adrehali;
                    publication probability of a bit measured in a rectilinear basis;
                    publication probability of a bit measured in a diagonal basis;
                    number of passes to be performed by CASCADE error correction algorithm]
        """
        self.genes = {
            'length': length,
            'rect_basis_prob': rect_basis_prob,
            'pub_prob_rect': publication_prob_rectilinear,
            'pub_prob_diag': publication_prob_diagonal,
            'no_pass': no_pass
        }
        self.qkd_results = {
            'error rate': None,
            'key length': None,
            'comp. cost': None,
            'no. del. bits': None,
            'no. cascade pass.': None
        }

    def set_results(self, results):
        self.qkd_results = results


class GeneticAlgorithm:
    def __init__(self, pop_size, max_basis_length, channel_gain, disturbance_prob, max_cascade_pass, elite_size,
                 target_key_length=256, mutation_prob=0.0, fit_fun='factored_fit', args={}, seed=None):
        """Variables have meaning:
        pop_size is the number of chromosomes in the (initial) population. It has to be a positive integer;

        max_basis_length is the maximum length of basis choices for Alice, took as read for number of bits sent through
        the quantum channel to Bob. Min is assumed to be 0. This variable will be normalised within all this class'
        methods. It has to be a positive integer;

        gain is the quantum channel gain, should be a numerical value between 0 and 1;

        disturbance_prob is the probability of disturbances in the quantum channel, i.e. the probability that a given
        bit will change into an opposite bit while being sent via the quantum channel;

        max_cascade_pass is the maximum number of CASCADE error correction algorithm's iterations - it has to be a
        positive integer;

        elite_size is the number of best chromosomes copied to the new population with no changes - it has to be a
        non-negative integer;

        target_key_length sets the down limit for lengths of final keys; if a key generated in a simulation is shorter,
        the fitness value of a given chromosome shall be 0.0;

        mutation_prob is the probability of a single chromosome to be randomly re-generated - it has to be numerical
        value between 0 and 1;

        fit_fun specifies which fitness function to use;

        args is a dictionary with arguments for the chosen fitness functions;

        If we wish so, we may specify seed of pseudo-random number generator by setting a seed value.
        """
        self.pop_size = pop_size
        self.max_basis_length = max_basis_length
        self.gain = channel_gain
        self.disturbance_prob = disturbance_prob
        self.max_cascade_pass = max_cascade_pass
        self.target_key_length = target_key_length
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.fitness_arg = args

        if seed is None:
            pass
        else:
            random.seed(a=seed)  # temporary, for debugging

        self.fit_fun = fit_fun  # we memorise the choice of fitness function
        self.generation = []  # list of all chromosomes (i.e. objects of the class Chromosome) in a given generation

        """What we need is to be able to sort whole generation based on fitness values AND remember chromosomes 
        indexes in their (generation) list in order to be able to crossbreed them with each other based on the
        fitness ranking. Thus, we create a list of dictionaries for this ranking. These dictionaries shall have
        just two keys: index (of a chromosome) and a fitness value (of the same chromosome). Once we compute such
        a fitness ranking for the whole generation, we shall sort it using sort_dict_by_fit function.
        """
        self.fitness_ranking = []

        """We build a list of all chromosomes in this initial population.
        Although with each population we will reuse the chromosomes (instances of a class Chromosome), 
        it's the only time when they are filled randomly - later on we shall have different methods
        for population creation.
        """
        for i in range(self.pop_size):  # creating initial generation
            """Since we aim to get at least a 256 bit-long key, there's no sense whatsoever 
            to exchange fewer bits than 256.
            """
            basis_length = random.randint(256, max_basis_length)  # without normalisation!
            basis_choice_prob = random.uniform(0, 1)  # probability of choosing rectilinear basis
            cascade_n_passes = random.randint(1, max_cascade_pass)
            """As we performe a refined error estimation, we need to generate publication probabilities for bits
            of either rectilinear or diagonal basis:
            """
            publication_prob_rect = random.uniform(0, 1)
            publication_prob_diag = random.uniform(0, 1)

            chromosome = Chromosome(
                length=basis_length,
                rect_basis_prob=basis_choice_prob,
                publication_prob_rectilinear=publication_prob_rect,
                publication_prob_diagonal=publication_prob_diag,
                no_pass=cascade_n_passes)  # we create chromosome with number of order - index i - in initial population
            self.generation.append(chromosome)  # we memorise this chromosome
        print('Initial generation created and ready for evaluation.')

    def generation_fitness(self):
        """Our goal is to calculate fitness values for the new generation. Because some chromosomes are simply copied
        as an elite, we don't want to waste time on calculating their fitness once more."""
        elite = self.fitness_ranking[:self.elite_size]
        self.fitness_ranking = []

        for i in tqdm.tqdm(range(len(self.generation)), desc='\nCalculating fitness for the new generation: '):
            fit = -100000.0  # this will be fit. val. if the fitness function is not properly set in __init__

            if i < self.elite_size:
                # print(elite[i])
                self.fitness_ranking.append({'index': i, 'fitness value': elite[i].get('fitness value')})
            else:
                chromosome = self.generation[i]

                match self.fit_fun:  # we calculate fit. val. for this chromosome with selected function
                    case 'factored_fit':
                        if chromosome.genes.get('pub_prob_rect') == 0 or chromosome.genes.get('pub_prob_diag') == 0:
                            fit = -1.0
                        else:
                            fit = factored_fit(disturbance_prob=self.disturbance_prob, quantum_gain=self.gain)
                self.fitness_ranking.append({'index': i, 'fitness value': fit})
        self.fitness_ranking.sort(key=sort_dict_by_fit, reverse=True)

    def best_fit(self):  # we return gene sequence of the chromosome of the highest fitness value with it's fit value
        bf = [self.generation[self.fitness_ranking[0].get('index')].genes,
              self.fitness_ranking[0].get('fitness value')]
        return bf

    def single_point_crossover(self, crossover_point, parent1, parent2):  # may be 'static'
        """Parents will be crossed such that genes from first one (numbered from 0) up to crossover_point
        included shall go to one child, and the rest to the other."""
        parent1_genes = list(parent1.genome.values())
        parent2_genes = list(parent2.genome.values())

        gene_counter = 0
        child1_genes = []
        child2_genes = []

        while gene_counter <= crossover_point:
            child1_genes.append(parent1_genes[gene_counter])
            child2_genes.append(parent2_genes[gene_counter])
            gene_counter += 1

        while gene_counter < len(parent1_genes):
            child1_genes.append(parent2_genes[gene_counter])
            child2_genes.append(parent1_genes[gene_counter])
            gene_counter += 1

        child1 = Chromosome(length=child1_genes[0],
                            rect_basis_prob=child1_genes[1],
                            publication_prob_rectilinear=child1_genes[2],
                            publication_prob_diagonal=child1_genes[3],
                            no_pass=child1_genes[4])

        child2 = Chromosome(length=child2_genes[0],
                            rect_basis_prob=child2_genes[1],
                            publication_prob_rectilinear=child2_genes[2],
                            publication_prob_diagonal=child2_genes[3],
                            no_pass=child2_genes[4])

        return [child1, child2]

    def uniform_crossover(self, parent1, parent2, choice_prob=0.5):  # may be 'static'
        """In this crossover method a gene mask is randomised. By default, there is 2 children. For the first one
        0 indicates genes from the first parent, while 1 - from the second one. For the second one contrarily.

        no_kids specifies how many children are to be breaded;

        choice_prob is the probability of choosing a gene from the first parent in a single Bernoulli trial.
        """
        parent1_genes = list(parent1.genome.values())
        parent2_genes = list(parent2.genome.values())

        child1_genes = []
        child2_genes = []

        gene_mask = np_binom(1, 1 - choice_prob, len(parent1_genes))
        index = 0

        for indicator in gene_mask:
            if indicator == 0:
                child1_genes.append(parent1_genes[index])
                child2_genes.append(parent2_genes[index])
            else:
                child1_genes.append(parent2_genes[index])
                child2_genes.append(parent1_genes[index])
            index += 1

        child1 = Chromosome(length=child1_genes[0],
                            rect_basis_prob=child1_genes[1],
                            publication_prob_rectilinear=child1_genes[2],
                            publication_prob_diagonal=child1_genes[3],
                            no_pass=child1_genes[4])

        child2 = Chromosome(length=child2_genes[0],
                            rect_basis_prob=child2_genes[1],
                            publication_prob_rectilinear=child2_genes[2],
                            publication_prob_diagonal=child2_genes[3],
                            no_pass=child2_genes[4])

        return [child1, child2]  # make function from method?

    def plco(self, parent1, parent2, alfa=0.5, beta=0.5):  # partially linear crossover operator (PLX?) & 'static'?
        """Two children are created; integer-valued genes are exchanged as in single crossover operator,
        while the real-valued genes are linearly combined using formula:

        child_gene = alfa * parent1_gene + beta * parent2_gene
        """
        parent1_genes = list(parent1.genome.values())
        parent2_genes = list(parent2.genome.values())

        child1 = Chromosome(
            length=parent1_genes[0],
            rect_basis_prob=(alfa * parent1_genes[1] + beta * parent2_genes[1]),
            publication_prob_rectilinear=(alfa * parent1_genes[2] + beta * parent2_genes[2]),
            publication_prob_diagonal=(alfa * parent1_genes[3] + beta * parent2_genes[3]),
            no_pass=parent2_genes[4]
        )

        child2 = Chromosome(
            length=parent2_genes[0],
            rect_basis_prob=(alfa * parent2_genes[1] + beta * parent1_genes[1]),
            publication_prob_rectilinear=(alfa * parent2_genes[2] + beta * parent1_genes[2]),
            publication_prob_diagonal=(alfa * parent2_genes[3] + beta * parent1_genes[3]),
            no_pass=parent1_genes[4]
        )

        return [child1, child2]

    def mutation(self):
        """If randomly generated number is less or equal to mutation_probability, each gene in a given chromosome
                is randomised again."""
        for chrom_no in range(len(self.generation)):
            if random.uniform(0, 1) <= self.mutation_prob:
                basis_length = random.randint(256, self.max_basis_length)
                basis_choice_prob = random.uniform(0, 1)
                pub_prob_rect = random.uniform(0, 1)
                pub_prob_diag = random.uniform(0, 1)
                cascade_n_passes = random.randint(1, self.max_cascade_pass)

                new_chromosome = Chromosome(
                    length=basis_length,
                    rect_basis_prob=basis_choice_prob,
                    publication_prob_rectilinear=pub_prob_rect,
                    publication_prob_diagonal=pub_prob_diag,
                    no_pass=cascade_n_passes
                )
                self.generation[chrom_no] = new_chromosome

    def ranking_selection(self, crossover_operator):  # creates new generation, overwrites the old one
        """We aim to create a new generation, for the moment using the single crossover operator by default.
        Just in case we start with reverse-sorting the fitness ranking, in which indexes of chromosomes from the
        old generation are stored.

        We use 'chromosome_counter' as an index from the fitness ranking per se. While it's smaller than the elite
        size for the whole population, we simply copy-paste the old chromosomes into the new generation.

        After that we cross pairs of chromosomes to obtain children for the new generation.
        """
        new_generation = []
        chromosome_counter = 0

        while chromosome_counter < self.elite_size:
            new_generation.append(self.generation[self.fitness_ranking[chromosome_counter].get('index')])
            new_generation.append(self.generation[self.fitness_ranking[chromosome_counter + 1].get('index')])
            chromosome_counter += 2

        while chromosome_counter < (self.elite_size + 2 * self.fitness_arg.get('parents_pair_no')):
            parent1 = self.generation[self.fitness_ranking[chromosome_counter].get('index')]
            parent2 = self.generation[self.fitness_ranking[chromosome_counter + 1].get('index')]

            children = crossover_operator(
                crossover_point=self.fitness_arg.get('cross_point'),
                parent1=parent1,
                parent2=parent2
            )
            new_generation.append(children[0])
            new_generation.append(children[1])

            chromosome_counter += 2
        self.generation = new_generation

    def roulette_wheel_selection(self, crossover_operator):
        fit_total = sum(self.fitness_ranking)  # jak wziąć same fitness value? 'find highest value for a given key'
        new_generation = []

        for i in range(self.pop_size):
            param = random.uniform(0, fit_total)
            fit_sum = 0
            index = 0

            while fit_sum < param and index < self.pop_size:
                fit_sum += self.fitness_ranking[index].get('fitness value')
                index += 1
            parent1 = self.generation[self.fitness_ranking[index].get('index')]

            param = random.uniform(0, fit_total)
            fit_sum = 0
            index = 0
            while fit_sum < param and index < self.pop_size:
                fit_sum += self.fitness_ranking[index].get('fitness value')
                index += 1
            parent2 = self.generation[self.fitness_ranking[index].get('index')]

            children = crossover_operator[parent1, parent2]
            new_generation.append(children[0])
            new_generation.append(children[1])
        self.generation = new_generation

    def stochastic_uniform_sampling(self, crossover_operator):
        fit_mean = np.mean(self.fitness_ranking)  # jak wziąć same fitness value?
        new_generation = []

        for i in range(self.pop_size):
            param = random.uniform(0, 1)
            fit_sum = self.fitness_ranking[0]
            delta = param * fit_mean
            index = 0

            while index < self.pop_size:
                if delta < fit_sum:
                    delta += fit_sum
                else:
                    fit_sum += self.fitness_ranking[index].get('fitness value')
                    index += 1

            parent1 = self.generation[self.fitness_ranking[index].get('index')]

            param = random.uniform(0, 1)
            fit_sum = self.fitness_ranking[0]
            delta = param * fit_mean
            index = 0

            while index < self.pop_size:
                if delta < fit_sum:
                    delta += fit_sum
                else:
                    fit_sum += self.fitness_ranking[index].get('fitness value')
                    index += 1

            parent2 = self.generation[self.fitness_ranking[index].get('index')]

            children = crossover_operator[parent1, parent2]
            new_generation.append(children[0])
            new_generation.append(children[1])
        self.generation = new_generation

    '''def linear_rank_selection(self, crossover_operator): -> to trzeba jeszcze zweryfikować, bo 2.001 w jednym
    mianowniku wygląda tak zaskakująco, że aż podejrzanie
    '''

    def exponential_rank_selection(self, crossover_operator):  # unfinished???
        new_generation = []

        n = len(self.fitness_ranking)
        c = 2 * n * (n - 1) / (6 * (n - 1) + n)

        for i in range(n):
            rank = n - 1 - i  # the highest rank goes for the best chromosome in the population
            alfa = random.uniform(c / 9, 2 / c)
            for j in range(n):  # is it possible to get fewer children then parents in the next generation?
                prob = 1.0 * np.exp(-1.0 * j / c)
                if prob <= alfa:
                    parent1 = self.generation[self.fitness_ranking[i].get('index')]
                    parent2 = self.generation[self.fitness_ranking[j].get('index')]

                    children = crossover_operator[parent1, parent2]
                    new_generation.append(children[0])
                    new_generation.append(children[1])

                    break
        self.generation = new_generation

    def tournament_selection(self, crossover_operator):
        """Useful for populations up to 300 individuals, given spipy's special.binom() - this one is the fastest:
        https://stackoverflow.com/questions/26560726/python-binomial-coefficient
        """
        tos_list = self.fitness_ranking
        random.shuffle(tos_list)
        n = len(tos_list)

        for i in range(n):
            for j in range(n):
                index1 = tos_list[i]
                for m in range(n):
                    index2 = tos_list[j + m]
        c1 = newton_binom(k - 1, n - 1)
        c2 = newton_binom(k, n)

    def euclidian_gene_dist(self, i, j):
        """Method calculating sum of euclidian distances in real number space of a given demension between i-th and j-th
        chromosome in the population, over i,j = 1,...,pop_size"""
        total_euclid_sum = 0

        for chrom1 in self.generation:
            for chrom2 in self.generation:
                total_euclid_sum += (chrom1.genes.get('length') - chrom2.genes.get('length')) ** 2 + (
                        chrom1.genes.get('rect_basis_prob') - chrom2.genes.get('rect_basis_prob')) ** 2 + (
                                            chrom1.genes.get('publication_prob_rectilinear') - chrom2.genes.get(
                                        'publication_prob_rectilinear')) ** 2 + (
                                            chrom1.genes.get('no_pass') - chrom2.genes.get('no_pass')) ** 2

        total_euclid_sum = np.sqrt(total_euclid_sum)

        return total_euclid_sum

    def diversity(self):
        div_list = []
        for index1 in range(self.pop_size - 1):
            for index2 in range(index1 + 1, self.pop_size, 1):
                chrom1_genes = [
                    self.generation[index1].genes.get('length'),
                    self.generation[index1].genes.get('rect_basis_prob'),
                    self.generation[index1].genes.get('publication_prob_rectilinear'),
                    self.generation[index1].genes.get('no_pass')
                ]
                chrom2_genes = [
                    self.generation[index2].genes.get('length'),
                    self.generation[index2].genes.get('rect_basis_prob'),
                    self.generation[index2].genes.get('publication_prob_rectilinear'),
                    self.generation[index2].genes.get('no_pass')
                ]

                euclidian_diff = 0
                for i in range(3):
                    euclidian_diff += (chrom1_genes[i] - chrom2_genes[i]) ** 2
                euclidian_diff = np.sqrt(euclidian_diff)
                div_list.append(euclidian_diff)
        return div_list

    '''Ja połączę tego Jibariego z parallel GA, na każdym wątku puszczę inny selection & crossover i porównam wyniki
    def jebari(self, selection_methods, crossover_operators, no_turns=1):
        """A selection method combining populations' genetic diversity and solutions' quality,
        proposed by Khalid Jebari of Methods for Genetic Algorithms' from 2013.

        selection_methods is a list of selection methods from the GeneticAlgorithmBB84 class to be compared;
        crossover_operators is a list of crossover operators from the GeneticAlgorithmBB84 class to be compared.
        """
        turn = 0
        while turn < no_turns:
    '''
