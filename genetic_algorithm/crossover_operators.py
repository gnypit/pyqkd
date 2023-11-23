"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/

This file contains pre-defined crossover operators for the Generation class from genetic_algorithm.py to use within
the genetic algorithm.
"""

from numpy.random import binomial as np_binom
from genetic_algorithm import Member


def single_point_crossover(crossover_point, parent1: Member, parent2: Member):
    """Parents will be crossed such that genes from first one (numbered from 0) up to crossover_point
    included shall go to one child, and the rest to the other."""

    parent1_genes = list(parent1.genes)
    parent2_genes = list(parent2.genes)

    # TODO: different working with genes whether it's a dict or a list

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

    return [child1_genes, child2_genes]


def uniform_crossover(parent1: Member, parent2: Member, choice_prob=0.5):
    """In this crossover method a gene mask is randomised. By default, there is 2 children. For the first one
    0 indicates genes from the first parent, while 1 - from the second one. For the second one contrarily.

    no_kids specifies how many children are to be breaded;

    choice_prob is the probability of choosing a gene from the first parent in a single Bernoulli trial.
    """
    parent1_genes = list(parent1.genes)
    parent2_genes = list(parent2.genes)

    # TODO: different working with genes whether it's a dict or a list

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

    return [child1_genes, child2_genes]


def plco(parent1: Member, parent2: Member, transit_point, alfa=0.5, beta=0.5):  # partially linear crossover operator
    """Two children are created; integer-valued genes are exchanged as in single crossover operator,
    while the real-valued genes are linearly combined using formula:

    child_gene = alfa * parent1_gene + beta * parent2_gene

    transit point is the index from which crossover is linear; beforehand it's single point
    """
    parent1_genes = list(parent1.genes)
    parent2_genes = list(parent2.genes)
    no_genes = len(parent1_genes)

    # TODO: different working with genes whether it's a dict or a list

    child1_genes = []
    child2_genes = []

    for index in range(transit_point // 2 + 1):  # single-point crossover part 1
        child1_genes.append(parent1_genes[index])
        child2_genes.append(parent2_genes[index])
    for index in range(transit_point // 2 + 1, transit_point):  # single-point crossover part 2
        child1_genes.append(parent2_genes[index])
        child2_genes.append(parent1_genes[index])
    for index in range(transit_point, no_genes):  # linear crossover
        child1_genes.append(alfa * parent1_genes[index] + beta * parent2_genes[index])
        child2_genes.append(alfa * parent2_genes[index] + beta * parent1_genes[index])

    return [child1_genes, child2_genes]