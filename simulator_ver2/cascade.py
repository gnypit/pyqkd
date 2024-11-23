import math
import random

import numpy as np
from scipy.stats import binom
from scipy.special import betainc


def numerical_error_prob(n_errors, pass_size, qber):  # probability that n_errors remain
    prob = binom.pmf(n_errors, pass_size, qber) + binom.pmf(n_errors + 1, pass_size, qber)
    return prob


def sum_error_prob_betainc(first_pass_size, qber, n_errors):
    """Simplified left side of (2) inequality for CASCADE blocks. It uses regularised incomplete beta function as a
    representation of binomial distribution CDF; after proposed [paper is currently being written] representation of (2)
    inequality this way the computations are significantly faster, at least by one order of magnitude for small limits
    of series and a few orders of magnitude (e.g. 5) for greater (thousands).

    Although by 1993 paper "Secret Key Reconciliation by Public Discussion" by Gilles Brassard and Louis Salvail,
    published in "Advances in Cryptography" proceedings, for 10 000 qubits the initial block size is only 73, this
    program aims to be a universal tool for simulations, allowing arbitrarily large amounts of qubits to be sent
    and post-processed. Moreover, with tens of thousands simulations performed, even one order of magnitude offers
    a significant speed-up."""
    prob = betainc(n_errors + 2, first_pass_size - n_errors - 1, qber)

    return prob


def cascade_blocks_sizes_old(quantum_bit_error_rate, key_length, n_passes=1):
    """An iterative procedure to find the largest initial block size for the CASCADE algorithm,
    fulfilling conditions (2) and (3) as described in 1993 paper "Secret Key Reconciliation by Public Discussion"
    by Gilles Brassard and Louis Salvail, published in "Advances in Cryptography" proceedings.

    This function searches in nested loops all possible combinations of numbers of errors and block sizes to identify
    the largest one suitable for the whole algorithm to be performed.
    """
    max_expected_value = -1 * math.log(0.5, math.e)
    best_size = 0

    for size in list(np.arange(2, key_length // 4 + 1, 1)):  # we need at lest 2 blocks to begin with

        """Firstly we check condition for expected values - (3) in the paper"""
        expected_value = 0

        for j in list(np.arange(1, size // 2 + 1, 1)):
            expected_value += 2 * j * numerical_error_prob(n_errors=2 * j, pass_size=size, qber=quantum_bit_error_rate)

        if expected_value > max_expected_value:
            """As for increasing sizes the expected value is non-decreasing. Thus, once the expected_value of number
            of errors remaining in the first block is greater than the max_expected_value, it'll always be.
            Therefore, it makes no sense to keep checking greater sizes - none of them will meet this requirement."""
            break

        """Secondly we check condition for probabilities per se - (2) in the paper"""
        second_condition = False
        for j in list(np.arange(0, size // 2 + 1, 1)):
            prob_sum = 0
            for l in list(np.arange(j + 1, size // 2 + 1, 1)):
                prob_sum += numerical_error_prob(n_errors=2 * l, pass_size=size, qber=quantum_bit_error_rate)

            if prob_sum <= numerical_error_prob(n_errors=2 * j, pass_size=size, qber=quantum_bit_error_rate) / 4:
                second_condition = True
            else:
                """This condition has to be met by all possible numbers of errors. If a single number of errors doesn't
                meet this requirement, there is no point in keeping checking the other for this particular size."""
                second_condition = False
                break

        if second_condition:
            if size > best_size:
                # best_expected_value = expected_value
                best_size = size

    sizes = [best_size]

    for _ in range(n_passes - 1):  # corrected interpretation of number of passes
        next_size = 2 * sizes[-1]
        if next_size <= key_length:
            sizes.append(next_size)
        else:
            break

    return sizes


def cascade_blocks_sizes(quantum_bit_error_rate, key_length, n_passes=2):
    """An iterative procedure to find the largest initial block size for the CASCADE algorithm,
    fulfilling conditions (2) and (3) as described in 1993 paper "Secret Key Reconciliation by Public Discussion"
    by Gilles Brassard and Louis Salvail, published in "Advances in Cryptography" proceedings.

    In this improved version of cascade_blocks_sizes functon the checks for the (2) & (3) of conditions from the '93
    CASCADE paper are simplified, resulting in lesser computational complexity. For additional context, these
    conditions are a system of non-linear inequalities that need to be fulfilled in order to have the probability
    of correcting at least 2 errors in a given block in any pass greater than 0.75

    In this approach we implement regularised incomplete beta function to represent binomial distribution CDF
    for a simplified left side of the (2) inequality from tha paper.

    Additionally, we use a single formula for the expected value (3) of number of errors in a given block after
    completion of the first CASCADE pass.
    """
    max_expected_value = -1 * math.log(0.5, math.e)
    best_size = 0  # all sizes fulfilling (2) & (3) ineq. will be greater than that; we're looking fo the greatest

    for size in list(np.arange(2, key_length // 4 + 1, 1)):
        """We need at lest 4 blocks to begin with - then we can perform 2 passes.

        Firstly we check condition for the expected value of number of errors remaining in a block
        in the first pass of CASCADE - (3) in the paper
        """
        expected_value = size * quantum_bit_error_rate - (1 - (1 - 2 * quantum_bit_error_rate) ** size) / 2
        if expected_value > max_expected_value:
            """As for increasing sizes the expected value is non-decreasing. Thus, once the expected_value of number
            of errors remaining in the first block is greater than the max_expected_value, it'll always be.
            Therefore, it makes no sense to keep checking greater sizes - none of them will meet this condition."""
            break

        """For the (2) condition (inequality)..."""
        second_condition = True
        for j in list(np.arange(0, size // 2, 1)):
            """For number of errors equal to the amount of bits (or one bit fewer) in a block.

            When you analyse the inequality (2) carefully, you'll notice, that for j = size of the first pass // 2 - 1
            the sum on the left side contains the probability of having 2 * (j + 1) errors, which is equal to the length
            of the first block. This means that for any j greater than the size of the first pass // 2 - 1 there are no
            expressions in the left-side sum "left", rendering it 0, which is always equal to or lesser than 
            a non-negative value of any probability..
            """
            right_side = numerical_error_prob(
                n_errors=2 * j,
                pass_size=size,
                qber=quantum_bit_error_rate) / 4
            left_side = sum_error_prob_betainc(  # regularised incomplete beta function is used here
                first_pass_size=size,
                n_errors=2 * j,
                qber=quantum_bit_error_rate
            )

            """Now we check inequality (2) - must work for all possible numbers of errors in a block of given size."""
            if left_side > right_side:
                second_condition = False
                break

        if second_condition:
            if size > best_size:
                best_size = size

    sizes = [best_size - best_size % 2]  # all sizes must be even numbers for BINARY (search for errors) to operate

    for j in range(n_passes - 1):  # corrected interpretation of number of passes
        next_size = 2 * sizes[-1]
        if next_size <= key_length:
            sizes.append(next_size)
        else:
            break

    return sizes


def cascade_blocks_generator(key_length, blocks_size):
    string_index = list(np.arange(0, key_length, 1))  # I create a list of all indexes (list of ndarray)
    blocks = random.sample(population=string_index, k=key_length)  # I shuffle the list randomly

    for j in range(0, key_length, blocks_size):  # I generate equally long chunks of shuffled indexes
        yield blocks[j:j + blocks_size]


def count_key_value_differences(dict1, dict2):
    """
    Computes the number of differing values under the same keys between two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        int: The count of keys with differing values between the two dictionaries.
    """
    differing_count = 0

    # Loop over the keys in the first dictionary
    for key in dict1:
        # Check if the key exists in the second dictionary and the values differ
        if key in dict2 and dict1[key] != dict2[key]:
            differing_count += 1

    return differing_count

