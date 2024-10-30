import random

import numpy as np


def naive_error(alice_key, bob_key, publication_prob_rect):
    """
    Originally I've been testing algorithm on short strings, i.e. up to 20 bits in original Alice's message.
    Sifted keys were therefore too short for publication of any parts of them for error estimation.

    A refined estimation is based on two subsets: one of the sifted key and the other on the bits originating from
    measurements with different basis (Alice's and Bob's). First we present the 'naive' estimation.

    Let's randomly publish subsets of bits of matching positions in the sifted keys strings. Then we count how many
    bits differ between these two strings, divide this by their length and get a naive estimator of the total error
    rate (between error correction algorithm, i.e. CASCADE).

    As a result of this function we return the naive estimator and Alice's & Bob's keys without published bits.
    """
    alice_published_bits = ''
    alice_sifted_key_after_error_estimation = ''
    bob_published_bits = ''
    bob_sifted_key_after_error_estimation = ''

    naive_error_estimate = 0

    for index in range(len(alice_key)):  # could be bob_key as well
        if random.uniform(0, 1) <= publication_prob_rect:
            alice_bit = alice_key[index]
            bob_bit = bob_key[index]

            """First we add those bits to strings meant for publication"""
            alice_published_bits += alice_bit
            bob_published_bits += bob_bit

            """Now for the estimation of the error:"""
            if alice_bit != bob_bit:
                naive_error_estimate += 1
        else:  # if a bit wasn't published, we reuse it in the sifted key
            alice_sifted_key_after_error_estimation += alice_key[index]
            bob_sifted_key_after_error_estimation += bob_key[index]

    try:
        naive_error_estimate = naive_error_estimate / len(alice_published_bits)
    except ZeroDivisionError:
        naive_error_estimate = 0  # this will obviously be false, but easy to notice and work on in genalqkd.py

    """At this point we count the number of bits exchanged between Alice and Bob via the public channel,
    for future estimation of the computational cost."""

    no_published_bits = len(alice_published_bits) + len(bob_published_bits)

    results = {
        'error estimator': naive_error_estimate,
        'alice key': alice_sifted_key_after_error_estimation,
        'bob key': bob_sifted_key_after_error_estimation,
        'number of published bits': no_published_bits
    }

    return results


def refined_average_error(rect_prob, rect_pub_prob, diag_pub_prob,
                          alice_bits, bob_bits, alice_basis, bob_basis):
    """In the refined error analysis for simplicity we DO NOT divide raw keys into two separate strings (by the basis).
    Instead, we create two empty strings - alice_key & bob_key - into which we shall rewrite bits unused for error
    estimation. As for the others, chosen with probability rect_pub_prob & diag_pub_prob, respectively, we count them
    as 'published' and additionally count an error, if they differ from each other. Those counter are:
    rect_pub_counter & diag_pub_counter, rect_error & diag_error. The last two will be divided at the end by the
    first two, respectively, to obtain estimations as ratios.
    """
    alice_key = ''
    bob_key = ''
    rect_error = 0
    rect_pub_counter = 0
    diag_error = 0
    diag_pub_counter = 0

    """By the way, we don't have to worry with the conditional probability, because the final formula for the error
    estimation takes that into consideration. However, due to randomising the bits which are up for publication,
    we should verify that the right amount is published. It is not a problem while sending qubits because of their sheer
    number, but for publishing the subsets might be so small, that the numbers of bits published might spread more.
    """
    number_of_all_bits = len(alice_bits)
    number_of_rect_bits_to_be_published = np.floor(rect_pub_prob * number_of_all_bits)
    number_of_diag_bits_to_be_published = np.floor(diag_pub_prob * number_of_all_bits)
    indexes_of_published_bits = []
    index = 0

    while rect_pub_counter < number_of_rect_bits_to_be_published or diag_pub_counter < number_of_diag_bits_to_be_published:
        """We iterate over the strings with bits, either publishing bits in their respective basis, or not. Bits 
        published have their indexes remembered, so that we don't accidentally publish them more than once, when 
        iterating over the initial strings."""
        if alice_basis[index] == bob_basis[index] == '0':  # rectilinear basis
            if random.uniform(0, 1) >= rect_pub_prob:
                alice_key += alice_bits[index]
                bob_key += bob_bits[index]
            else:
                rect_pub_counter += 1
                indexes_of_published_bits.append(index)
                if alice_bits[index] != bob_bits[index]:
                    rect_error += 1
        else:
            if random.uniform(0, 1) >= diag_pub_prob:
                alice_key += alice_bits[index]
                bob_key += bob_bits[index]
            else:
                diag_pub_counter += 1
                indexes_of_published_bits.append(index)
                if alice_bits[index] != bob_bits[index]:
                    diag_error += 1
        if index < number_of_all_bits:
            index += 1
        else:
            """If we have already run over all bits and didn't publish the expected amount, we start going through
                        the strings once again"""
            index = 0

    """As it is possible to get a VERY small probability of publication, we check for possible divisions by zero:"""
    try:
        rect_error = float(rect_error) / float(rect_pub_counter)
    except ZeroDivisionError:
        rect_error = 0.0

    try:
        diag_error = float(diag_error) / float(diag_pub_counter)
    except ZeroDivisionError:
        diag_error = 0.0

    """Now, given that measurements in the rectilinear basis were not necessarily with the same probability 
    as those in the diagonal basis, we need a more complicated formula for the 'average error estimate' 
    (Lo, Chau, Ardehali, 2004).
    """
    p = rect_prob  # just a reminder that it's the probability of choosing rect. basis for measurements
    e1 = rect_error
    e2 = diag_error

    e = (p**2 * e1 + (1 - p)**2 * e2) / (p**2 + (1 - p)**2)

    results = {
        'error estimator': e,
        'alice key': alice_key,
        'bob key': bob_key,
        'number of published bits': rect_pub_counter + diag_pub_counter
    }

    return results