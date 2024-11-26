import random

import numpy as np


def naive_error(alice_key, bob_key, publication_probability):
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
    alice_published_bits = []
    alice_key_after_error_estimation = []
    bob_published_bits = []
    bob_key_after_error_estimation = []

    naive_error_estimate = 0

    for index in range(len(alice_key)):  # could be bob_key as well
        if random.uniform(0, 1) <= publication_probability:
            alice_bit = alice_key[index]
            bob_bit = bob_key[index]

            """First we add those bits to strings meant for publication"""
            alice_published_bits.append(alice_bit)
            bob_published_bits.append(bob_bit)

            """Now for the estimation of the error:"""
            if alice_bit != bob_bit:
                naive_error_estimate += 1
        else:  # if a bit wasn't published, we reuse it in the sifted key
            alice_key_after_error_estimation.append(alice_key[index])
            bob_key_after_error_estimation.append(bob_key[index])

    try:
        naive_error_estimate = naive_error_estimate / len(alice_published_bits)
    except ZeroDivisionError:
        naive_error_estimate = 0  # this will obviously be false, but easy to notice and work on in genalqkd.py

    """At this point we count the number of bits exchanged between Alice and Bob via the public channel,
    for future estimation of the computational cost."""

    no_published_bits = len(alice_published_bits) + len(bob_published_bits)

    results = {
        'error estimator': naive_error_estimate,
        'alice key': alice_key_after_error_estimation,
        'bob key': bob_key_after_error_estimation,
        'number of published bits': no_published_bits
    }

    return results


def refined_average_error(rect_prob, rect_pub_prob, diag_pub_prob,
                          alice_bits, bob_bits, basis):
    """In the refined error analysis for simplicity we DO NOT divide raw keys into two separate strings (by the basis).
    Instead, we create two empty strings - alice_key & bob_key - into which we shall rewrite bits unused for error
    estimation. As for the others, chosen with probability rect_pub_prob & diag_pub_prob, respectively, we count them
    as 'published' and additionally count an error, if they differ from each other. Those counter are:
    rect_pub_counter & diag_pub_counter, rect_error & diag_error. The last two will be divided at the end by the
    first two, respectively, to obtain estimations as ratios.
    """
    alice_key = []
    bob_key = []
    length = len(basis)

    """Vectorize generation of random publication decision"""
    random_vals = np.random.uniform(0, 1, length)

    """Separate indices by basis"""
    rect_indices = [index for index in range(length) if basis[index] == 0]
    diag_indices = [index for index in range(length) if basis[index] == 1]

    """Determine which rectilinear and diagonal bits are published"""
    alice_rect_published = [alice_bits[index] for index in rect_indices if random_vals[index] < rect_pub_prob]
    alice_diag_published = [alice_bits[index] for index in diag_indices if random_vals[index] < diag_pub_prob]
    bob_rect_published = [bob_bits[index] for index in rect_indices if random_vals[index] < rect_pub_prob]
    bob_diag_published = [bob_bits[index] for index in diag_indices if random_vals[index] < diag_pub_prob]
    # rect_published = rect_indices[random_vals[rect_indices] < rect_pub_prob]
    # diag_published = diag_indices[random_vals[diag_indices] < diag_pub_prob]

    """Count errors and published bits for rectilinear and diagonal bases"""
    # rect_error = np.sum(alice_bits[rect_published] != bob_bits[rect_published])
    # diag_error = np.sum(alice_bits[diag_published] != bob_bits[diag_published])
    rect_pub_len = len(alice_rect_published)
    diag_pub_len = len(alice_diag_published)

    """Error calculations, with safe handling for division by zero"""
    rect_error = np.sum(alice_rect_published != bob_rect_published) / rect_pub_len if rect_pub_len > 0 else 0.0
    diag_error = np.sum(alice_diag_published != bob_diag_published) / diag_pub_len if diag_pub_len > 0 else 0.0
    # rect_error = rect_error / rect_pub_counter if rect_pub_counter > 0 else 0.0
    # diag_error = diag_error / diag_pub_counter if diag_pub_counter > 0 else 0.0

    """Now, given that measurements in the rectilinear basis were not necessarily with the same probability 
    as those in the diagonal basis, we need a more complicated formula for the 'average error estimate' 
    (Lo, Chau, Ardehali, 2004).
    """
    p = rect_prob
    e1 = rect_error
    e2 = diag_error
    e = (p ** 2 * e1 + (1 - p) ** 2 * e2) / (p ** 2 + (1 - p) ** 2) if (p ** 2 + (1 - p) ** 2) > 0 else 0.0

    """Collect non-published bits for keys"""
    unused_rect_indices = rect_indices[random_vals[rect_indices] >= rect_pub_prob]  # TODO: update the code to properly handle bits that weren't published, in their right order!
    unused_diag_indices = diag_indices[random_vals[diag_indices] >= diag_pub_prob]

    """To chyba będzie wygodniej zrobić już na słowniku, porównując oryginalny indeks bitu u Alicji/Boba
    z indeksem danej bazy, a przez to prawdopodobieństwem publikacji."""

    alice_key.extend(alice_bits[unused_rect_indices])
    bob_key.extend(bob_bits[unused_rect_indices])
    alice_key.extend(alice_bits[unused_diag_indices])
    bob_key.extend(bob_bits[unused_diag_indices])

    results = {
        'error estimator': e,
        'alice key': alice_key,
        'bob key': bob_key,
        'number of published bits': rect_pub_len + diag_pub_len
    }

    return results
