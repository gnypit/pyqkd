import random
import math
import time
import numpy as np

from scipy.stats import binom
from scipy.special import betainc

"""Global variables, characterising the quantum channel:"""
basis_mapping = {'rectilinear': 0, 'diagonal': 1}
states_mapping = {'0': 0, '1': 1, '+': 0, '-': 1}
quantum_channel = {
    'rectilinear': {
        'basis_vectors': {'first_state': '0', 'second_state': '1'}
    },
    'diagonal': {
        'basis_vectors': {'first_state': '+', 'second_state': '-'}
    }
}


def qc_gain(mean_photon_number=1., fiber_loss=1., detection_efficiency=1., k_dead=1.,
            additional_loss=1.):
    """Calculating the quantum channel gain as per (3.1) formula in "Applied Quantum Cryptography",
     section 3 by M. Pivk
     """
    g = mean_photon_number * fiber_loss * detection_efficiency * k_dead * additional_loss
    return g


def received_key_material(quantum_channel_gain, sender_data_rate):
    """Simple method for calculating receiver's data rate"""
    receiver = quantum_channel_gain * sender_data_rate
    return receiver


def random_choice(length, p=0.5):  # function for random choosing of basis for each photon
    """p -> probability of selecting rectilinear basis"""
    chosen_basis = ''
    for index in range(int(np.floor(length))):
        basis = random.uniform(0, 1)
        if basis <= p:
            chosen_basis += str(0)
        else:
            chosen_basis += str(1)

    return chosen_basis


def measurement(state, basis):  # TODO: update & optimise

    if basis == '1':  # meaning diagonal basis
        basis = 'diagonal'

        if state == '+':
            final_state = '+'
        elif state == '-':
            final_state = '-'
        elif state == '0' or state == '1':  # in this case there's a 50% chance of getting either polarization
            if random.randint(0, 1) == 0:
                final_state = quantum_channel.get(basis).get('basis_vectors').get('first_state')
            else:
                final_state = quantum_channel.get(basis).get('basis_vectors').get('second_state')
        else:
            return 'U'  # U for unknown

    elif basis == '0':  # meaning rectilinear basis

        if state == '0':
            final_state = '0'
        elif state == '1':
            final_state = '1'
        elif state == '+' or state == '-':
            final_state = str(random.randint(0, 1))  # since '0' and '1' are states, there's no need for if...else
        else:
            return 'U'  # U for unknown

    elif basis == 'L':
        final_state = 'L'  # L for loss, as basis L reflects unperformed measurement due to quantum channel loss
    else:
        return 'U'  # U for unknown

    return final_state


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
