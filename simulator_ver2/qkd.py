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
    '0': {  # for the rectilinear basis
        'basis_vectors': {'first_state': '0', 'second_state': '1'}
    },
    '1': {  # for the diagonal basis
        'basis_vectors': {'first_state': '+', 'second_state': '-'}
    }
}


def qc_gain(mean_photon_number=1., fiber_loss=1., detection_efficiency=1., k_dead=1., additional_loss=1.):
    """Calculating the quantum channel gain as per (3.1) formula in "Applied Quantum Cryptography",
     section 3 by M. Pivk
     """
    g = mean_photon_number * fiber_loss * detection_efficiency * k_dead * additional_loss
    return g


def received_key_material(quantum_channel_gain, sender_data_rate):
    """Simple method for calculating receiver's data rate"""
    receiver = quantum_channel_gain * sender_data_rate
    return receiver


def random_choice(length, p=0.5):
    """Function for randomising basis for a given number of photons
    p -> probability of selecting rectilinear basis
    """
    chosen_basis = ''
    for index in range(int(np.floor(length))):
        basis = random.uniform(0, 1)
        if basis <= p:
            chosen_basis += str(0)
        else:
            chosen_basis += str(1)

    return chosen_basis


def measurement(state, basis):
    """This function simulates a simple measurement of photon's state, encoded in polarisation. It receives the
    original state of photon and the basis, in which this photon is being measured. For details of mathematics behind
    this operation please refer to 'Applied Quantum Cryptography', sections 2 & 3, authored by M. Pivk

    First, the worst-case scenario is handled, that is when a photon is not received, or an error is encountered:
    """
    if basis == 'L':
        final_state = 'L'  # L for loss, as basis L reflects unperformed measurement due to quantum channel loss
        return final_state

    """Now that the loss in quantum channel is handled, the actual measurement can be simulated:"""
    possible_scenarios = {
        '1': {  # diagonal basis
            '+': '+',  # states from the diagonal basis measured in it remain the same
            '-': '-',  # states from the diagonal basis measured in it remain the same
            '0': 'random',  # states from the rectilinear basis measured in the diagonal one yield random results
            '1': 'random'  # states from the rectilinear basis measured in the diagonal one yield random results
        },
        '0': {  # rectilinear basis
            '+': 'random',  # states from the diagonal basis measured in the rectilinear one yield random results
            '-': 'random',  # states from the diagonal basis measured in the rectilinear one yield random results
            '0': '0',  # states from the rectilinear basis measured in it remain the same
            '1': '0'  # states from the rectilinear basis measured in it remain the same
        }
    }

    final_state = possible_scenarios.get(basis).get(state)
    if final_state == 'random':
        """In this case there's a 50% chance of getting either polarization"""
        if random.randint(0, 1) == 0:
            final_state = quantum_channel.get(basis).get('basis_vectors').get('first_state')
            return final_state
        else:
            final_state = quantum_channel.get(basis).get('basis_vectors').get('second_state')
            return final_state
    else:
        return final_state


def numerical_error_prob(n_errors, pass_size, qber):
    """Probability that n_errors remain in a given block, as per formula in 7.2 section in the '93 paper
     'Secret-Key Reconciliation by Public Discussion' by Brassard, Salvail.
     """
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
    a significant speed-up.
    """
    prob = betainc(n_errors + 2, first_pass_size - n_errors - 1, qber)

    return prob
