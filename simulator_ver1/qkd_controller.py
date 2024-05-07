"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""

import ast
import random
import math
import numpy as np
from PyQt6.QtWidgets import QLabel

from binary import binary
from scipy.stats import binom
import time
from demonstrator_gui import QKDWindow
from bb84 import ModelControllerBB84


"""Let's set up the quantum channel (BB84)"""
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
'''basis_vectors_mapping = {'0': '0', '90': '1', '+45': '+', '-45': '-'}
bits_mapping = {'0': 0, '90': 1, '+45': 0, '-45': 1}
basis_names = ['rectilinear', 'diagonal']
'''


def qc_gain(mean_photon_number=1., fiber_loss=1., detection_efficiency=1., k_dead=1.,
            additional_loss=1.):  # quantum channel gain -> do lamusa
    g = mean_photon_number * fiber_loss * detection_efficiency * k_dead * additional_loss
    return g


def received_key_material(quantum_channel_gain, sender_data_rate):
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


def measurement(state, basis):  # meant for classical encoding

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


def numerical_error_prob(n_errors, pass_size, qber):  # probability that 2*n_errors remain
    prob = binom.pmf(2 * n_errors, pass_size, qber) + binom.pmf(2 * n_errors + 1, pass_size, qber)
    return prob


def cascade_blocks_sizes(qber, key_length, n_passes=1):
    max_expected_value = -1 * math.log(0.5, math.e)
    # best_expected_value = max_expected_value
    best_size = key_length

    for size in range(key_length // 2):  # we need at lest 2 blocks to begin with

        # Firstly we check condition for expected values
        expected_value = 0

        for j in range(size // 2):
            expected_value += 2 * (j + 1) * numerical_error_prob(n_errors=(j + 1), pass_size=size, qber=qber)

        if expected_value <= max_expected_value:
            first_condition = True
        else:
            first_condition = False

        # Secondly we check condition for probabilities per se
        second_condition = False
        for j in range(size // 2):
            prob_sum = 0
            for k in list(np.arange(j + 1, size // 2 + 1, 1)):
                prob_sum += numerical_error_prob(n_errors=k, pass_size=size, qber=qber)

            if prob_sum <= numerical_error_prob(n_errors=j, pass_size=size, qber=qber) / 4:
                second_condition = True
            else:
                second_condition = False

        if first_condition and second_condition:
            if size > best_size:
                # best_expected_value = expected_value
                best_size = size

    sizes = [best_size]

    for j in range(n_passes - 1):  # corrected interpretation of number of passes
        next_size = 2 * sizes[-1]
        if next_size <= key_length:
            sizes.append(next_size)
        else:
            break

    return sizes


def cascade_blocks_generator(string_length, blocks_size):
    string_index = list(np.arange(0, string_length, 1))  # I create a list of all indexes (list of ndarray)
    blocks = random.sample(population=string_index, k=string_length)  # I shuffle the list randomly

    for j in range(0, string_length, blocks_size):  # I generate equally long chunks of shuffled indexes
        yield blocks[j:j + blocks_size]


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
    estimation takes that into consideration.
    """
    for index in range(len(alice_bits)):
        if alice_basis[index] == bob_basis[index] == '0':  # rectilinear basis
            if random.uniform(0, 1) >= rect_pub_prob:
                alice_key += alice_bits[index]
                bob_key += bob_bits[index]
            else:
                rect_pub_counter += 1
                if alice_bits[index] != bob_bits[index]:
                    rect_error += 1
        else:
            if random.uniform(0, 1) >= diag_pub_prob:
                alice_key += alice_bits[index]
                bob_key += bob_bits[index]
            else:
                diag_pub_counter += 1
                if alice_bits[index] != bob_bits[index]:
                    diag_error += 1

    """As it is possible to get a VERY small probability of publication, we check for possible divisions by zero:"""
    try:
        rect_error = float(rect_error) / float(rect_pub_counter)
    except ZeroDivisionError:
        rect_error = 0.0  # this will obviously be false, but easy to notice and work with in genalqkd.py

    try:
        diag_error = float(diag_error) / float(diag_pub_counter)
    except ZeroDivisionError:
        diag_error = 0.0  # this will obviously be false, but easy to notice and work with in genalqkd.py

    """Now, given that measurements in the rectilinear basis were not necessarily with the same probability 
    as those in the diagonal basis, we need a more complicated formula for the 'average error estimate' 
    (Lo, Chau, Ardehali, 2004).
    """
    p = rect_prob  # just a reminder that it's the probability of choosing rect. basis for measurements
    e1 = rect_error
    e2 = diag_error

    e = (p ** 2 * e1 + (1 - p) ** 2 * e2) / (p ** 2 + (1 - p) ** 2)

    results = {
        'error estimator': e,
        'alice key': alice_key,
        'bob key': bob_key,
        'number of published bits': rect_pub_counter + diag_pub_counter
    }

    return results


class QKDController:
    def __init__(self, model: ModelControllerBB84, view: QKDWindow):
        """Method for initialisation of a single BB84 protocol's logic for demonstration. In internal variables are
        created, but no operations are performed yet, as the User didn't have a chance to act."""
        self.size = None  # number of quantum states to be prepared by Alice
        self._model = model
        self._view = view

        """Finally, the signals and slots are connected:"""
        self._connect_signals_and_slots()

    def _connect_signals_and_slots(self):
        self._view.simulate_button.clicked.connect(self._simulate)
        self._model.status_update_signal.connect(self._handle_status_update)

    def _simulate(self):
        """This method is responsible for passing input to the simulation of the QKD protocol (right now only BB84)
        and receiving the results."""
        self._view.quantum_prompt_label.setText(f"Preparing the simulation - please wait.")

        # TODO: generalisation of passing arguments to different simulations
        try:
            simulation_results = self._model.simulation_bb84(
                self._view.simulation_hiperarameters.get('gain'),
                self._view.simulation_hiperarameters.get('no_quantum_states'),
                self._view.simulation_hiperarameters.get('rect_prob'),
                self._view.simulation_hiperarameters.get('dist_prob'),
                self._view.simulation_hiperarameters.get('rect_pub_prob'),
                self._view.simulation_hiperarameters.get('diag_pub_prob'),
                self._view.simulation_hiperarameters.get('no_cascade')
            )

            self._view.quantum_prompt_label = QLabel(
                f"Simulation prepared successfully. To walk through it step by step,"
                f"please click 'Play'. To pause at any point, click 'Pause' - you can "
                f"resume by clicking 'Play' again. Click 'Fast Forward' to speed the "
                f"simulation up.")
        except AttributeError:
            self._view.quantum_prompt_label.setText(f"There has been an issue while preparing the simulation "
                                                    f"(AttributeError). Please try changing the input data or contact "
                                                    f"the Author.")
        except ValueError:
            self._view.quantum_prompt_label.setText(f"There has been an issue while preparing the simulation "
                                                    f"(ValueError). Please try changing the input data or contact "
                                                    f"the Author.")
        except TypeError:
            self._view.quantum_prompt_label.setText(f"There has been an issue while preparing the simulation "
                                                    f"(TypeError). Please try changing the input data or contact "
                                                    f"the Author.")

    def _handle_status_update(self):
        # TODO: signal handling from ModelBB84
        self._view.quantum_prompt_label.setText("Test")

    def _play_quantum(self):
        # TODO: filling in the table on the quantum channel tab in QKDWindow
        pass

    def _pause_quantum(self):
        # TODO: stop filling in the table on the quantum channel tab in QKDWindow & memorise where to pick up later
        pass

    def _fast_forward_quantum(self):
        # TODO: same as play, but with no time delays
        pass

    def _sifting(self):
        """If necessary and not included as part of '_play_quantum', method for crossing out verses in the quantum
        table which represent data for different basis choices by Alice & Bob."""
        pass
