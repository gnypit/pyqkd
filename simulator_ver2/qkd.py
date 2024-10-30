import random
import math
import time
from typing import Tuple, Any

import numpy as np

from scipy.stats import binom
from scipy.special import betainc

from sympy import symbols, I, Matrix, sqrt, simplify
from sympy.physics.quantum import TensorProduct, InnerProduct, OuterProduct
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator

"""Global variables, characterising the quantum channel:"""
basis_mapping = {'rectilinear': 0, 'diagonal': 1}
states_to_bit_mapping = {'|0>': 0, '|1>': 1, '|+>': 0, '|->': 1}
states_to_matrix_mapping = {
    '|0>': Matrix([[1], [0]]),
    '|1>': Matrix([[0], [1]]),
    '|+>': (Matrix([[1], [0]]) + Matrix([[0], [1]])) / sqrt(2),
    '|->': (Matrix([[1], [0]]) - Matrix([[0], [1]])) / sqrt(2)
}
quantum_channel = {
    '0': {  # for the rectilinear basis
        'first_state': '0',
        'second_state': '1'
    },
    '1': {  # for the diagonal basis
        'first_state': '+',
        'second_state': '-'
    }
}
"""Measurement operators defined as:
        M_0 = |0><0|,
        M_1 = |1><1|,
        M_+ = |+><+|,
        M_- = |-><-|
"""
measurement_operators = {
    'm0': TensorProduct(states_to_matrix_mapping.get('|0>'), Dagger(states_to_matrix_mapping.get('|0>'))),
    'm1': TensorProduct(states_to_matrix_mapping.get('|1>'), Dagger(states_to_matrix_mapping.get('|1>'))),
    'm+': TensorProduct(states_to_matrix_mapping.get('|+>'), Dagger(states_to_matrix_mapping.get('|+>'))),
    'm-': TensorProduct(states_to_matrix_mapping.get('|->'), Dagger(states_to_matrix_mapping.get('|->'))),
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


def simple_measurement_polarisation(state, basis):
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
            final_state = quantum_channel.get(basis).get('first_state')
            return final_state
        else:
            final_state = quantum_channel.get(basis).get('second_state')
            return final_state
    else:
        return final_state


def numerical_error_prob(n_errors, pass_size, qber):
    """Probability that n_errors remain in a given block, as per formula in 7.2 section in the '93 paper
     'Secret-Key Reconciliation by Public Discussion' by Brassard, Salvail.
     """
    prob = binom.pmf(n_errors, pass_size, qber) + binom.pmf(n_errors + 1, pass_size, qber)
    return prob


class Qubit:
    """This class is meant mainly for representation purposes, in simulations. It creates a qubit in either rectilinear
    or diagonal basis and allows to access both bra-ket and bit representations of the qubit."""

    def __init__(self, alfa, beta, basis):
        self.measurement_operator = None
        self.state_bit = None
        self.basis_bit = None
        self.alfa = float(alfa)
        self.beta = float(beta)
        self.basis = str(basis)

        self.first_base_vector = Ket(symbols(quantum_channel.get(self.basis).get('first_state')))
        self.second_base_vector = Ket(symbols(quantum_channel.get(self.basis).get('second_state')))

        self.superposition = self.alfa * self.first_base_vector + self.beta * self.second_base_vector

    def get_state(self):
        """For viewing bra-ket representation"""
        return self.superposition

    def _bit_representation(self):
        """Private method to convert bra-ket state into a pair of bits"""
        self.basis_bit = basis_mapping.get(self.basis)
        if self.alfa == 1.0:
            self.state_bit = states_to_bit_mapping.get(str(self.first_base_vector))
        elif self.beta == 1.0:
            self.state_bit = states_to_bit_mapping.get(str(self.second_base_vector))
        else:
            self.state_bit = None  # I don't have a precise idea how to handle non-pure states, yet

    def get_bits(self):
        """For viewing the pair-of-bits representation"""
        self._bit_representation()
        return self.basis_bit, self.state_bit

    def send(self, channel_gain=1.0):
        """This method simulates sending a qubit via the quantum channel. A pseudo-random number from a uniform
        distribution between 0 and 1 is selected. If it's smaller than the specified gain of the channel, than
        the qubit is properly received after sending; otherwise it is lost, resulting in a None state
        and 'L' (for 'loss') basis."""
        if random.uniform(0, 1) < channel_gain:
            self.get_bits()
        else:
            self.basis_bit = 'L'
            self.state_bit = None
            return self.basis_bit, self.state_bit

    def measure(self, measurement_operator):
        """There are four measurement matrices to choose from:
        'm0' = |0><0|,
        'm1' = |1><1|,
        'm+' = |+><+|,
        'm-' = |-><-|
        """
        self.measurement_operator = measurement_operators.get(measurement_operator)

        first_base_vector = states_to_matrix_mapping.get(str(self.first_base_vector))
        second_base_vector = states_to_matrix_mapping.get(str(self.second_base_vector))
        current_state = self.alfa * first_base_vector + self.beta * second_base_vector

        """The square root of probability has to be a 1x1 matrix, so we automatically get the numerical value from it
        to a float variable:"""
        sqrt_of_probability = sqrt(
            Dagger(current_state) * Dagger(self.measurement_operator) * self.measurement_operator * current_state
        )
        sqrt_of_probability = float(sqrt_of_probability.as_mutable()[0])

        new_state = self.measurement_operator * current_state / float(sqrt_of_probability)

        return new_state


class QMessage:
    """This class gets lists/strings of parameters and basis choices for multiple qubits to be created for a
            message to be sent via the quantum channel in a given protocol."""

    def __init__(self, alfa_list=None, beta_list=None, basis_list=None):
        """Variables should be either strings or tables of equal length. If all of them are received and of
        equal length, a table of Qubits based on specified coordinates in a given basis shall be created."""
        self.qubit_list = []
        self.status = None

        if len(alfa_list) == 0 or len(beta_list) == 0 or len(basis_list) == 0:
            self.status = 'Missing input'
            self.alfa_list = alfa_list
            self.beta_list = beta_list
            self.basis_list = basis_list
        else:
            self.status = 'Input received'
            self.alfa_list = alfa_list
            self.beta_list = beta_list
            self.basis_list = basis_list

            for index in range(len(self.basis_list)):
                self.qubit_list.append(
                    Qubit(
                        alfa=self.alfa_list[index],
                        beta=self.beta_list[index],
                        basis=self.basis_list[index]
                    ))


class Participant:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class QuantumChannel:
    """This is a class simulating """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class PublicChannel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class Eavesdropper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Protocol:
    """This class is a base for QKD protocols between any number of parties communicating and those eavesdropping.
    It has to consist of an error estimation function and error correction function/algorithm.
    It has a QBER field and a dictionary 'history' to store how raw keys change while the protocol is being performed.

    Once initiated, quantum channels, public channels, communicating parties (in short: participants) and eavesdroppers
    can be added to construct a whole network, over which the protocol is executed.

    Quantum channels have to be created between specific participants, while public channels can be available to any
    number of participants to use. Eavesdropper have to be assigned to specific quantum channels, only one for each
    eavesdropper.
    """

    def __init__(self, error_estimation_function, error_correction_algorithm):
        """These variables are valid for any protocol:"""
        self.participants = {}  # storage of communicating parties, e.g., Alice, Bob, etc.
        self.eavesdroppers = {}  # storage of eavesdroppers, e.g., Eve
        self.quantum_channels = {}  # storage of quantum channels
        self.public_channels = {}  # storage of public channels
        self.error_estimation_fun = error_estimation_function
        self.error_correction_algorithm = error_correction_algorithm
        self.qber = None  # quantum bit error rate, should be of type float

        """Additionally, for research/demonstration purposes, a history of the protocol can be recorded, e.g., the raw 
        key after each error correction algorithm's pass."""
        self.history = {}

    def __getstate__(self):
        state = {
            'error estimation function': self.error_estimation_fun,
            'error correction algorithm': self.error_correction_algorithm,
            'quantum bit error rate': self.qber,
            'history': self.history
        }
        return state

    def add_participants(self, **participants: dict[str, Participant]):
        """Method for adding any number of uniquely named participants."""
        for name, participant in participants.items():
            self.participants[name] = participant

    def add_quantum_channels(self, **quantum_channels: dict[str, QuantumChannel]):
        """Method for adding any number of quantum channels to be used by the protocol."""
        self.quantum_channels = quantum_channels

    def reconnect_participants(self):
        """Method for changing which participants are connected via which quantum channels; this info is stored within
        QuantumChannel class, in analogy to a graph."""


class BB84(Protocol):
    def __init__(self, *args, **kwargs):
        super().__init__(args)  # TODO: how to pass kwargs?
        self.qubits_sent = []
        self.qubits_received = []
        self.basis_sender = ''
        self.basis_receiver = ''
        self.bits_sender = ''
        self.bits_receiver = ''
        self.raw_key_sender = ''
        self.raw_key_receiver = ''

    def get_raw_key(self):
        """This method allows to get raw keys in between tasks and operations on them, to check the status of protocol,
        progress of error correction, etc."""
        raw_keys = {'sender': self.raw_key_sender, 'receiver': self.raw_key_receiver}
        return raw_keys

    def _qubit_exchange(self):
        self.qubit_exchange_args = self.kwargs.get('qubit exchange')

    def _sifting(self):
        self.sifting_args = self.kwargs.get('sifting')

    def _error_estimation(self):
        self.error_estimation_args = self.kwargs.get('error estimation')

    def _error_correction(self):
        self.error_correction_args = self.kwargs.get('error correction')

    def _privacy_amplification(self):
        self.privacy_amplification_args = self.kwargs.get('privacy amplification')

    def execute(self):
        self._qubit_exchange()
        self._sifting()
        self._error_estimation()
        self._error_correction()
        self._privacy_amplification()


class QKDProcedure:
    """A basic parent class for any stage in a QKD protocol."""
    def __init__(self, input, *args, **kwargs):
        self.input = input
        self.output = None
        self.args = args,
        self.kwargs = kwargs

    def get_results(self):
        return self.output

def main():
    """For testing/debugging"""
    nowy_qubit = Qubit(1, 0, 0)
    wynik = nowy_qubit.measure('m0')
    print(wynik)


if __name__ == "__main__":
    main()
