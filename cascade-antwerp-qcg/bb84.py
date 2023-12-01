"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
"""

import ast
import random
import math
import time
import numpy as np

from binary import binary
from scipy.stats import binom  # it cannot find binom in scipy.stats ?!


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


def numerical_error_prob(n_errors, pass_size, qber):  # probability that 2*n_errors remain
    prob = binom.pmf(2 * n_errors, pass_size, qber) + binom.pmf(2 * n_errors + 1, pass_size, qber)
    return prob


def cascade_blocks_sizes_old(quantum_bit_error_rate, key_length, n_passes=1):
    """An iterative procedure to find the largest initial block size for the CASCADE algorithm,
    fulfilling conditions (2) and (3) as described in 1993 paper "Secret Key Reconciliation by Public Discussion"
    by Gilles Brassard and Louis Salvail, published in "Advances in Cryptography" proceedings.

    This function searches in nested loops all possible combinations of numbers of errors and block sizes to identify
    the largest one suitable for the whole algorithm to be performed.
    """
    max_expected_value = -1 * math.log(0.5, math.e)
    # best_expected_value = max_expected_value
    best_size = key_length

    for size in range(key_length // 2):  # we need at lest 2 blocks to begin with

        # Firstly we check condition for expected values - (3) in the paper
        expected_value = 0

        for j in range(size // 2):
            expected_value += 2 * (j + 1) * numerical_error_prob(n_errors=(j + 1), pass_size=size,
                                                                 qber=quantum_bit_error_rate)

        if expected_value <= max_expected_value:
            first_condition = True
        else:
            first_condition = False

        # Secondly we check condition for probabilities per se - (2) in the paper
        second_condition = False
        for j in range(size // 2):
            prob_sum = 0
            for k in list(np.arange(j + 1, size // 2 + 1, 1)):
                prob_sum += numerical_error_prob(n_errors=k, pass_size=size, qber=quantum_bit_error_rate)

            if prob_sum <= numerical_error_prob(n_errors=j, pass_size=size, qber=quantum_bit_error_rate) / 4:
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


def cascade_blocks_sizes(quantum_bit_error_rate, key_length, n_passes=1):
    """An iterative procedure to find the largest initial block size for the CASCADE algorithm,
    fulfilling conditions (2) and (3) as described in 1993 paper "Secret Key Reconciliation by Public Discussion"
    by Gilles Brassard and Louis Salvail, published in "Advances in Cryptography" proceedings.

    In this improved version of cascade_blocks_sizes functon the checks for the (2) & (3) of conditions from the '93
    CASCADE paper are simplified, resulting in lesser computational complexity. For additional context, these
    conditions are a system of non-linear inequalities that need to be fulfilled in order to have the probability
    of correcting at least 2 errors in a given block in any pass greater than 0.75"""
    max_expected_value = -1 * math.log(0.5, math.e)
    best_size = key_length

    """In this approach we implement dynamical storage of calculated error probabilities. We need to remember both 
    individual probabilities and their sums (series) for the recurrent formula of the 2nd condition. Additionally, we
    use a single formula for the expected value of number of errors in a given block after completion of the first
    CASCADE pass."""
    number_of_errors_probabilities = []
    series_of_probabilities = []

    for size in range(key_length // 2):  # we need at lest 2 blocks to begin with

        """Firstly we check condition for the expected value of number of errors remaining in a block
        in the first pass of CASCADE - (3) in the paper"""
        expected_value = size * quantum_bit_error_rate - (1 - (1 - 2 * quantum_bit_error_rate)**size) / 2
        if expected_value <= max_expected_value:
            first_condition = True
        else:
            first_condition = False

        """For the (2) condition (inequality) we first calculate all probabilities of numbers of errors from 0 errors 
        to size//2 errors and the first sum that appears on the left side of the inequality in the paper 
        (from 1 to size//2)."""
        for j in range(size // 2):
            number_of_errors_probabilities.append(
                numerical_error_prob(n_errors=j, pass_size=size, qber=quantum_bit_error_rate)
            )
        series_of_probabilities.append(sum(number_of_errors_probabilities) - number_of_errors_probabilities[0])

        """Next, in a loop we simply use the stored probabilities and their sums to check (2) for consecutive numbers
        of errors."""
        # for j in range(size // 2)

    # TODO: simplified 2nd check

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


def simulation_bb84(gain=1., alice_basis_length=256, rectilinear_basis_prob=0.5, disturbance_probability=0.1,
                    publication_probability_rectilinear=0.2, publication_probability_diagonal=0.2, cascade_n_passes=1,
                    error_estimation=refined_average_error):
    """This function is supposed to perform BB84 protocol with given parameters and return: error rate of the
    final key, length of the final key, a dictionary containing computational costs of consecutive parts of the
    algorithm, equal to number of bits exchanged (either via quantum or public channel) and number of last performed
    CASCADE pass - after each one error rate is tested and if equals 0, the error correction algorithm is stopped."""
    global_time_start = time.time()  # Beginning of time measurement for fitness evaluation

    """We want to store numbers of bits exchanged in each phase of the simulation: while using quantum channel,
    while performing sifting, error estimation and finally CASCADE.
    """
    computational_cost = {'qubits': alice_basis_length, 'sifting': 0, 'error estimation': 0,
                          'error correction': 0}  # we will systematically update this dictionary

    '''
    alice_basis = random_choice(length=alice_basis_length, p=rectilinear_basis_prob)
    alice_bits = random_choice(length=alice_basis_length, p=rectilinear_basis_prob)
    '''
    alice_basis_list = np.random.binomial(1, 1 - rectilinear_basis_prob, alice_basis_length)
    alice_basis = ''
    for basis in alice_basis_list:
        alice_basis += str(int(basis))

    alice_bits_list = np.random.binomial(1, 0.5, alice_basis_length)
    alice_bits = ''
    for bit in alice_bits_list:
        alice_bits += str(int(bit))

    """At this point it is impractical to encode states as bits because Bob's measurements results depend on both basis
    and bits choices of Alice, but he shouldn't know the first one. Because of that we will now translate Alice's bits
    to proper states, changing 0 and 1 into + and - for the diagonal basis, respectively.
    """

    i = 0
    alice_states_list = list(alice_bits)
    for bit in alice_bits:
        if alice_basis[i] == '1':
            if bit == '0':
                alice_states_list[i] = '+'
            elif bit == '1':
                alice_states_list[i] = '-'
            else:
                alice_states_list[i] = 'U'  # U for unknown
        i += 1

    alice_states = ''.join(alice_states_list)

    """Now that we have Alice's data rate, quantum channel's gain and Alice's states,
    we can randomly choose m (Alice's basis choices number) bases for Bob. While he performs his measurements,
    a portion of Alice's photons do not reach him due to loss on quantum channel. We will reflect that by choosing Bob's
    bases inside a for loop instead of using random_choice function defined earlier.
    """

    bob_basis_list = np.random.binomial(1, 1 - rectilinear_basis_prob, alice_basis_length)
    bob_basis = ''
    for basis in bob_basis_list:
        if random.uniform(0, 1) <= gain:  # for small numbers of bits quantum gain won't change anything in here
            bob_basis += str(int(basis))
        else:
            bob_basis += 'L'  # L for loss

    """There's a probability that due to disturbances in quantum channel or eavesdropping 
    some bits change while being sent.
    """

    """Now that we know the probability of change, we can perform such disturbances:"""

    received_states = ''
    change_states = {'0': '0', '1': '1', '2': '+', '3': '-'}  # dictionary for randomizing changed states
    for state in alice_states:  # to be updated following concrete attack strategies
        if random.uniform(0, 1) <= disturbance_probability:
            '''
            if state == '0':
                received_states += '1'
            elif state == '1':
                received_states += '0'
            else:
                received_states += 'C'  # C for change
            '''
            change_indicator = str(random.randint(0, 3))
            while state == change_states.get(
                    change_indicator):  # we repeat as long as it takes to actually change state
                change_indicator = str(random.randint(0, 3))
            received_states += change_states.get(change_indicator)
        else:
            received_states += state

    """After Bob chose a basis and a photon has reached him, he performs his measurement. If for this particular photon 
    he chooses the same basis as Alice chose before, his measurement result will be the same - that's because
    in reality Bob is choosing polarisators for photons to go through. If photon's polarization is the same as
    polarisators, then there's 100% probability of preserving this polarization. This is reflected by the same bit
    as before the measurement. Otherwise polarization after measurement is random. This mechanism is implemented in
    "measurement" function.
    
    Results of Bob's measurements are states - either 0 for |0> or 1 for |1> or + for |+> or - for |->. If he couldn't
    perform the measurement (encoded by basis L) then the result will be encoded in the string by L as well.
    """

    bob_states = ''
    for i in range(alice_basis_length):  # it's the same length as of Bob's basis choices
        bob_states += measurement(state=received_states[i], basis=bob_basis[i])

    """Now, having Bob's measurement results, we can translate states into bits."""

    bob_bits = ''
    for state in bob_states:
        try:
            bob_bits += str(states_mapping[state])  # we want to use indexing to raise errors for unsuccessful measur.
        except KeyError:
            bob_bits += 'E'  # E for error
            continue

    """Alice and Bob each have a string of bits, which will shortly become a key for cipher.
    At this point Alice and Bob can switch to communicating on a public channel. Their first step is to 
    perform sifting - decide which bits to keep in their key.
    
    Bob begins by telling Alice, which photons he measured. He then tells her which basis were used in each measurement.
    Then it's Alice's turn to send Bob her basis and to cancel out bits (representing states!) from her string 
    that do not match both successful Bob's measurement and his usage of the same basis in each case.
    """

    bob_measurement_indicators = ''
    for bit in bob_bits:  # is it possible to optimise length of such an indicator?
        if bit == '0' or bit == '1':
            bob_measurement_indicators += '1'
        else:
            bob_measurement_indicators += '0'

    bob_indicated_basis = ''
    bob_indicated_bits = ''
    alice_indicated_bits = ''
    alice_indicated_basis = ''
    index = 0

    for indicator in bob_measurement_indicators:  # in optimised approach send only bases for successful measurements
        if indicator == '1':
            bob_indicated_basis += bob_basis[index]
            bob_indicated_bits += bob_bits[index]
            alice_indicated_bits += alice_bits[index]
            alice_indicated_basis += alice_basis[index]
        index += 1

    """ Now we move towards the sifting itself. For the refined error analysis I'll need a string listing basis 
    choices by Alice for the sifted key, not only the bits:"""
    alice_sifted_key = ''
    alice_sifted_basis = ''
    index = 0
    for basis in alice_indicated_basis:
        if basis == bob_indicated_basis[index]:
            alice_sifted_key += alice_indicated_bits[index]
            alice_sifted_basis += alice_basis[index]
        index += 1

    """Now Bob gets info from Alice about her choices of bases, so that he can omit bits resulting from measurements
    when he used different basis than Alice. For him basis choices for bits in the sifted key are memorised, as well.
    """
    bob_sifted_key = ''
    bob_sifted_basis = ''

    index = 0
    for basis in bob_indicated_basis:
        if basis == alice_indicated_basis[index]:
            bob_sifted_key += bob_indicated_bits[index]
            bob_sifted_basis += bob_indicated_basis[index]
        index += 1

    """At this point we memorise number of bits exchanged during this phase"""
    computational_cost['sifting'] = len(bob_measurement_indicators) + len(alice_indicated_basis)

    """Sifted keys generally differ from each other due to changes between states sent by Alice and received by Bob.
    In order to estimate empirical probability of error occurrence in the sifted keys we can publish parts
    of keys, compare them and calculate numbers of errors. Then published parts of key should be deleted, as
    they have just been exchanged via the public channel.
    
    Originally I've been using 'normal' error estimation; naive one. Now I want to choose between this and refined one
    (Lo, Chau, Ardehali, 2004). For this purpose I defined above two functions: naive_error & refined_average_error.
    Below we will use (by default) the refined one.
    """

    if error_estimation == refined_average_error:
        error_estimation_results = refined_average_error(
            rect_prob=rectilinear_basis_prob,
            rect_pub_prob=publication_probability_rectilinear,
            diag_pub_prob=publication_probability_diagonal,
            alice_bits=alice_sifted_key,
            bob_bits=bob_sifted_key,
            alice_basis=alice_sifted_basis,
            bob_basis=bob_sifted_basis
        )
    else:
        error_estimation_results = naive_error(
            alice_key=alice_sifted_key,
            bob_key=bob_sifted_key,
            publication_prob_rect=publication_probability_rectilinear
        )

    error_estimate = error_estimation_results.get('error estimator')
    alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
    bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')

    """Bits published during error rate estimation do not count in the case of leaked information, 
    as they have been deleted from the raw key afterwards. Nevertheless, they still matter in regards to 
    computational cost:
    """
    computational_cost['error estimation'] = error_estimation_results.get('number of published bits')
    exchanged_bits_counter = 0

    """Naturally we assume it's Bob's key that's flawed.
    
    We begin by checking the parity of Alice's and Bob's sifted keys, 
    shortened by the subsets used for error estimation.
    
    CASCADE: 1st I need to assign bits to their indexes in original strings. Therefore I create dictionaries
    for Alice and for Bob.
    """
    n = len(alice_sifted_key_after_error_estimation)
    alice_cascade = {}
    bob_cascade = {}

    for i in range(n):  # I dynamically create dictionaries with indexes as keys and bits as values
        alice_cascade[str(i)] = alice_sifted_key_after_error_estimation[i]
        bob_cascade[str(i)] = bob_sifted_key_after_error_estimation[i]

    """Now we need to set up CASCADE itself: sizes of blocks in each pass, numeration of passes and a distionary
    for corrected bits with their indexes from original Bob's string as keys and correct bits as values.
    """

    blocks_sizes = cascade_blocks_sizes(quantum_bit_error_rate=error_estimate, key_length=n, n_passes=cascade_n_passes)

    """In order to return to blocks from earlier passes of CASCADE we need a history of blocks with indexes and bits,
    so implemented by dictionaries as list elements per pass, nested in general history list:
    """
    history = []
    error_rates = []
    pass_number = 0

    for size in blocks_sizes:
        try:
            pass_number_of_blocks = int(
                -1 * np.floor(-1 * n // size))  # I calculate how many blocks are in total in this pass
        except ZeroDivisionError:
            global_time_end = time.time()
            error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                             error_estimate, n, global_time_end - global_time_start,
                             rectilinear_basis_prob, publication_probability_rectilinear, cascade_n_passes]
            print(error_message)
            continue

        alice_pass_parity_list = []
        bob_pass_parity_list = []
        alice_blocks = []
        bob_blocks = []

        for block_index in cascade_blocks_generator(string_length=n, blocks_size=size):

            alice_block = {}  # a dictionary for a single block for Alice
            bob_block = {}  # a dictionary for a single block for Bob

            for index in block_index:  # I add proper bits to these dictionaries
                alice_block[str(index)] = alice_cascade[str(index)]
                bob_block[str(index)] = bob_cascade[str(index)]

            """I append single blocks created for given indexes to lists of block for this particular CASCADE's pass"""
            alice_blocks.append(alice_block)
            bob_blocks.append(bob_block)

        for i in range(pass_number_of_blocks):

            current_indexes = list(alice_blocks[i].keys())  # same as Bob's

            alice_current_bits = list(alice_blocks[i].values())
            bob_current_bits = list(bob_blocks[i].values())

            alice_bit_values = []
            bob_bit_values = []

            for j in range(len(current_indexes)):
                alice_bit_values.append(int(alice_current_bits[j]))
                bob_bit_values.append(int(bob_current_bits[j]))

            alice_pass_parity_list.append(sum(alice_bit_values) % 2)
            bob_pass_parity_list.append(sum(bob_bit_values) % 2)

            if alice_pass_parity_list[i] != bob_pass_parity_list[i]:  # we check if we should perform BINARY

                binary_results = binary(
                    sender_block=alice_blocks[i],
                    receiver_block=bob_blocks[i],
                    indexes=current_indexes
                )

                """Firstly we add the number of exchanged bits during this BINARY performance to the general number
                of bits exchanged via the public channel.
                """
                exchanged_bits_counter += binary_results[2]

                """Secondly we change main dictionary with final results and current blocks for history"""
                bob_cascade[binary_results[1]] = binary_results[0]
                bob_blocks[i][binary_results[1]] = binary_results[0]

                """Thirdly we change the error bit in blocks' history
                We need to perform BINARY on all blocks which we correct in history list
                history[number of pass][owner][number of block]
                """
                if pass_number > 0:  # in the first pass of CASCADE there are no previous blocks
                    for n_pass in range(pass_number):  # we check all previous passes
                        for n_block in range(
                                len(history[0][n_pass][1])):  # we check all Bob's blocks in each previous pass
                            if binary_results[1] in history[n_pass][1][n_block]:
                                history[n_pass][1][n_block] = binary_results[0]

                                try:
                                    if type(history[n_pass][1][n_block]) == str:
                                        indexes = ast.literal_eval(history[n_pass][1][n_block])
                                        binary_previous = binary(
                                            sender_block=history[n_pass][0][n_block],
                                            receiver_block=history[n_pass][1][n_block],
                                            indexes=indexes.keys()
                                        )
                                    elif type(history[n_pass][1][n_block]) == dict:
                                        binary_previous = binary(
                                            sender_block=history[n_pass][0][n_block],
                                            receiver_block=history[n_pass][1][n_block],
                                            indexes=history[n_pass][1][n_block].keys()
                                        )
                                except AttributeError:
                                    global_time_end = time.time()
                                    error_message = [blocks_sizes, alice_basis_length, gain, disturbance_probability,
                                                     error_estimate, n, global_time_end - global_time_start,
                                                     rectilinear_basis_prob, publication_probability_rectilinear,
                                                     cascade_n_passes, "AttributeError for binary_previous"]
                                    print(error_message)

                                    file = open("error.txt", "w")
                                    file.write('\n' + 'type of history: ' + str(type(history)) + '\n' + 'type of '
                                                                                                        'history['
                                                                                                        'n_pass]: ' +
                                               str(type(history[n_pass])) + '\n' + 'type of history[n_pass][1]: ' +
                                               str(type(history[n_pass][1])) + '\n' + 'type of history[n_pass][1]['
                                                                                      'n_block]: ' +
                                               str(type(history[n_pass][1][n_block])) + '\n' + str(history) + '\n')
                                    file.close()
                                    exit()

                                exchanged_bits_counter += binary_previous[2]
                                bob_cascade[binary_previous[1]] = binary_previous[0]
                                bob_blocks[i][binary_previous[1]] = binary_previous[0]

        history.append([alice_blocks, bob_blocks])
        pass_number += 1

        """For the purposes of optimizing CASCADE we check the error rate after each pass:"""
        alice_key_error_check = ''.join(list(alice_cascade.values()))
        bob_key_error_check = ''.join(list(bob_cascade.values()))

        "Poniższe już można by wbić w osobną funkcję"
        key_error_rate = 0
        index = 0
        for bit in alice_key_error_check:
            if bit != bob_key_error_check[index]:
                key_error_rate += 1
            index += 1
        try:
            key_error_rate = key_error_rate / len(alice_key_error_check)
            error_rates.append(key_error_rate)  # its length is equivalent to no. CASCADE passes performed
            if key_error_rate < 0.01:  # VERY ARBITRARY!!! + ryzyko odejmowania małych liczb? Co z tym?
                break  # let's not waste time for more CASCADE passes if there are 'no more' errors
        except ZeroDivisionError:
            global_time_end = time.time()
            error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                             error_estimate, n, global_time_end - global_time_start,
                             rectilinear_basis_prob, publication_probability_rectilinear, cascade_n_passes]
            print(error_message)

    """Time to create strings from cascade dictionaries into corrected keys"""
    alice_correct_key = ''.join(list(alice_cascade.values()))
    bob_correct_key = ''.join(list(bob_cascade.values()))

    """All that remains is to randomly choose number of bits for deletion, equal to number of exchanged bits
    during error correction phase. It's a form of a rudimentary privacy amplification. Let's say Alice randomly deletes 
    bits and informs Bob which indexes they were on, so that the computational cost would be equal 
    to the number of deleted bits.
    """

    deleted_bits_counter = 0
    try:
        deletion_prob = exchanged_bits_counter / len(alice_correct_key)
    except ZeroDivisionError:
        global_time_end = time.time()
        error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                         error_estimate, n, global_time_end - global_time_start,
                         rectilinear_basis_prob, publication_probability_rectilinear, cascade_n_passes]
        print(error_message)
        deletion_prob = 0  # no idea how to set it better in such a case
    index = 0

    while deleted_bits_counter < exchanged_bits_counter:
        if index == len(alice_correct_key):  # just in case we won't delete enough bits in the first 'run'
            index = 0
        if random.uniform(0, 1) <= deletion_prob:  # we "increase" the prob. by < OR =
            alice_correct_key = alice_correct_key[0: index:] + alice_correct_key[index + 1::]
            bob_correct_key = bob_correct_key[0: index] + bob_correct_key[index + 1::]
            deleted_bits_counter += 1

        index += 1

    """Now we finally have the proper keys"""
    global_time_end = time.time()
    computational_cost['error correction'] = exchanged_bits_counter + deleted_bits_counter  # should be an even number

    """Let's calculate key error rate"""
    final_key_error_rate = 0
    index = 0
    for bit in alice_correct_key:

        if bit != bob_correct_key[index]:
            final_key_error_rate += 1

        index += 1

    try:
        final_key_error_rate = final_key_error_rate / len(alice_correct_key)
    except ZeroDivisionError:
        global_time_end = time.time()
        error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                         error_estimate, n, global_time_end - global_time_start,
                         rectilinear_basis_prob, publication_probability_rectilinear, cascade_n_passes]
        print(error_message)
        final_key_error_rate = 1  # we set a max value to punish such a case
    key_length = len(alice_correct_key)

    results = {
        'error rate': final_key_error_rate,
        'global time': global_time_end - global_time_start,
        'key length': key_length,
        'comp. cost': computational_cost,
        'no. del. bits': deleted_bits_counter,  # just in the case
        'no. cascade pass.': len(error_rates),
        'cascade history': history,  # for the demonstrator
        'alice states': alice_states,  # for the demonstrator
        'bob states': bob_states,  # for the demonstrator
        'alice basis': alice_basis,  # for the demonstrator
        'bob basis': bob_basis,  # for the demonstrator
        'alice bits': alice_bits,  # for the demonstrator
        'bob bits': bob_bits,  # for the demonstrator
        'alice sifted key': alice_sifted_key,  # for the demonstrator
        'bob sifted key': bob_sifted_key,  # for the demonstrator
        'alice sifted key after error estimation': alice_sifted_key_after_error_estimation,  # for the demonstrator
        'bob sifted key after error estimation': bob_sifted_key_after_error_estimation,  # for the demonstrator
        'error estimate': error_estimate,  # for the demonstrator
        'alice correct key': alice_correct_key,  # for the demonstrator
        'bob correct_key': bob_correct_key  # for the demonstrator
    }

    return results
