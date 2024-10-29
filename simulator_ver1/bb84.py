"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
"""

import random
import time
import numpy as np

from simulator_ver2.error_estimation import naive_error, refined_average_error
from simulator_ver2.cascade import cascade_blocks_sizes, cascade_blocks_sizes_old, cascade_blocks_generator

from simulator_ver1.binary import binary


# TODO: rename variable to have the content first, name later, e.g., bits_alice instead of alice_bits
# TODO: add a variable demonstration=FALSE, which makes the code omit parts which are for the demonstrator


def simulation_bb84(gain=1., alice_basis_length=256, rectilinear_basis_prob=0.5, disturbance_probability=0.1,
                    publication_probability_rectilinear=0.2, publication_probability_diagonal=0.2, cascade_n_passes=1,
                    error_estimation=refined_average_error):
    """This function is supposed to perform BB84 protocol with given parameters and return: error rate of the
    final key, length of the final key, a dictionary containing computational costs of consecutive parts of the
    algorithm, equal to number of bits exchanged (either via quantum or public channel) and number of last performed
    CASCADE pass - after each one error rate is tested and if equals 0, the error correction algorithm is stopped."""

    """I want to have a history of key lengths through the whole simulation: after using the quantum channel,
    after performing sifting, error estimation and finally CASCADE.
    """
    key_length_history = {
        'qubits': alice_basis_length,  # after qubits exchange in the quantum channel
        'sifting': 0,  # after sifting phase
        'error estimation': 0,  # after error estimation phase
        'error correction': 0  # after error correction phase (CASCADE)
    }  # we will systematically update this dictionary

    """After I optimised the search for the initial CASCADE block size, I want to measure the time needed for each
    phase
    """
    time_history = {
        'qubits': 0,  # after qubits exchange in the quantum channel
        'sifting': 0,  # after sifting phase
        'error estimation': 0,  # after error estimation phase
        'error correction': 0  # after error correction phase (CASCADE)
    }  # we will systematically update this dictionary

    # TODO: which is better for research, lists or strings? For visualisation strings.

    time_quantum_channel_start = time.time()

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

    block_number = 0
    alice_states_list = list(alice_bits)
    for bit in alice_bits:
        if alice_basis[block_number] == '1':
            if bit == '0':
                alice_states_list[block_number] = '+'
            elif bit == '1':
                alice_states_list[block_number] = '-'
            else:
                alice_states_list[block_number] = 'U'  # U for unknown
        block_number += 1

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
    for block_number in range(alice_basis_length):  # it's the same length as of Bob's basis choices
        bob_states += measurement(state=received_states[block_number], basis=bob_basis[block_number])

    """Now, having Bob's measurement results, we can translate states into bits."""

    bob_bits = ''
    for state in bob_states:
        try:
            bob_bits += str(states_mapping[state])  # we want to use indexing to raise errors for unsuccessful measur.
        except KeyError:
            bob_bits += 'E'  # E for error
            continue

    """End of quantum channel measurements."""
    time_quantum_channel_end = time.time()
    time_history['qubits'] = time_quantum_channel_end - time_quantum_channel_start

    """Alice and Bob each have a string of bits, which will shortly become a key for cipher.
    At this point Alice and Bob can switch to communicating on a public channel. Their first step is to 
    perform sifting - decide which bits to keep in their key.
    
    Bob begins by telling Alice, which photons he measured. He then tells her which basis were used in each measurement.
    Then it's Alice's turn to send Bob her basis and to cancel out bits (representing states!) from her string 
    that do not match both successful Bob's measurement and his usage of the same basis in each case.
    """

    time_sifting_start = time.time()

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

    """End of the sifting phase"""
    key_length_history['sifting'] = len(bob_sifted_key)
    time_sifting_end = time.time()
    time_history['qubits'] = time_sifting_end - time_sifting_start

    """Sifted keys generally differ from each other due to changes between states sent by Alice and received by Bob.
    In order to estimate empirical probability of error occurrence in the sifted keys we can publish parts
    of keys, compare them and calculate numbers of errors. Then published parts of key should be deleted, as
    they have just been exchanged via the public channel.
    
    Originally I've been using 'normal' error estimation; naive one. Now I want to choose between this and refined one
    (Lo, Chau, Ardehali, 2004). For this purpose I defined above two functions: naive_error & refined_average_error.
    Below we will use (by default) the refined one.
    """

    time_error_estimation_start = time.time()

    if error_estimation == refined_average_error:
        # TODO: better handling of the estimator cases
        error_estimation_results = refined_average_error(
            rect_prob=rectilinear_basis_prob,
            rect_pub_prob=publication_probability_rectilinear,
            diag_pub_prob=publication_probability_diagonal,
            alice_bits=alice_sifted_key,
            bob_bits=bob_sifted_key,
            alice_basis=alice_sifted_basis,
            bob_basis=bob_sifted_basis
        )

        error_estimate = error_estimation_results.get('error estimator')
        alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
        bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')
    elif error_estimation == naive_error:
        error_estimation_results = naive_error(
            alice_key=alice_sifted_key,
            bob_key=bob_sifted_key,
            publication_prob_rect=publication_probability_rectilinear
        )

        error_estimate = error_estimation_results.get('error estimator')
        alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
        bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')
    else:
        error_estimate = disturbance_probability
        alice_sifted_key_after_error_estimation = alice_sifted_key
        bob_sifted_key_after_error_estimation = bob_sifted_key

    key_len = len(alice_sifted_key_after_error_estimation)
    key_length_history['error estimation'] = key_len
    time_error_estimation_end = time.time()
    time_history['error estimation'] = time_error_estimation_end - time_error_estimation_start

    """Naturally we assume it's Bob's key that's flawed.
    
    We begin by checking the parity of Alice's and Bob's sifted keys, 
    shortened by the subsets used for error estimation.
    
    CASCADE: 1st I need to assign bits to their indexes in original strings. Therefore I create dictionaries
    for Alice and for Bob.
    """
    time_error_correction_start = time.time()
    alice_cascade = {}
    bob_cascade = {}

    for block_number in range(key_len):  # I dynamically create dictionaries with indexes as keys and bits as values
        alice_cascade[str(block_number)] = alice_sifted_key_after_error_estimation[block_number]
        bob_cascade[str(block_number)] = bob_sifted_key_after_error_estimation[block_number]

    """Now we need to set up CASCADE itself: sizes of blocks in each pass, numeration of passes and a dictionary
    for corrected bits with their indexes from original Bob's string as keys and correct bits as values.
    """

    blocks_sizes = cascade_blocks_sizes(
        quantum_bit_error_rate=error_estimate,
        key_length=key_len,
        n_passes=cascade_n_passes
    )

    """In order to return to blocks from earlier passes of CASCADE we need to be able to access blocks from previous
    passes. For this purpose we create a history_cascade list, which will store for each pass a dict with lists
    of the blocks, accessible this way:
    
    history_cascade[number of the pass][either 'Alice blocks' or 'Bob blocks'][number of the block in the given pass]
    """
    history_cascade = []
    error_rates = []
    pass_number = 0
    exchanged_bits_counter = 0

    for size in blocks_sizes:
        """For nested loops we need to know how many blocks are in total in each pass"""
        try:
            pass_number_of_blocks = int(np.floor(key_len // size))
        except ZeroDivisionError:
            error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                             error_estimate, key_len, rectilinear_basis_prob, publication_probability_rectilinear,
                             cascade_n_passes, 'ZeroDivisionError with size']
            print(error_message)
            continue

        alice_pass_parity_list = []
        bob_pass_parity_list = []
        alice_blocks = []
        bob_blocks = []

        for block_index in cascade_blocks_generator(string_length=key_len, blocks_size=size):

            alice_block = {}  # a dictionary for a single block for Alice
            bob_block = {}  # a dictionary for a single block for Bob

            for index in block_index:  # I add proper bits to these dictionaries
                alice_block[str(index)] = alice_cascade[str(index)]
                bob_block[str(index)] = bob_cascade[str(index)]

            """I append single blocks created for given indexes to lists of block for this particular CASCADE's pass"""
            alice_blocks.append(alice_block)
            bob_blocks.append(bob_block)

        for block_number in range(pass_number_of_blocks):

            current_indexes = list(alice_blocks[block_number].keys())  # same as Bob's

            alice_current_bits = list(alice_blocks[block_number].values())
            bob_current_bits = list(bob_blocks[block_number].values())

            alice_bit_values = []
            bob_bit_values = []

            for j in range(len(current_indexes)):
                alice_bit_values.append(int(alice_current_bits[j]))
                bob_bit_values.append(int(bob_current_bits[j]))

            alice_pass_parity_list.append(sum(alice_bit_values) % 2)
            bob_pass_parity_list.append(sum(bob_bit_values) % 2)

            if alice_pass_parity_list[block_number] != bob_pass_parity_list[block_number]:
                """Since parities of given blocks are different for Alice and Bob, Bob must have an odd number
                of errors; we we should search for them - and correct one of them - with BINARY"""
                binary_results = binary(
                    sender_block=alice_blocks[block_number],
                    receiver_block=bob_blocks[block_number],
                    indexes=current_indexes
                )
                binary_correct_bit_value = binary_results.get('Correct bit value')
                binary_correct_bit_index = binary_results.get('Corrected bit index')

                """Firstly we add the number of exchanged bits during this BINARY performance to the general number
                of bits exchanged via the public channel.
                """
                exchanged_bits_counter += binary_results.get('Bit counter')

                """Secondly we change main dictionary with final results and current blocks for history"""
                bob_cascade[binary_correct_bit_index] = binary_correct_bit_value
                bob_blocks[block_number][binary_correct_bit_index] = binary_correct_bit_value

                """Thirdly we change the error bit in blocks' history_cascade:"""
                if pass_number > 0:  # in the first pass of CASCADE there are no previous blocks
                    for n_pass in range(pass_number):  # we check all previous passes
                        previous_pass_blocks_alice = history_cascade[n_pass].get('Alice blocks')
                        previous_pass_blocks_bob = history_cascade[n_pass].get('Bob blocks')
                        for n_block in range(len(previous_pass_blocks_bob)):
                            """We check all Bob's blocks in each previous pass"""
                            if binary_correct_bit_index in previous_pass_blocks_bob[n_block]:
                                previous_pass_blocks_bob[n_block][binary_correct_bit_index] = binary_correct_bit_value
                                try:
                                    binary_previous = binary(
                                        sender_block=previous_pass_blocks_alice[n_block],
                                        receiver_block=previous_pass_blocks_bob[n_block],
                                        indexes=list(previous_pass_blocks_alice[n_block].keys())
                                    )

                                    exchanged_bits_counter += binary_previous.get('Bit counter')
                                    bob_cascade[binary_previous['Corrected bit index']] = binary_previous.get(
                                        'Correct bit value')
                                    bob_blocks[block_number][
                                        binary_previous['Corrected bit index']] = binary_previous.get(
                                        'Correct bit value')
                                except AttributeError:
                                    error_message = [blocks_sizes, alice_basis_length, gain, disturbance_probability,
                                                     error_estimate, key_len, rectilinear_basis_prob,
                                                     publication_probability_rectilinear,
                                                     cascade_n_passes, "AttributeError for binary_previous"]
                                    print(error_message)
                                    return error_message
                                except KeyError:
                                    print("KeyError for binary_previous")
                                    print(previous_pass_blocks_alice[n_block])
                                    print(previous_pass_blocks_bob[n_block])
                                    print(list(previous_pass_blocks_bob[n_block].keys()))

        history_cascade.append({'Alice blocks': alice_blocks, 'Bob blocks': bob_blocks})
        pass_number += 1

        """For the purposes of optimizing CASCADE we check the error rate after each pass:"""
        alice_key_error_check = ''.join(list(alice_cascade.values()))
        bob_key_error_check = ''.join(list(bob_cascade.values()))

        # TODO: can we make it a separate function?
        key_error_rate = 0
        index = 0
        for bit in alice_key_error_check:
            if bit != bob_key_error_check[index]:
                key_error_rate += 1
            index += 1
        try:
            key_error_rate = key_error_rate / len(alice_key_error_check)
            error_rates.append(key_error_rate)  # its length is equivalent to no. CASCADE passes performed
            if key_error_rate < 0.0001:  # TODO: is 0.1% a small enough number?
                break  # let's not waste time for more CASCADE passes if there are 'no more' errors
        except ZeroDivisionError:
            error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                             error_estimate, key_len, rectilinear_basis_prob, publication_probability_rectilinear,
                             cascade_n_passes, 'ZeroDivisionError with len(alice_key_error_check)']
            print(error_message)

    """Time to create strings from cascade dictionaries into corrected keys"""
    alice_correct_key = ''.join(list(alice_cascade.values()))
    bob_correct_key = ''.join(list(bob_cascade.values()))
    time_error_correction_end = time.time()
    time_history['error correction'] = time_error_correction_end - time_error_correction_start

    """All that remains is to randomly choose number of bits for deletion, equal to number of exchanged bits
    during error correction phase. It's a form of a rudimentary privacy amplification. Let's say Alice randomly deletes 
    bits and informs Bob which indexes they were on, so that the computational cost would be equal 
    to the number of deleted bits.
    """

    deleted_bits_counter = 0
    try:
        deletion_prob = exchanged_bits_counter / len(alice_correct_key)
    except ZeroDivisionError:
        error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                         error_estimate, key_len, rectilinear_basis_prob, publication_probability_rectilinear,
                         cascade_n_passes, 'ZeroDivisionError with len(alice_correct_key)']
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

    """Now we finally have the proper keys. Let's calculate key error rate"""
    final_key_error_rate = 0
    index = 0
    for bit in alice_correct_key:

        if bit != bob_correct_key[index]:
            final_key_error_rate += 1

        index += 1

    try:
        final_key_error_rate = final_key_error_rate / len(alice_correct_key)
    except ZeroDivisionError:
        error_message = [blocks_sizes, pass_number, alice_basis_length, gain, disturbance_probability,
                         error_estimate, key_len, rectilinear_basis_prob, publication_probability_rectilinear,
                         cascade_n_passes, 'ZeroDivisionError with len(alice_correct_key)']
        print(error_message)
        final_key_error_rate = 1  # we set a max value to punish such a case
    key_length_history['error correction'] = len(alice_correct_key)

    results = {
        'error rate': final_key_error_rate,
        'time_history': time_history,
        'key length history': key_length_history,
        'no. del. bits': deleted_bits_counter,
        'no. cascade pass.': len(error_rates),  # TODO: why is it None???
        'cascade history': history_cascade,
        'alice states': alice_states,
        'bob states': bob_states,
        'alice basis': alice_basis,
        'bob basis': bob_basis,
        'alice bits': alice_bits,
        'bob bits': bob_bits,
        'alice sifted key': alice_sifted_key,
        'bob sifted key': bob_sifted_key,
        'alice sifted key after error estimation': alice_sifted_key_after_error_estimation,
        'bob sifted key after error estimation': bob_sifted_key_after_error_estimation,
        'error estimate': error_estimate,
        'alice correct key': alice_correct_key,
        'bob correct_key': bob_correct_key
    }

    return results
