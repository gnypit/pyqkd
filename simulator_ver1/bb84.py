"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
"""

import random
import time
import numpy as np

from simulator_ver2.error_estimation import naive_error, refined_average_error
from simulator_ver2.cascade import cascade_blocks_sizes, cascade_blocks_generator, count_key_value_differences

from simulator_ver1.binary import binary


# TODO: rename variable to have the content first, name later, e.g., bits_alice instead of alice_bits
# TODO: add a variable demonstration=FALSE, which makes the code omit parts which are for the demonstrator


states_mapping = {
    "0": 0,
    "1": 1,
    "+": 0,
    "-": 1
}


def measurement(state, basis):
    if basis == 1:
        if state == "+":
            result = 0
        else:
            result = 1
    else:
        result = state

    return result


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

    time_quantum_channel_start = time.time()

    """Lists of Alice's (sender's) basis choices and generated bits are created"""
    alice_basis = list(np.random.binomial(1, 1 - rectilinear_basis_prob, alice_basis_length))
    alice_bits = list(np.random.binomial(1, 0.5, alice_basis_length))

    """After we prepare basis choices and bits, we can simulate quantum channel's gain with a mask"""
    if gain < 1.0:
        # Generate a random mask where elements greater than 'gain' are marked as True
        mask = np.random.uniform(0, 1, alice_basis_length) > gain

        # Apply mask to set the chosen elements to None
        alice_bits[mask] = None

    """While in reality at this stage Bob (receiver) doesn't know the basis choices of Alice (sender) and is receiving 
    laser impulses instead of a whole 'message', for simulation purposes lists of Alice's basis choices and bits are now 
    transferred to Bob (receiver) to be efficiently compared with his basis choices and have his bits rendered.
    """
    bob_basis = np.random.binomial(1, 1 - rectilinear_basis_prob, alice_basis_length)
    bob_bits = []
    for index in range(alice_basis_length):
        if bob_basis[index] == alice_basis[index] and random.uniform(0, 1) > disturbance_probability:
            """If they measure in the same base and there's no disturbance in the quantum channel, 
            Bob's bit should be the same.
            """
            bob_bits.append(alice_bits[index])
        else:
            """If either they measure in a different base or there's disturbance in the quantum channel, 
            Bob's bit will be random.
            """
            bob_bits.append(random.randint(0, 1))

    """End of quantum channel stage of the BB84 protocol."""
    time_quantum_channel_end = time.time()
    time_history['qubits'] = time_quantum_channel_end - time_quantum_channel_start

    """Alice and Bob each have a list of bits, which will shortly become a key for cipher.
    At this point Alice and Bob can switch to communicating on a public channel. Their first step is to 
    perform sifting - decide which bits to keep in their key.
    
    Alice and Bob (sender and receiver, respectively) exchange information on basis used. Checking successful 
    measurements is not performed explicitly in this simulation, since the quantum channel's gain has already been
    applied to Alice's (sender's) bits using a mask above.
    """
    time_sifting_start = time.time()
    alice_sifted_key = []
    bob_sifted_key = []
    sifted_basis = []

    for index in range(alice_basis_length):
        alice_base = alice_basis[index]  # could be used twice, so the values is remembered beforehand
        if alice_base == bob_basis[index]:
            sifted_basis.append(alice_base)
            alice_sifted_key.append(alice_bits[index])
            bob_sifted_key.append(bob_bits[index])

    """End of the sifting phase"""
    key_length_history['sifting'] = len(bob_sifted_key)
    time_sifting_end = time.time()
    time_history['sifting'] = time_sifting_end - time_sifting_start

    """Sifted keys generally differ from each other due to changes between states sent by Alice and received by Bob.
    Above, that is simulated by random.randint() if Alice's and Bob's basis are different and using the 
    'disturbance_probability' to account for random 'flipping' of a bit (state) even though it reaches Bob successfully,
    after the quantum channel's gain is simulated.
    
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
            basis=sifted_basis
        )

        error_estimate = error_estimation_results.get('error estimator')
        alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
        bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')
    elif error_estimation == naive_error:
        error_estimation_results = naive_error(
            alice_key=alice_sifted_key,
            bob_key=bob_sifted_key,
            publication_probability=publication_probability_rectilinear
        )

        error_estimate = error_estimation_results.get('error estimator')
        alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
        bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')
    else:
        error_estimate = disturbance_probability
        alice_sifted_key_after_error_estimation = alice_sifted_key  # it is a list
        bob_sifted_key_after_error_estimation = bob_sifted_key  # it is a list

    key_len = len(alice_sifted_key_after_error_estimation)
    key_length_history['error estimation'] = key_len
    time_error_estimation_end = time.time()
    time_history['error estimation'] = time_error_estimation_end - time_error_estimation_start

    """In BB84, it is the receiver's (Bob's) key that contains errors. Error correction algorithm 'CASCADE' is thus
    performed, to use binary searches for errors in subsets of the sifted keys. For optimisation purposes all of the 
    blocks from all of the CASCADE's passes will be stored and QBER exact values will be computed after each pass.  
    
    CASCADE: 
        1) Bits are assigned to their indexes in new dicts, which will be gradually updated with corrected values.
        2) Block sizes for the whole CASCADE algorithm (all passes) are computed.
        3) 
    """
    time_error_correction_start = time.time()
    alice_cascade = {}
    bob_cascade = {}

    for index in range(key_len):  # I dynamically create dictionaries with indexes as keys and bits as values
        alice_cascade[index] = alice_sifted_key_after_error_estimation[index]
        bob_cascade[index] = bob_sifted_key_after_error_estimation[index]

    blocks_sizes = cascade_blocks_sizes(  # these are computed for all the given number of CASCADE's passes
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
            print(f"ZeroDivisionError. Computed sizes of the CASCADE's blocks are {blocks_sizes}.")
            continue

        alice_pass_parity_list = []
        bob_pass_parity_list = []
        alice_blocks = []
        bob_blocks = []

        for block_index in cascade_blocks_generator(string_length=key_len, blocks_size=size):

            alice_block = {}  # a dictionary for a single block for Alice
            bob_block = {}  # a dictionary for a single block for Bob

            for index in block_index:  # I add proper bits to these dictionaries
                alice_block[str(index)] = alice_cascade[index]
                bob_block[str(index)] = bob_cascade[index]

            """I append single blocks created for given indexes to lists of block for this particular CASCADE's pass"""
            alice_blocks.append(alice_block)
            bob_blocks.append(bob_block)

        for index in range(pass_number_of_blocks):

            current_indexes = list(alice_blocks[index].keys())  # same as Bob's

            alice_current_bits = list(alice_blocks[index].values())
            bob_current_bits = list(bob_blocks[index].values())

            alice_bit_values = []
            bob_bit_values = []

            for j in range(len(current_indexes)):
                alice_bit_values.append(int(alice_current_bits[j]))
                bob_bit_values.append(int(bob_current_bits[j]))

            alice_pass_parity_list.append(sum(alice_bit_values) % 2)
            bob_pass_parity_list.append(sum(bob_bit_values) % 2)

            if alice_pass_parity_list[index] != bob_pass_parity_list[index]:
                """Since parities of given blocks are different for Alice and Bob, Bob must have an odd number
                of errors; we we should search for them - and correct one of them - with BINARY"""
                binary_results = binary(
                    sender_block=alice_blocks[index],
                    receiver_block=bob_blocks[index],
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
                bob_blocks[index][binary_correct_bit_index] = binary_correct_bit_value

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
                                    bob_blocks[index][
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

        """For the purposes of optimizing CASCADE, the exact QBER is computed and remembered after each pass:"""
        qber_after_pass = count_key_value_differences(dict1=alice_cascade, dict2=bob_cascade) / len(alice_cascade)
        error_rates.append(qber_after_pass)
        if qber_after_pass == 0:
            """If all the errors have already been corrected, CASCADE is terminated."""
            break

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


def main():  # for testing & debugging
    simulation_bb84()


if __name__ == "__main__":
    main()
