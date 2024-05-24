import math
import random
import time

import numpy as np
from scipy.stats import binom
from scipy.special import betainc

from simulator_ver1.binary import binary


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
            left_side = sum_error_prob_betainc(
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


def cascade_blocks_generator(string_length, blocks_size):
    string_index = list(np.arange(0, string_length, 1))  # I create a list of all indexes (list of ndarray)
    blocks = random.sample(population=string_index, k=string_length)  # I shuffle the list randomly

    for j in range(0, string_length, blocks_size):  # I generate equally long chunks of shuffled indexes
        yield blocks[j:j + blocks_size]


class Block:
    """This class is representing a single block of the Cascade error correction algorithm. It contains bits
    with their indexes in the raw key in a given QKD protocol, in which the error correction is being run. It also
    holds in memory an information of which bits are erroneous for statistical analysis of the post-processing,
    for optimisation purposes.
    """
    size: int = None
    bits: dict = {}  # keys are indexes in the raw key and values are the bits
    erroneous_bits_indexes: list = []

    def __init__(self, size):  # TODO: differentiate between constructor and bit assignment?
        """In the constructor this class requires to get information of the expected size of this particular CASCADE
        block, the bits and their indexes to be assigned to this block.
        """
        self.size = size


class Cascade:
    """A general class for the CASCADE error correction algorithm for QKD. It is based mainly on the original
    proposition from 1993, with improvements motivated by lower computation cost & time. Variations of this algorithm
    are meant to be children of this parent class.

    This class is meant to enable tracking bits and their permutations (especially the erroneous ones) on the final key
    rate of a given QKD protocol.
    """
    current_pass_no: int = 0
    sets_of_blocks: dict = {}
    time_error_correction_start = None  # for time measurement
    time_error_correction_end = None
    raw_key_sender: str = ''
    raw_key_receiver: str = ''
    raw_key_length: int = None
    total_no_passes: int = None
    qber: float = None
    blocks_sizes: list = []
    sender_cascade: dict = {}
    receiver_cascade: dict = {}

    """In order to return to blocks from earlier passes of CASCADE we need to be able to access blocks from previous
    passes. For this purpose we create a history_cascade list, which will store for each pass a dict with lists
    of the blocks, accessible this way:

    history_cascade[number of the pass][either 'Alice blocks' or 'Bob blocks'][number of the block in the given pass]
    """
    history_cascade: list = []
    error_rates: list = []
    exchanged_bits_counter: int = 0

    def __init__(self, raw_key_sender, raw_key_receiver, quantum_bit_error_rate, number_of_passes=2):
        """In the constructor this class requires to get information of the expected size of this particular CASCADE
        block and raw keys of both sender and receiver (Alice & Bob), of equal length."""
        self.raw_key_sender = raw_key_sender
        self.raw_key_receiver = raw_key_receiver
        self.total_no_passes = number_of_passes
        self.qber = quantum_bit_error_rate

        if len(raw_key_sender) == len(raw_key_receiver):
            self.raw_key_length = len(raw_key_sender)
        else:
            raise ValueError(f"Expected raw_key_sender and raw_key_receiver to be strings of equal length, got"
                             f"length of sender's raw key equal to {len(raw_key_sender)} and receiver's equal to"
                             f"{len(raw_key_receiver)}")

    def _cascade_blocks_sizes(self):
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

        for size in list(np.arange(2, self.raw_key_length // 4 + 1, 1)):
            """We need at lest 4 blocks to begin with - then we can perform 2 passes.

            Firstly we check condition for the expected value of number of errors remaining in a block
            in the first pass of CASCADE - (3) in the paper
            """
            expected_value = size * self.qber - (1 - (1 - 2 * self.qber) ** size) / 2
            if expected_value > max_expected_value:
                """As for increasing sizes the expected value is non-decreasing. Thus, once the expected_value of number
                of errors remaining in the first block is greater than the max_expected_value, it'll always be.
                Therefore, it makes no sense to keep checking greater sizes - none of them will meet this condition.
                """
                break

            """For the (2) condition (inequality)..."""
            second_condition = True
            for j in list(np.arange(0, size // 2, 1)):
                """For number of errors equal to the amount of bits (or one bit fewer) in a block.

                When you analyse the inequality (2) carefully, you'll notice, that for 
                    j = size of the first pass // 2 - 1
                the sum on the left side contains the probability of having 2 * (j + 1) errors, which is equal to the 
                length of the first block. This means that for any j greater than the size of the first pass // 2 - 1 
                there are no expressions in the left-side sum "left", rendering it 0, which is always equal to or lesser 
                than a non-negative value of any probability.
                """
                right_side = numerical_error_prob(
                    n_errors=2 * j,
                    pass_size=size,
                    qber=self.qber) / 4
                left_side = sum_error_prob_betainc(
                    first_pass_size=size,
                    n_errors=2 * j,
                    qber=self.qber
                )

                # Now we check inequality (2) - must work for all possible numbers of errors in a block of given size
                if left_side > right_side:
                    second_condition = False
                    break

            if second_condition:
                if size > best_size:
                    best_size = size

        sizes = [best_size - best_size % 2]  # all sizes must be even numbers for BINARY (search for errors) to operate

        for j in range(self.total_no_passes - 1):  # corrected interpretation of number of passes
            next_size = 2 * sizes[-1]
            if next_size <= self.raw_key_length:
                sizes.append(next_size)
            else:
                break

        self.blocks_sizes = sizes

    def _cascade_blocks_generator(self, single_block_size):
        """This method generates lists of bits' indexes to be assigned for particular blocks."""
        string_index = list(np.arange(0, self.raw_key_length, 1))  # I create a list of all indexes (list of ndarray)
        blocks = random.sample(population=string_index, k=self.raw_key_length)  # I shuffle the list randomly

        for j in range(0, self.raw_key_length, single_block_size):  # I generate equally long chunks of shuffled indexes
            yield blocks[j:j + single_block_size]

    def execute(self):
        """
        CASCADE: 1st I need to assign bits to their indexes in original strings. Therefore, I create dictionaries
        for Alice (sender) and for Bob (receiver).
        """
        self.time_error_correction_start = time.time()

        # I dynamically create dictionaries with indexes as keys and bits as values
        for bit_index in range(self.raw_key_length):
            self.sender_cascade[str(bit_index)] = self.raw_key_sender[bit_index]
            self.receiver_cascade[str(bit_index)] = self.raw_key_receiver[bit_index]

        """Now we need to set up CASCADE itself: sizes of blocks in each pass, numeration of passes and a dictionary
        for corrected bits with their indexes from original Bob's string as keys and correct bits as values.
        """
        self._cascade_blocks_sizes()

        for size in self.blocks_sizes:
            """For nested loops we need to know how many blocks are in total in each pass"""
            try:
                pass_number_of_blocks = int(np.floor(self.raw_key_length // size))
            except ZeroDivisionError:
                error_message = 'ZeroDivisionError with size'
                print(error_message)
                continue

            alice_pass_parity_list = []
            bob_pass_parity_list = []
            alice_blocks = []
            bob_blocks = []

            for block_index in self._cascade_blocks_generator(single_block_size=size):
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
                    self.exchanged_bits_counter += binary_results.get('Bit counter')

                    """Secondly we change main dictionary with final results and current blocks for history"""
                    self.receiver_cascade[binary_correct_bit_index] = binary_correct_bit_value
                    bob_blocks[block_number][binary_correct_bit_index] = binary_correct_bit_value

                    """Thirdly we change the error bit in blocks' history_cascade:"""
                    if self.current_pass_no > 0:  # in the first pass of CASCADE there are no previous blocks
                        for n_pass in range(pass_number):  # we check all previous passes
                            previous_pass_blocks_alice = history_cascade[n_pass].get('Alice blocks')
                            previous_pass_blocks_bob = history_cascade[n_pass].get('Bob blocks')
                            for n_block in range(len(previous_pass_blocks_bob)):
                                """We check all Bob's blocks in each previous pass"""
                                if binary_correct_bit_index in previous_pass_blocks_bob[n_block]:
                                    previous_pass_blocks_bob[n_block][
                                        binary_correct_bit_index] = binary_correct_bit_value
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
                                        error_message = [blocks_sizes, alice_basis_length, gain,
                                                         disturbance_probability,
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
