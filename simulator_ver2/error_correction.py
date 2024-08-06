# TODO: can I save a class as a JSON to work on later? I run the code, do something with an object, save it and than load it with it's fields and method to run again?

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


class PairOfBlocks:
    """This class is representing a single block of the Cascade error correction algorithm. It contains bits
    with their indexes in the raw key in a given QKD protocol, in which the error correction is being run. It also
    holds in memory the information of which bits are erroneous for statistical analysis of the post-processing,
    for optimization purposes.
    """
    size: int = None
    indexes: list = []
    sender_bits: dict[int, int]  # keys are indexes in the raw key and values are the bits
    receiver_bits: dict[int, int]  # keys are indexes in the raw key and values are the bits
    erroneous_bits_indexes: list = []
    original_sender_bits: list[int]
    original_receiver_bits: list[int]

    def __init__(self, size):
        """In the constructor, this class requires the expected size of this particular CASCADE block."""
        self.size = size

    def add_bits(self, index: int, sender_bit: int,
                 receiver_bit: int):  # I want to store indexes as numbers for easier statistical analysis afterwards
        """This method is called upon inside the Cascade's execute() main method, while assigning bits for error
        correction to a given block in a given pass.
        """
        self.sender_bits[index] = sender_bit
        self.receiver_bits[index] = receiver_bit
        self.indexes.append(index)

    def flag_errors(self):
        """To track all actual errors, to verify the influence of the errors' grouping on the final key rate
        in QKD protocols, this method bluntly compares received sender's and receiver's bits. It creates strings of the
        original bits and a list with indexes of the erroneous ones."""
        for index in self.indexes:
            sender_bit = self.sender_bits.get(index)
            receiver_bit = self.receiver_bits.get(index)

            self.original_sender_bits.append(sender_bit)
            self.original_receiver_bits.append(receiver_bit)

            if sender_bit != receiver_bit:
                self.erroneous_bits_indexes.append(index)

    def get_original_bits(self):
        """This method returns results of the flag_errors() as a dict, if necessary."""
        results = {
            'original sender string': self.original_sender_bits,
            'original receiver string': self.original_receiver_bits,
            'indexes of all erroneous bits': self.erroneous_bits_indexes
        }

        return results

    def parity_check(self):
        """This method computes the parity of strings of bits for both Alice (sender) and Bob (receiver)."""
        sender_parity = sum(list(self.sender_bits.values())) % 2
        receiver_parity = sum(list(self.receiver_bits.values())) % 2

        return sender_parity, receiver_parity

    def binary(self):
        """
        Contrary to real-life applications in this simulation of the BINARY algorithm Alice (sender) and Bob (receiver)
        do not exchange messages. Instead, we count how many bits should be exchanged between them so that the algorithm
        would end successfully. Afterwards, we return this value together with the bit to be changed as a result of the
        algorithm.
        """
        is_binary = True
        bit_counter = 0

        while is_binary:
            """Sender starts by sending to the Receiver parity of the first half of her string"""
            half_index = len(self.sender_bits) // 2  # same as Bob's
            first_half_indexes = self.indexes[0:half_index:1]  # same as Bob's
            sender_first_half_list = []

            for index in first_half_indexes:
                sender_first_half_list.append(int(self.sender_bits.get(index)))

            sender_first_half_parity = sum(sender_first_half_list) % 2
            bit_counter += 1  # At this point sender informs receiver about their 1st half's parity

            """Now Receiver determines whether an odd number of errors occurred in the first or in the
            second half by testing the parity of his string and comparing it to the parity sent
            by Sender
            """
            receiver_first_half_list = []

            for index in first_half_indexes:
                receiver_first_half_list.append(int(self.receiver_bits.get(index)))

            receiver_first_half_parity = sum(receiver_first_half_list) % 2

            """Single (at least) error is in the 'half' of a different parity; we change current strings
            that are analysed into halves of different parities until one bit is left - the error
            """
            sender_subscription_block = {}
            receiver_subscription_block = {}

            if receiver_first_half_parity != sender_first_half_parity:
                bit_counter += 1  # At this point, the receiver sends a mess. About an odd number of errors in 1st half
                for index in first_half_indexes:
                    receiver_subscription_block[index] = self.receiver_bits.get(index)
                    sender_subscription_block[index] = self.sender_bits.get(index)

                sender_current_block = sender_subscription_block
                receiver_current_block = receiver_subscription_block

                indexes = list(sender_current_block.keys())  # same as Bob's
            else:
                bit_counter += 1  # At this point, the receiver sends a mess. about an odd number of errors in 2nd half

                """We have to repeat the whole procedure for the second halves"""
                second_half_indexes = self.indexes[half_index::1]

                for index in second_half_indexes:
                    receiver_subscription_block[index] = self.receiver_bits.get(index)
                    sender_subscription_block[index] = self.sender_bits.get(index)

                sender_current_block = sender_subscription_block
                receiver_current_block = receiver_subscription_block

                indexes = list(sender_current_block.keys())  # same as Bob's

            if len(receiver_current_block) == 1:  # at some point, this clause will be true
                bit_counter += 1  # At this point receiver would send a message (?) about one bit left and changing it

                """Finally we change the error bit in Bob's original dictionary of all bits"""
                if receiver_current_block[indexes[0]] == 0:
                    self.receiver_bits[indexes[0]] = 1
                    return {'Correct bit value': 1, 'Corrected bit index': indexes[0], 'Bit counter': bit_counter}
                else:
                    self.receiver_bits[indexes[0]] = 0
                    return {'Correct bit value': 0, 'Corrected bit index': indexes[0], 'Bit counter': bit_counter}


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
    raw_key_sender: list[int]  # list of bits as ints consumes less memory than a string of bits
    raw_key_receiver: list[int]  # list of bits as ints consumes less memory than a string of bits
    raw_key_length: int = None
    total_no_passes: int = None
    qber: float = None
    blocks_sizes: list = []
    sender_cascade: dict = {}  # is it even necessary?
    receiver_cascade: dict = {}
    corrected_bits_history: dict = {}

    """In order to return to blocks from earlier passes of CASCADE we need to be able to access blocks from previous
    passes. For this purpose we create a history_cascade list, which will store for each pass a dict with lists
    of the blocks, accessible this way:

    history_cascade[number of the pass][either 'Alice blocks' or 'Bob blocks'][number of the block in the given pass]
    """
    history_cascade: list = []  # TODO: actually, history is not required as such anymore, since we have an object for CASCADE and blocks saved within it as separate objects; instead the list of these pairs of blocks could be something between a pandas DataFrame and a dict
    error_rates: list = []
    exchanged_bits_counter: int = 0
    report: str

    def __init__(self, raw_key_sender, raw_key_receiver, quantum_bit_error_rate, number_of_passes=2,
                 blocks_sizes_method='automatic', initial_block_size=None):  # TODO add checking type, etc.
        """In the constructor, this class requires raw keys of both sender and receiver (Alice & Bob), of equal length,
        QBER and the number of iterations/passes of the algorithm to be performed. Additionally, a method for specifying
        blocks' length should be provided. By default, it's the one described in the '93 paper and implemented
        in a private method '_cascade_blocks_sizes'. Alternatively, one can choose a 'manual' approach and set the sizes
        in advance, treated as a constant from now on.
        """
        self.raw_key_sender = raw_key_sender
        self.raw_key_receiver = raw_key_receiver
        self.total_no_passes = number_of_passes
        self.qber = quantum_bit_error_rate
        self.blocks_sizes_method = blocks_sizes_method
        self.initial_block_size = initial_block_size
        self.report = "Instance of class Cascade created\n"

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

        In this improved version of cascade_blocks_sizes function, the checks for the (2) & (3) of conditions from the
        '93 CASCADE paper are simplified, resulting in lesser computational complexity. For additional context, these
        conditions are a system of non-linear inequalities that need to be fulfilled to have the probability of
        correcting at least two errors in a given block in any pass greater than 0.75

        In this approach, we implement a regularized incomplete beta function to represent binomial distribution CDF
        for a simplified left side of the (2) inequality from that paper.

        Additionally, we use a single formula for the expected value (3) of number of errors in a given block after
        completion of the first CASCADE pass.
        """
        max_expected_value = -1 * math.log(0.5, math.e)
        if self.blocks_sizes_method == 'automatic':
            """All sizes fulfilling (2) & (3) ineq. will be greater than that; we're looking fo the greatest"""
            self.initial_block_size = 0

            for size in list(np.arange(2, self.raw_key_length // 4 + 1, 1)):
                """We need at lest 4 blocks to begin with - then we can perform 2 passes.

                Firstly we check condition for the expected value of number of errors remaining in a block
                in the first pass of CASCADE - (3) in the paper
                """
                expected_value = size * self.qber - (1 - (1 - 2 * self.qber) ** size) / 2
                if expected_value > max_expected_value:
                    """As for increasing sizes the expected value is non-decreasing. Thus, once the expected_value of 
                    number of errors remaining in the first block is greater than the max_expected_value, it'll always 
                    be. Therefore, it makes no sense to keep checking greater sizes - none of them will meet this 
                    condition.
                    """
                    break

                """For the (2) condition (inequality)..."""
                second_condition = True
                for j in list(np.arange(0, size // 2, 1)):
                    """For number of errors equal to the amount of bits (or one bit fewer) in a block.

                    When you analyse the inequality (2) carefully, you'll notice, that for 
                        j = size of the first pass // 2 - 1
                    the sum on the left side contains the probability of having 2 * (j + 1) errors, which is equal to 
                    the length of the first block. This means that for any j greater than the size of 
                        the first pass // 2 - 1 
                    there are no expressions in the left-side sum "left", rendering it 0, which is always equal to or 
                    lesser than a non-negative value of any probability.
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

                    """Now we check inequality (2) - must work for all possible numbers of errors 
                    in a block of given size
                    """
                    if left_side > right_side:
                        second_condition = False
                        break

                if second_condition:
                    if size > self.initial_block_size:
                        self.initial_block_size = size

        """At this point we have an initial size of the block either computed or manually provided.
        
        Since all sizes must be even numbers for BINARY (search for errors) to operate, we subtract from initial size
        it's modulo 2. We also create a list of all sizes with the initial size as a first element, to be used in
        a loop to compute other sizes.
        """
        sizes = [self.initial_block_size - self.initial_block_size % 2]

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

    def execute(self):  # should this be a recurrent function???
        """CASCADE: First, bits need to be assigned to their indexes in original strings."""
        self.time_error_correction_start = time.time()

        """I dynamically create dictionaries with indexes as keys and bits as values"""
        for bit_index in range(self.raw_key_length):
            self.sender_cascade[str(bit_index)] = self.raw_key_sender[bit_index]
            self.receiver_cascade[str(bit_index)] = self.raw_key_receiver[bit_index]

        """Now we need to set up CASCADE itself: sizes of blocks in each pass, numeration of passes and a dictionary
        for corrected bits with their indexes from original Bob's string as keys and correct bits as values.
        """
        self._cascade_blocks_sizes()
        self.report += "Block sizes computed\n"

        for size in self.blocks_sizes:  # for each CASCADE pass the size of blocks is different
            """For nested loops we need to know how many blocks are in total in each pass"""
            try:
                pass_number_of_blocks = int(np.floor(self.raw_key_length // size))
            except ZeroDivisionError:
                var = self.__dict__
                raise ZeroDivisionError(f"ZeroDivisionError with size. Current field values are: {var}")

            self.report += "Blocks are planned\n"

            """Time to generate blocks for this pass. For each pair of blocks of bits with the same indexes in the raw 
            key - one with Alice's (sender) bits, and the other one with Bob's (receiver) bits - an instance of
            'PairOfBlocks' is created and stored in the 'list_of_pairs_of_blocks' list.  
            """
            list_of_pairs_of_blocks = []  # List[PairOfBlocks]
            for block_index in self._cascade_blocks_generator(single_block_size=size):
                """
                alice_block = {}  # a dictionary for a single block for Alice
                bob_block = {}  # a dictionary for a single block for Bob

                for index in block_index:  # I add proper bits to these dictionaries
                    alice_block[str(index)] = self.sender_cascade[str(index)]
                    bob_block[str(index)] = self.receiver_cascade[str(index)]
                """

                current_block = PairOfBlocks(size=size)
                for index in block_index:
                    current_block.add_bits(
                        index=index,
                        sender_bit=self.sender_cascade[index],
                        receiver_bit=self.receiver_cascade[index]
                    )
                list_of_pairs_of_blocks.append(current_block)  # TODO: should it be a dict or DataFrame?

            """The most important part of CASCADE is remembering all the blocks from all the algorithm's passes:"""
            self.sets_of_blocks[self.current_pass_no] = list_of_pairs_of_blocks
            self.report += f"Bits are assigned to blocks in CASCADE pass {self.current_pass_no}\n"

            """Lists for parity checks between blocks of bits are created independently of these blocks being stored in
            pairs in the 'PairOfBlocks' class' instances:"""
            sender_pass_parity_list = []
            receiver_pass_parity_list = []

            # alice_blocks = []
            # bob_blocks = []

            """Now, binary is performed on each pair of blocks. If there have already been any cascade passes, the
            corrected bit will be updated in all previous blocks & binary will be run on them. The number of 'reviews'
            is closely followed and stored in the raport string.
            """
            for block_number in range(pass_number_of_blocks):  # TODO: majority of this loop (until if self.current_pass_no > 0:) should be put inside the 'PairOfBlocks' class
                """Parity of bits in the current pair of blocks is computed. Results are remembered."""
                sender_parity, receiver_parity = list_of_pairs_of_blocks[block_number].parity_check()
                sender_pass_parity_list.append(sender_parity)
                receiver_pass_parity_list.append(receiver_parity)

                """Next, a parity check is performed - if failed, BINARY is run on this pair of blocks.
                If parities of given blocks are different for Alice (sender) and Bob (receiver), Bob must have an odd 
                number of errors (in protocols like BB84 for sure - otherwise its arbitrary which communicating party is 
                assumed to have errors, and which the correct bits). Obviously we assume that it is the receiver, Bob, 
                who should change his bits to have them identical to Alice's. 
                
                The errors are searched for and corrected (if possible) with BINARY, implemented as a method of the 
                'PairOfBlocks' class. This means that after running BINARY on a given pair of blocks, a single error is 
                already corrected, and so what the method returns only needs to be updated in general statistics and 
                the final key. 
                """
                if sender_parity != receiver_parity:
                    self.report += (f"Different parity in block {block_number} out of {pass_number_of_blocks} blocks"
                                    f"in CASCADE pass {self.current_pass_no}\n")
                    binary_results = list_of_pairs_of_blocks[block_number].binary()
                    binary_correct_bit_value = binary_results.get('Correct bit value')
                    binary_correct_bit_index = binary_results.get('Corrected bit index')

                    """Firstly we add the number of exchanged bits during this BINARY performance to the general number
                    of bits exchanged via the public channel.
                    """
                    self.exchanged_bits_counter += binary_results.get('Bit counter')

                    """Secondly we change main dictionary with final results"""
                    self.receiver_cascade[binary_correct_bit_index] = binary_correct_bit_value
                    # receiver_blocks[block_number][binary_correct_bit_index] = binary_correct_bit_value
                    self.corrected_bits_history[binary_correct_bit_index] = binary_correct_bit_value

                    """Thirdly we change the error bit in blocks' history_cascade:"""
                    if self.current_pass_no > 0:  # in the first pass of CASCADE there are no previous blocks
                        self.report += (f"As it is CASCADE pass {self.current_pass_no}, previous blocks are searched "
                                        f"for the bit to be corrected\n")
                        for previous_pass_index in range(self.current_pass_no):  # we check all previous passes
                            # TODO: here we need to change working on previous blocks into PairOfBlocks implementation with binary as it's method
                            """
                            previous_pass_blocks_sender = self.history_cascade[previous_pass_index].get('Alice blocks')
                            previous_pass_blocks_receiver = self.history_cascade[previous_pass_index].get('Bob blocks')
                            """
                            for n_block in range(len(previous_pass_blocks_receiver)):  # rekurencja na 99%
                                """We check all Bob's blocks in each previous pass"""
                                if binary_correct_bit_index in previous_pass_blocks_receiver[n_block]:
                                    previous_pass_blocks_receiver[n_block][
                                        binary_correct_bit_index] = binary_correct_bit_value
                                    try:
                                        binary_previous = binary(
                                            sender_block=previous_pass_blocks_sender[n_block],
                                            receiver_block=previous_pass_blocks_receiver[n_block],
                                            indexes=list(previous_pass_blocks_sender[n_block].keys())
                                        )

                                        self.exchanged_bits_counter += binary_previous.get('Bit counter')
                                        self.receiver_cascade[
                                            binary_previous['Corrected bit index']] = binary_previous.get(
                                            'Correct bit value')
                                        receiver_blocks[block_number][
                                            binary_previous['Corrected bit index']] = binary_previous.get(
                                            'Correct bit value')
                                    except AttributeError:
                                        var = self.__dict__
                                        raise AttributeError(
                                            f"AttributeError for binary_previous. Current field values are: {var}")
                                    except KeyError:
                                        var = self.__dict__
                                        raise KeyError(
                                            f"KeyError for binary_previous. Current field values are: {var}")

            self.history_cascade.append({'Alice blocks': list_of_pairs_of_blocks, 'Bob blocks': receiver_blocks})
            self.current_pass_no += 1

            """For the purposes of optimizing CASCADE we check the error rate after each pass:"""
            alice_key_error_check = ''.join(list(self.sender_cascade.values()))
            bob_key_error_check = ''.join(list(self.receiver_cascade.values()))

            # TODO: can we make it a separate function?
            key_error_rate = 0
            index = 0
            for bit in alice_key_error_check:
                if bit != bob_key_error_check[index]:
                    key_error_rate += 1
                index += 1
            try:
                key_error_rate = key_error_rate / len(alice_key_error_check)
                self.error_rates.append(key_error_rate)  # its length is equivalent to no. CASCADE passes performed
                if key_error_rate < 0.0001:  # TODO: is 0.01% a small enough number?
                    break  # let's not waste time for more CASCADE passes if there are 'no more' errors
            except ZeroDivisionError:
                error_message = 'ZeroDivisionError with len(alice_key_error_check)'
                print(error_message)
