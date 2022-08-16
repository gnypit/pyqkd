"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
"""

import random
import math
import numpy as np
from binary import binary
from scipy.stats import binom

random.seed(a=261135)  # useful for debugging, should be omitted for practical uses

# Let's set up the quantum channel (BB84)
# basis_names = ['rectilinear', 'diagonal']
basis_mapping = {'rectilinear': 0, 'diagonal': 1}  # useful
# basis_vectors_mapping = {'0': '0', '90': '1', '+45': '+', '-45': '-'}
# bits_mapping = {'0': 0, '90': 1, '+45': 0, '-45': 1}
states_mapping = {'0': 0, '1': 1, '+': 0, '-': 1}  # useful

quantum_channel = {  # useful
    'rectilinear': {
        'basis_vectors': {'first_state': '0', 'second_state': '1'}
    },
    'diagonal': {
        'basis_vectors': {'first_state': '+', 'second_state': '-'}
    }
}


def qc_gain(mean_photon_number=1., fiber_loss=1., detection_efficiency=1., k_dead=1.,
            additional_loss=1.):  # quantum channel gain
    g = mean_photon_number * fiber_loss * detection_efficiency * k_dead * additional_loss
    return g


def received_key_material(quantum_channel_gain, sender_data_rate):
    receiver = quantum_channel_gain * sender_data_rate
    return receiver


def random_choice(length):
    chosen_basis = ''
    for index in range(length):
        chosen_basis += str(random.randint(0, 1))

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
    best_expected_value = max_expected_value
    best_size = key_length

    for size in range(key_length // 2):  # we need at lest 2 blocks to begin with

        # Firstly we check condition for expected values
        expected_value = 0

        for j in range(size // 2):
            expected_value += 2 * j * numerical_error_prob(n_errors=j, pass_size=size, qber=qber)

        if expected_value <= max_expected_value:
            first_condition = True
        else:
            first_condition = False

        # Secondly we check condition for probabilities per se
        second_condition = False
        for j in range(size // 2):
            prob_sum = 0
            for l in list(np.arange(j+1, size // 2 + 1, 1)):
                prob_sum += numerical_error_prob(n_errors=l, pass_size=size, qber=qber)

            if prob_sum <= numerical_error_prob(n_errors=j, pass_size=size, qber=qber) / 4:
                second_condition = True
            else:
                second_condition = False

        if first_condition and second_condition:
            if expected_value < best_expected_value:
                best_expected_value = expected_value
                best_size = size

    sizes = [best_size]

    for j in range(n_passes):
        sizes.append(2 * sizes[-1])

    return sizes


def cascade_blocks_generator(string_length, blocks_size):
    string_index = list(np.arange(0, string_length, 1))  # I create a list of all indexes (list of ndarray)
    blocks = random.sample(population=string_index, k=string_length)  # I shuffle the list randomly

    for j in range(0, string_length, blocks_size):  # I generate equally long chunks of shuffled indexes
        yield blocks[j:j + blocks_size]


# Start of program
gain = 1
print('Hello. This is a program for QKD. '
      '\nRectilinear basis is encoded by 0, diagonal by 1. States 0, 1 are encoded 0, 1.'
      '\nStates +, - are also encoded 0, 1.'
      '\nWould you like to specify quantum gain of quantum channel used by Alice and Bob ("yes"/"no")?')
while True:
    answer1 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer1 == 'yes':
    print('Please set mean photon number:')
    while True:
        try:
            mu = float(input())
        except TypeError:
            print('Please enter a numerical value:')
            continue
        else:
            break

    print('Please set fiber loss:')
    while True:
        try:
            alpha_f = float(input())
        except TypeError:
            print('Please enter a numerical value:')
            continue
        else:
            break

    print('Please set detection efficiency:')
    while True:
        try:
            eta_det = float(input())
        except TypeError:
            print('Please enter a numerical value:')
            continue
        else:
            break

    print('Please set a factor accounting for the reduction of the photon detection rate due to the dead time:')
    while True:
        try:
            k = float(input())
        except TypeError:
            print('Please enter a numerical value:')
            continue
        else:
            break

    print('Please set the additional loss of the system:')
    while True:
        try:
            alpha_e = float(input())
        except TypeError:
            print('Please enter a numerical value:')
            continue
        else:
            break

    gain = qc_gain(
        mean_photon_number=mu,
        fiber_loss=alpha_f,
        detection_efficiency=eta_det,
        k_dead=k,
        additional_loss=alpha_e
    )
    print('Quantum gain is {}'.format(gain))
elif answer1 == 'no':
    print('Quantum gain is by default set to 1.')
else:
    print('Wrong answer. Bye bye.')
    quit()

print("What is Alice's data rate?")
while True:
    try:
        alice_data_rate = float(input())
    except TypeError:
        print('Please give a numerical value:')
        continue
    else:
        break

alice_basis = ''
alice_basis_length = len(alice_basis)
print("Would you like to set Alice's basis choices ('yes'/'no')?")
while True:
    answer2 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer2 == 'yes':
    print('Please give a string of 0 and 1 (e.g. 0110010101):')
    try:
        alice_basis = str(input())
    except TypeError:
        print('Given value cannot be interpreted as string. Alice will choose her basis randomly.')
        alice_basis = ''
elif answer2 == 'no':
    print('Alice will choose her basis randomly. How long should the string with her choices be?')
    while True:
        try:
            alice_basis_length = int(input())
        except TypeError:
            print('Please give an integer')
            continue
        else:
            break
else:
    print('Wrong answer. Bye bye.')
    quit()

if alice_basis_length != 0:  # firstly it was 0, if now it's !=0, then we are supposed to choose randomly her basis
    alice_basis = random_choice(alice_basis_length)
    print('Random choices of basis for Alice are: {}'.format(alice_basis))

alice_bits = ''
alice_bits_length = len(alice_bits)
print("Alice's basis choices string is {}-bit long".format(alice_basis_length),
      "\nWould you like to specify {} bits for Alice to send to Bob ('yes'/'no')?".format(alice_basis_length))
while True:
    answer3 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer3 == 'yes':
    print('Please give a string of {} bits for Alice to send to Bob (e.g. 0110101)'.format(alice_basis_length))
    while True:
        try:
            alice_bits = str(input())
            alice_bits_length = len(alice_bits)
            if alice_basis_length != alice_bits_length:
                print("Please give as many bits as is the length of Alice's bases choices.")
                continue
            else:
                break
        except TypeError:
            print('Given string is incorrect. Alice will choose her bits randomly')
elif answer3 == 'no':
    print('Alice will choose her bits randomly')

if alice_bits_length == 0:  # if it's 0, then we didn't get the bits form the user
    alice_bits_length = alice_basis_length
    alice_bits = random_choice(alice_bits_length)
    print('Random choices of bits for Alice are: {}'.format(alice_bits))

'''
At this point it is impractical to encode states as bits because Bob's measurements results depend on both basis
and bit choice of Alice, but he shouldn't know the first one. Because of that we will now translate Alice's bits
to proper states, changing 0 and 1 into + and - for the diagonal basis.
'''

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
print("Alice's states are: {}, where U stands for unknown states due to incorrect encoding of bits.".format(
    alice_states
))

'''
Now that we have Alice's data rate, quantum channel's gain and Alice's states,
we can randomly choose m (Alice's basis choices number) bases for Bob. While he performs his measurements,
a portion of Alice's photons do not reach him due to loss on quantum channel. We will reflect that by choosing Bob's
bases inside a for loop instead of using random_choice function defined earlier.
'''

bob_basis = ''
for i in range(alice_basis_length):
    if random.uniform(0, 1) <= gain:  # for small numbers of bits quantum gain won't change anything in here
        bob_basis += str(random.randint(0, 1))
    else:
        bob_basis += 'L'  # L for loss

print("Bob's basis choices are: {}, where L stands for measurement that cannot be done".format(bob_basis),
      "\ndue to quantum channel loss.")

'''
There's a probability that due to disturbances in quantum channel or eavesdropping some bits change while being sent.
'''

change_probability = 0.1
print(
    "Default probability of change of sent states due to disturbances in quantum channel or eavesdropping is {}".format(
        change_probability),
    "\nDo you want to change it ('yes'/'no')?")
while True:
    answer4 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer4 == 'yes':
    print("Please set probability of change as a numerical value between 0 and 1 (e.g. 0.57):")
    while True:
        try:
            change_probability = float(input())
            if 1.0 >= change_probability >= 0.0:
                break
            else:
                print('Please enter a numerical value between 0 and 1:')
                continue
        except TypeError:
            print('Please enter a numerical value between 0 and 1:')
            continue

'''
Now that we know the probability of change, we can perform such disturbances:
'''

received_states = ''
change_states = {'0': '0', '1': '1', '2': '+', '3': '-'}  # dictionary for randomizing changed states
for state in alice_states:  # to be updated following concrete attack strategies
    if random.uniform(0, 1) <= change_probability:
        '''
        if state == '0':
            received_states += '1'
        elif state == '1':
            received_states += '0'
        else:
            received_states += 'C'  # C for change
        '''
        change_indicator = str(random.randint(0, 3))
        while state == change_states.get(change_indicator):  # we repeat as long as it takes to actually change state
            change_indicator = str(random.randint(0, 3))
        received_states += change_states.get(change_indicator)
    else:
        received_states += state

print("States received by Bob before measurement are: {}".format(received_states))

'''
After Bob chose a basis and a photon has reached him, he performs his measurement. If for this particular photon 
he chooses the same basis as Alice chose before, his measurement result will be the same - that's because
in reality Bob is choosing polarizators for photons to go through. If photon's polarization is the same as
polarizators, then there's 100% probability of preserving this polarization. This is reflected by the same bit
as before the measurement. Otherwise polarization after measurement is random. This mechanism is implemented in
"measurement" function.

Results of Bob's measurements are states - either 0 for |0> or 1 for |1> or + for |+> or - for |->. If he couldn't
perform the measurement (encoded by basis L) then the result will be encoded in the string by L as well.
'''

bob_states = ''
for i in range(alice_basis_length):  # it's the same length as of Bob's basis choices
    bob_states += measurement(state=received_states[i], basis=bob_basis[i])

print("Bob's measurement results are: {}".format(bob_states))

'''
Now, having Bob's measurement results, we can translate states into bits.
'''

bob_bits = ''
for state in bob_states:
    try:
        bob_bits += str(states_mapping[state])
    except KeyError:
        bob_bits += 'E'  # E for error
        # bob_bits += str(random.randint(0, 1)) if there's disturbance in quantum channel - or smth like that - we randomize result
        continue

print("Bob's bits are: {}".format(bob_bits))

'''
Alice and Bob have each a string of bits, which will shortly become a key for cipher.
At this point Alice and Bob can switch to communicating on a public channel. Their first step is to 
perform sifting - decide which bits to keep in their key.

Bob begins by telling Alice, which photons he measured. He then tells her which bases were used in each measurement.
'''

bob_measurement_indicators = ''
for bit in bob_bits:  # is it possible to optimise length of such an indicator?
    if bit == '0' or bit == '1':
        bob_measurement_indicators += '1'
    else:
        bob_measurement_indicators += '0'

print("[Bob] My measurement indicator: {}".format(bob_measurement_indicators))

bob_indicated_basis = ''
bob_indicated_bits = ''
index = 0

for indicator in bob_measurement_indicators:  # in optimised approach send only bases for successful measurements
    if indicator == '1':
        bob_indicated_basis += bob_basis[index]
        bob_indicated_bits += bob_bits[index]
    index += 1

print("[Bob] I used bases: {}".format(bob_indicated_basis))

'''
Now it's Alice's turn to send Bob her bases and to cancel out bits (representing states!) from her string that do not
match both successful Bob's measurement and his usage of the same basis in each case.
'''

alice_indicated_bits = ''
alice_indicated_basis = ''
alice_sifted_key = ''

index = 0
for indicator in bob_measurement_indicators:  # this loop ic practically copied - room for optimisation
    if indicator == '1':
        alice_indicated_bits += alice_bits[index]
        alice_indicated_basis += alice_basis[index]
    index += 1

print("[Alice] I used bases: {}".format(alice_indicated_basis))

index = 0
for basis in alice_indicated_basis:
    if basis == bob_indicated_basis[index]:
        alice_sifted_key += alice_indicated_bits[index]
    index += 1

'''
No Bob gets info from Alice about her choices of bases, so that he can omit bits resulting from measurements
when he used different basis than Alice.
'''

bob_sifted_key = ''

index = 0
for basis in bob_indicated_basis:
    if basis == alice_indicated_basis[index]:
        bob_sifted_key += bob_indicated_bits[index]
    index += 1

print("Alice's sifted key is {}, while Bob's sifted key is {}.".format(
    alice_sifted_key, bob_sifted_key
))

'''
Sifted keys generally differ from each other due to changes between states sent by Alice and received by Bob.
In order to estimate empirical probability of error occurrence in the sifted keys we can publish parts
of keys, compare them and calculate numbers of errors. Then published parts of key should be deleted, as
they have just been exchanged via the public channel.

Originally I've been testing algorithm above on short strings, i.e. up to 20 bits in original Alice's message.
Sifted keys were therefore too short for publication of any parts of them for error estimation.

With 20 bits in original message, 0.9 quantum gain and 0.1 probability of change with random seed 251135
Alice gets a sifted key 11001, while Bob gets 11011. If we change original length to a 100, we get 47-bits
long sifted keys with 3 mistakes, ergo exact empirical probability of error is 3/48 = 6.25 %.

A refined estimation is based on two subsets: one of the sifted key and the other on the bits originating from
measurements with different basis (Alice's and Bob's). First we present the 'naive' estimation.

Let's randomly publish subsets of bits of matching positions in the sifted keys strings.
'''

published_key_length_ratio = 0.2
print("Default ratio of length of published part of the key to it's full length is 0.2.",
      "\nDo you want to change it ('yes'/'no')?")
while True:
    answer5 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer5 == 'yes':
    print("Please set ratio of number of published bits of the key to it's full length",
          "\n as a numerical value between 0 and 1, e.g. 0.15:")
    while True:
        try:
            published_key_length_ratio = float(input())
            if 0 <= published_key_length_ratio <= 1:
                break
            else:
                print("Please enter a numerical value between 0 and 1")
                continue
        except TypeError:
            print("Please enter a numerical value between 0 and 1")
            continue

alice_published_bits = ''
# alice_sifted_key_list = list(alice_sifted_key)
alice_sifted_key_after_error_estimation = ''

bob_published_bits = ''
# bob_sifted_key_list = list(bob_sifted_key)
bob_sifted_key_after_error_estimation = ''

naive_error_estimate = 0

for index in range(len(alice_sifted_key)):  # could be bob_sifted_key as well
    if random.uniform(0, 1) <= published_key_length_ratio:  # znowu mam zawyżone prawdopodobieństwo
        # alice_bit = alice_sifted_key_list[index]
        # bob_bit = bob_sifted_key_list[index]
        alice_bit = alice_sifted_key[index]
        bob_bit = bob_sifted_key[index]

        # First we add those bits to strings meant for publication
        alice_published_bits += alice_bit
        bob_published_bits += bob_bit

        # Now for the estimation of the error:
        if alice_bit != bob_bit:
            naive_error_estimate += 1
    else:  # if a bit wasn't published, we reuse it in the sifted key
        alice_sifted_key_after_error_estimation += alice_sifted_key[index]
        bob_sifted_key_after_error_estimation += bob_sifted_key[index]

naive_error_estimate = naive_error_estimate / len(alice_published_bits)

print("[Alice] My subset of bits is: {}.".format(alice_published_bits))
print("[Bob] My subset of bits is: {}.".format(bob_published_bits))
print("Estimate of error empirical probability is: {}".format(naive_error_estimate))

if naive_error_estimate >= 0.11:
    print("The estimate of empirical error probability is equal to or greater then 11% threshold.",
          "\nAs the best error correction code approaches a maximal tolerated error rate of 12.9%",
          "\nAlice and Bob should restart the whole procedure on another quantum channel.")

'''
Assuming the estimate of empirical probability of error is reasonably small we can continue
with the error correction. Naturally we assume it's Bob's key that's flawed.

We begin by checking the parity of Alice's and Bob's sifted keys, shortened by the subsets used for error estimation.
'''

alice_parity_list = []
bob_parity_list = []
for index in range(len(alice_sifted_key_after_error_estimation)):
    alice_parity_list.append(int(alice_sifted_key_after_error_estimation[index]))
    bob_parity_list.append((int(bob_sifted_key_after_error_estimation[index])))

alice_parity = sum(alice_parity_list) % 2
bob_parity = sum(bob_parity_list) % 2

print("[Alice] My parity is {}.".format(alice_parity))
print("[Bob] My parity is {}.".format(bob_parity))

'''
CASCADE: 1st I need to assign bits to their indexes in original strings. Therefore I create dictionaries
for Alice and for Bob.
'''
n = len(alice_sifted_key_after_error_estimation)
alice_cascade = {}
bob_cascade = {}

for i in range(n):  # I dynamically create dictionaries with indexes as keys and bits as values
    alice_cascade[str(i)] = alice_sifted_key_after_error_estimation[i]
    bob_cascade[str(i)] = bob_sifted_key_after_error_estimation[i]

'''
Now we need to set up CASCADE itself: sizes of blocks in each pass, numeration of passes and a distionary
for corrected bits with their indexes from original Bob's string as keys and correct bits as values.
'''

blocks_sizes = cascade_blocks_sizes(qber=naive_error_estimate, key_length=n)
bob_corrected_bits = {}

'''
In order to return to blocks from earlier passes of CASCADE we need a history of blocks with indexes and bits,
so implemented by dictionaries as list elements per pass, nested in general history list:
'''
history = []
pass_number = 0

for size in blocks_sizes:
    pass_number_of_blocks = int(-1 * np.floor(-1 * n // size))  # I calculate how many blocks are in total in this pass

    alice_pass_strings = []
    alice_pass_parity_list = []

    bob_pass_strings = []
    bob_pass_parity_list = []
    bob_pass_corrected_strings = []

    alice_blocks = []
    bob_blocks = []
    for block_index in cascade_blocks_generator(string_length=n, blocks_size=size):
        alice_block = {}  # a dictionary for a single block for Alice
        bob_block = {}  # a dictionary for a single block for Bob

        for index in block_index:  # I add proper bits to these dictionaries
            alice_block[str(index)] = alice_cascade[str(index)]
            bob_block[str(index)] = bob_cascade[str(index)]

        # I append single blocks created for given indexes to lists of block for this particular CASCADE's pass
        alice_blocks.append(alice_block)
        bob_blocks.append(bob_block)

    for i in range(pass_number_of_blocks):

        '''
        alice_pass_strings.append(alice_sifted_key_after_error_estimation[start_index:stop_index:1])
        alice_current_bits = []
        for bit in alice_pass_strings[i]:  # te dwie pętle do połączenia w celu optymalizacji
            alice_current_bits.append(int(bit))
        alice_pass_parity_list.append(sum(alice_current_bits) % 2)

        bob_pass_strings.append(bob_sifted_key_after_error_estimation[start_index:stop_index:1])
        bob_current_bits = []
        for bit in bob_pass_strings[i]:  # te dwie pętle do połączenia w celu optymalizacji
            bob_current_bits.append(int(bit))
        bob_pass_parity_list.append(sum(bob_current_bits) % 2)
        '''

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
            '''
            # We perform BINARY
            is_binary = True
            number_of_runs = 0
            alice_current_string = ''.join(alice_current_bits)
            alice_current_block = alice_blocks[i]
            bob_current_string = ''.join(bob_current_bits)
            bob_current_block = bob_blocks[i]

            while is_binary:
                # Alice starts by sending Bob parity of the first half of her string
                half_index = len(alice_current_block) // 2  # same as Bob's
                first_half_indexes = current_indexes[0:half_index:1]  # same as Bob's
                alice_first_half_list = []

                for index in first_half_indexes:
                    alice_first_half_list.append(int(alice_current_block[index]))

                alice_first_half_parity = sum(alice_first_half_list) % 2
                print("[Alice] My string's first half has a parity: {}".format(alice_first_half_parity))

                # Now Bob determines whether an odd number of errors occurred in the first or in the
                # second half by testing the parity of his string and comparing it to the parity sent
                # by Alice

                bob_first_half_list = []

                for index in first_half_indexes:
                    bob_first_half_list.append(int(bob_current_block[index]))

                bob_first_half_parity = sum(bob_first_half_list) % 2

                # Single (at least) error is in the 'half' of a different parity; we change current strings
                # that are analysed into halves of different parities until one bit is left - the error

                if bob_first_half_parity != alice_first_half_parity:
                    print("[Bob] I have an odd number of errors in my first half.")

                    alice_subscription_block = {}
                    bob_subscription_block = {}

                    for index in first_half_indexes:
                        bob_subscription_block[index] = bob_current_block[index]
                        alice_subscription_block[index] = alice_current_block[index]

                    alice_current_block = alice_subscription_block
                    bob_current_block = bob_subscription_block

                    current_indexes = list(alice_current_block.keys())  # same as Bob's

                    first_half = True
                else:
                    print("[Bob] I have an odd number of errors in my second half.")

                    # We have to repeat the whole procedure for the second halves
                    second_half_indexes = current_indexes[half_index::1]
                    alice_subscription_block = {}
                    bob_subscription_block = {}

                    for index in second_half_indexes:
                        bob_subscription_block[index] = bob_current_block[index]
                        alice_subscription_block[index] = alice_current_block[index]

                    alice_current_block = alice_subscription_block
                    bob_current_block = bob_subscription_block

                    current_indexes = list(alice_current_block.keys())  # same as Bob's

                    first_half = False

                if len(bob_current_block) == 1:  # at some point this clause will be true
                    print("[Bob] I have one bit left, I'm changing it.")

                    # Firstly we change the error bit in Bob's original dictionary of all bits
                    if bob_current_block[current_indexes[0]] == '0':
                        bob_cascade[current_indexes[0]] = '1'
                    else:
                        bob_cascade[current_indexes[0]] = '0'

                    # Secondly we change the error bit in blocks' history
                    # We need to perform BINARY on all blocks which we correct in history list
                    # history[number of pass][owner][number of block]

                    is_binary = False  # we break the loop, end of BINARY
                    '''
            binary_results = binary(
                sender_block=alice_blocks[i],
                receiver_block=bob_blocks[i],
                indexes=current_indexes
            )

            # Firstly we change main dictionary with final results and current blocks for history
            bob_cascade[binary_results[1]] = binary_results[0]
            bob_blocks[i][binary_results[1]] = binary_results[0]

            # Secondly we change the error bit in blocks' history
            # We need to perform BINARY on all blocks which we correct in history list
            # history[number of pass][owner][number of block]
            if pass_number > 0:  # in the first pass of CASCADE there are no previous blocks
                for n_pass in range(pass_number):  # we check all previous passes
                    for n_block in range(len(history[0][n_pass][1])):  # we check all Bob's blocks in each prevoius pass
                        if binary_results[1] in history[n_pass][1][n_block]:
                            history[n_pass][1][n_block] = binary_results[0]
                            binary_previous = binary(
                                sender_block=history[n_pass][0][n_block],
                                receiver_block=history[n_pass][1][n_block],
                                indexes=history[n_pass][1][n_block].keys()
                            )

    history.append([alice_blocks, bob_blocks])
    pass_number += 1

# All that remains is to create strings from cascade dictionaries into corrected keys

alice_correct_key = ''.join(list(alice_cascade.values()))
bob_correct_key = ''.join(list(bob_cascade.values()))

print("Alice's correct key:", "\n{}".format(alice_correct_key))
print("Bob's key after performing CASCADE error correction:", "\n{}".format(bob_correct_key))

print("History:", "\n{}".format(history))

# Finally we perform privacy amplification
