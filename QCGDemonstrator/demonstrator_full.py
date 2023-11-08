"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
"""

import ast
import random
import math
import numpy as np
from binary import binary
from scipy.stats import binom
from hashing import sha1

random.seed(a=261135)  # useful for debugging, should be omitted for practical uses

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


def cascade_blocks_sizes(quantum_bit_error_rate, key_length, n_passes=1):
    print('--------------------\nCOMPUTING SIZES OF BLOCKS FOR CASCADE')
    max_expected_value = -1 * math.log(0.5, math.e)
    # best_expected_value = max_expected_value
    best_size = key_length

    for size in range(key_length // 2):  # we need at lest 2 blocks to begin with
        print("Currently the greatest size of block fulfilling both criteria is {}.".format(best_size))
        # Firstly we check condition for expected values
        expected_value = 0

        for j in range(size // 2):
            expected_value += 2 * (j + 1) * numerical_error_prob(n_errors=(j + 1), pass_size=size,
                                                                 qber=quantum_bit_error_rate)

        if expected_value <= max_expected_value:
            first_condition = True
        else:
            first_condition = False

        # Secondly we check condition for probabilities per se
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

    print("Since the best initial size of blocks equals {} and we are ment to perform {} passes of the algorithm,"
          "sizes of blocks for consecutive passes are: {}".format(best_size, n_passes, sizes))

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
    print('----------------------------------------\nPERFORMING NAIVE ERROR ESTIMATION\n')
    alice_published_bits = ''
    alice_sifted_key_after_publication = ''
    bob_published_bits = ''
    bob_sifted_key_after_publication = ''

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
            alice_sifted_key_after_publication += alice_key[index]
            bob_sifted_key_after_publication += bob_key[index]

    print('[Alice] My subset of bits for error estimation is: {}'.format(alice_published_bits))
    print('[Bob] My subset of bits for error estimation is: {}'.format(bob_published_bits))

    try:
        naive_error_estimate = naive_error_estimate / len(alice_published_bits)
    except ZeroDivisionError:
        naive_error_estimate = 0  # this will obviously be false, but easy to notice and work on in genalqkd.py

    """At this point we count the number of bits exchanged between Alice and Bob via the public channel,
    for future estimation of the computational cost."""

    no_published_bits = len(alice_published_bits) + len(bob_published_bits)
    print('The probability of bit publishing was {}. In actuality {} bits were published out of {} bits succesfully '
          'sent & received.'.format(publication_prob_rect, no_published_bits, len(alice_key)))

    results = {
        'error estimator': naive_error_estimate,
        'alice key': alice_sifted_key_after_publication,
        'bob key': bob_sifted_key_after_publication,
        'number of published bits': no_published_bits
    }

    return results


def refined_average_error(rect_prob, rect_pub_prob, diag_pub_prob, alice_key, bob_key, alice_bases, bob_bases):
    """In the refined error analysis for simplicity we DO NOT divide raw keys into two separate strings (by the basis).
    Instead, we create two empty strings - alice_key_after_error_estimation & bob_key_after_error_estimation - into
    which we shall rewrite bits unused for error estimation. As for the others, chosen with probability
    rect_pub_prob & diag_pub_prob, respectively, we count them as 'published' and additionally count an error,
    if they differ from each other. Those counter are:
    rect_pub_counter & diag_pub_counter, rect_error & diag_error.

    The last two will be divided at the end by the first two, respectively, to obtain estimations as ratios.
    """
    print('----------------------------------------\nPERFORMING REFINED ERROR ESTIMATION\n')
    alice_key_after_error_estimation = ''
    bob_key_after_error_estimation = ''
    rect_error = 0
    rect_pub_counter = 0
    diag_error = 0
    diag_pub_counter = 0
    alice_published_bits_rect = ''
    bob_published_bits_rect = ''
    alice_published_bits_diag = ''
    bob_published_bits_diag = ''

    """By the way, we don't have to worry with the conditional probability, because the final formula for the error
    estimation takes that into consideration.
    """
    for index in range(len(alice_key)):
        if alice_bases[index] == bob_bases[index] == '0':  # rectilinear basis
            if random.uniform(0, 1) >= rect_pub_prob:
                alice_key_after_error_estimation += alice_key[index]
                bob_key_after_error_estimation += bob_key[index]
            else:
                rect_pub_counter += 1
                if alice_key[index] != bob_key[index]:
                    rect_error += 1
                alice_published_bits_rect += alice_key[index]
                bob_published_bits_rect += bob_key[index]
        else:
            if random.uniform(0, 1) >= diag_pub_prob:
                alice_key_after_error_estimation += alice_key[index]
                bob_key_after_error_estimation += bob_key[index]
            else:
                diag_pub_counter += 1
                if alice_key[index] != bob_key[index]:
                    diag_error += 1
                alice_published_bits_diag += alice_key[index]
                bob_published_bits_diag += bob_key[index]

    """As it is possible to get a VERY small probability of publication, we check for possible divisions by zero:"""
    try:
        rect_error = float(rect_error) / float(rect_pub_counter)
    except ZeroDivisionError:
        rect_error = 0.0

    try:
        diag_error = float(diag_error) / float(diag_pub_counter)
    except ZeroDivisionError:
        diag_error = 0.0

    print('[Alice] My subset of bits measured in the rectilinear basis is: {}'.format(alice_published_bits_rect))
    print('[Bob] My subset of bits measured in the rectilinear basis is: {}'.format(bob_published_bits_rect))
    print('The error rate estimate for the rectilinear basis is {}.\n'.format(rect_error))

    print('[Alice] My subset of bits measured in the diagonal basis is: {}'.format(alice_published_bits_diag))
    print('[Bob] My subset of bits measured in the diagonal basis is: {}'.format(bob_published_bits_diag))
    print('The error rate estimate for the diagonal basis is {}.\n'.format(diag_error))

    """Now, given that measurements in the rectilinear basis were not necessarily with the same probability 
    as those in the diagonal basis, we need a more complicated formula for the 'average error estimate' 
    (Lo, Chau, Ardehali, 2004).
    """
    p = rect_prob  # just a reminder that it's the probability of choosing rect. basis for measurements
    e1 = rect_error
    e2 = diag_error

    e = (p ** 2 * e1 + (1 - p) ** 2 * e2) / (p ** 2 + (1 - p) ** 2)
    print('The average error estimate is {}.'.format(e))

    results = {
        'error estimator': e,
        'alice key': alice_key_after_error_estimation,
        'bob key': bob_key_after_error_estimation,
        'number of published bits': rect_pub_counter + diag_pub_counter
    }

    return results


"""Start of program"""
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
            print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
            continue
        else:
            break

    print('Please set fiber loss:')
    while True:
        try:
            alpha_f = float(input())
        except TypeError:
            print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
            continue
        else:
            break

    print('Please set detection efficiency:')
    while True:
        try:
            eta_det = float(input())
        except TypeError:
            print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
            continue
        else:
            break

    print('Please set a factor accounting for the reduction of the photon detection rate due to the dead time:')
    while True:
        try:
            k = float(input())
        except TypeError:
            print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
            continue
        else:
            break

    print('Please set the additional loss of the system:')
    while True:
        try:
            alpha_e = float(input())
        except TypeError:
            print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
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
        print('Please enter a numerical value between 0 and 1, e.g. 0.95:')
        continue
    else:
        break

alice_basis = ''
alice_basis_length = len(alice_basis)
rectilinear_basis_prob = 0.5  # default value
print("Would you like to set Alice's basis choices ('yes'/'no')?")
while True:
    answer2 = str(input())
    if answer1 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer2 == 'yes':
    while True:
        try:
            print('Please give a string of 0 and 1 (e.g. 0110010101):')
            alice_basis = str(input())
        except TypeError:
            print('Given value cannot be interpreted as string.')
            continue
        else:
            break
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

    while True:
        print("Would you like to set Alice's probability of choosing a rectilinear basis ('yes'/'no')?")
        answer3 = str(input())
        if answer3 not in ('yes', 'no'):
            print("Sorry, I didn't understand that.")
            continue
        else:
            break
    if answer3 == 'yes':
        print("Please set the Alice's probability of choosing a rectilinear basis:")
        while True:
            try:
                rectilinear_basis_prob = float(input())
            except TypeError:
                print('Please enter a numerical value between 0 and 1, e.g. 0.57:')
                continue
            else:
                break
    else:
        print("Alice will choose basis with equal probability, i.e. 0.5.")
else:
    print('Wrong answer. Bye bye.')
    quit()

"""At this point the quantum channel is set up and we know how Alice prepares the measurements.
It's time to perform sending states by Alice.
"""

if alice_basis_length != 0:  # firstly it was 0, if now it's !=0, then we are supposed to choose randomly her basis
    alice_basis = random_choice(alice_basis_length)
    print('Random choices of basis for Alice are: {}'.format(alice_basis))

alice_bits = ''
alice_bits_length = len(alice_bits)
print("Alice's basis choices string is {}-bit long".format(alice_basis_length),
      "\nWould you like to specify {} bits for Alice to send to Bob ('yes'/'no')?".format(alice_basis_length))
while True:
    answer4 = str(input())
    if answer4 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer4 == 'yes':
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
elif answer4 == 'no':
    print('Alice will choose her bits randomly')

if alice_bits_length == 0:  # if it's 0, then we didn't get the bits form the user
    alice_bits_length = alice_basis_length
    alice_bits = random_choice(alice_bits_length)
    print('Random choices of bits for Alice are: {}'.format(alice_bits))

"""At this point it is impractical to encode states as bits because Bob's measurements results depend on both basis
and bit choice of Alice, but he shouldn't know the first one. Because of that we will now translate Alice's bits
to proper states, changing 0 and 1 into + and - for the diagonal basis, respectively.

Please remember that we do that in order to demonstrate what exactly is going on - in practice a measurement is 
performed, results are remembered and post-processed and they are not being published, in contrary to us printing
all of the information ;)
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
print("Alice's states are: {}, where U stands for unknown states due to incorrect encoding of bits.".format(
    alice_states
))

"""Now that we have Alice's data rate, quantum channel's gain and Alice's states,
we can randomly choose m (Alice's basis choices number) bases for Bob. While he performs his measurements,
a portion of Alice's photons do not reach him due to loss on quantum channel.
"""

bob_basis_list = np.random.binomial(1, 1 - rectilinear_basis_prob, alice_basis_length)
bob_basis = ''
for basis in bob_basis_list:
    if random.uniform(0, 1) <= gain:  # for small numbers of bits quantum gain won't change anything in here
        bob_basis += str(int(basis))
    else:
        bob_basis += 'L'  # L for loss

print("Bob's basis choices are: {}, where L stands for measurement that cannot be done".format(bob_basis),
      "\ndue to quantum channel loss.")

"""There's a probability that due to disturbances in quantum channel or eavesdropping 
some bits change while being sent.
"""

change_probability = 0.1
print(
    "Default probability of change of sent states due to disturbances in quantum channel or eavesdropping is {}".format(
        change_probability),
    "\nDo you want to change it ('yes'/'no')?")
while True:
    answer5 = str(input())
    if answer5 not in ('yes', 'no'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer5 == 'yes':
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

"""Now that we know the probability of change, we can perform such disturbances:"""

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

print("Bob's measurement results are: {}".format(bob_states))

"""Now, having Bob's measurement results, we can translate states into bits."""

bob_bits = ''
for state in bob_states:
    try:
        bob_bits += str(states_mapping[state])  # we want to use indexing to raise errors for unsuccessful measur.
    except KeyError:
        bob_bits += 'E'  # E for error
        continue

print("Bob's bits are: {}".format(bob_bits))

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

"""Now it's Alice's turn to send Bob her bases and to cancel out bits (representing states!) from her string that do not
match both successful Bob's measurement and his usage of the same basis in each case.
"""

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

"""No Bob gets info from Alice about her choices of bases, so that he can omit bits resulting from measurements
when he used different basis than Alice.
"""
bob_sifted_key = ''

index = 0
for basis in bob_indicated_basis:
    if basis == alice_indicated_basis[index]:
        bob_sifted_key += bob_indicated_bits[index]
    index += 1

print("Alice's sifted key is {},\nwhile Bob's sifted key is {}.".format(
    alice_sifted_key, bob_sifted_key
))

"""Sifted keys generally differ from each other due to changes between states sent by Alice and received by Bob.
In order to estimate empirical probability of error occurrence in the sifted keys we can publish parts
of keys, compare them and calculate numbers of errors. Then published parts of key should be deleted, as
they have just been exchanged via the public channel.

Originally I've been using 'normal' error estimation; naive one. Now I want to choose between this and refined one
(Lo, Chau, Ardehali, 2004). For this purpose I defined above two functions: naive_error & refined_average_error.
Therefor, we must choose the method of error estimation.
"""
alice_sifted_key_after_error_estimation = ''
bob_sifted_key_after_error_estimation = ''
qber = None
pub_prob_rect = 0.5  # default probability of publication of bits measured in rectilinear basis for refined error est.
pub_prob_diag = 0.5  # default probability of publication of bits measured in diagonal basis for refined error est.
while True:
    print('Please choose a method of error estimation (naive/refined):')
    answer6 = str(input())
    if answer6 not in ('naive', 'refined'):
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if answer6 == 'naive':
    published_key_length_ratio = 0.2
    print("Default ratio of length of published part of the key to it's full length is 0.2.",
          "\nDo you want to change it ('yes'/'no')?")
    while True:
        answer7 = str(input())
        if answer7 not in ('yes', 'no'):
            print("Sorry, I didn't understand that.")
            continue
        else:
            break

    if answer7 == 'yes':
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
    naive_results = naive_error(
        alice_key=alice_bits,
        bob_key=bob_bits,
        publication_prob_rect=published_key_length_ratio)
    qber = naive_results.get('error estimation')
    alice_sifted_key_after_error_estimation = naive_results.get('alice key')
    bob_sifted_key_after_error_estimation = naive_results.get('bob key')
else:  # refined error estimation
    while True:
        try:
            print("Please set the probability of bits' publication in the rectilinear basis:")
            pub_prob_rect = float(input())
        except TypeError:
            print("Please enter a numerical value between 0 and 1, e.g. 0.57.")
            continue
        else:
            break
    while True:
        try:
            print("Please set the probability of bits' publication in the diagonal basis:")
            pub_prob_diag = float(input())
        except TypeError:
            print("Please enter a numerical value between 0 and 1, e.g. 0.57.")
            continue
        else:
            break
    """Now that we have the inputs for the refined error estimation, we perform it with a function defined above."""
    refined_results = refined_average_error(
        rect_prob=rectilinear_basis_prob,
        rect_pub_prob=pub_prob_rect,
        diag_pub_prob=pub_prob_diag,
        alice_key=alice_bits,
        bob_key=bob_bits,
        alice_bases=alice_basis,
        bob_bases=bob_basis
    )
    qber = refined_results.get('error estimation')
    alice_sifted_key_after_error_estimation = refined_results.get('alice key')
    bob_sifted_key_after_error_estimation = refined_results.get('bob key')

"""Once we have an estimation of the QBER we can verify if it's not too great for us to perform error correction."""
if qber >= 0.11:
    print("The estimate of empirical error probability is equal to or greater then 11% threshold.",
          "\nAs the best error correction code approaches a maximal tolerated error rate of 12.9%",
          "\nAlice and Bob should restart the whole procedure on another quantum channel.")

"""Assuming the estimate of empirical probability of error is reasonably small we can continue
with the error correction. Naturally for BB84 we assume it's Bob's key that's flawed.

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

blocks_sizes = cascade_blocks_sizes(quantum_bit_error_rate=qber, key_length=n)
bob_corrected_bits = {}

"""In order to return to blocks from earlier passes of CASCADE we need a history of blocks with indexes and bits,
so implemented by dictionaries as list elements per pass, nested in general history list. Moreover, we create lists
and variables to remember error rates of bits after each consecutive pass of CASCADE and to count the bits exchanged.
"""
history = []
error_rates = []
pass_number = 0
exchanged_bits_counter = 0

for size in blocks_sizes:
    print("--------------------\nThis is CASCADE pass number {}".format(pass_number))
    try:
        pass_number_of_blocks = int(
            -1 * np.floor(-1 * n // size))  # I calculate how many blocks are in total in this pass
    except ZeroDivisionError:
        print("Initial block size equal to 0, please check why.")
        quit()

    print("With {} bits in a single block and {} bits in total we have {} blocks in this CASCADE pass".format(
        size, n, pass_number_of_blocks
    ))

    alice_pass_parity_list = []
    bob_pass_parity_list = []
    alice_blocks = []
    bob_blocks = []

    for block_index in cascade_blocks_generator(string_length=n, blocks_size=size):
        """We sample bits from the raw key (after the error estimation phase) into blocks, with the generator
        defined above.
        """
        alice_block = {}  # a dictionary for a single block for Alice
        bob_block = {}  # a dictionary for a single block for Bob

        for index in block_index:  # I add proper bits to these dictionaries
            alice_block[str(index)] = alice_cascade[str(index)]
            bob_block[str(index)] = bob_cascade[str(index)]

        """I append single blocks created for given indexes to lists of block for this particular CASCADE's pass"""
        alice_blocks.append(alice_block)
        bob_blocks.append(bob_block)

    for i in range(pass_number_of_blocks):
        print("->\nThis is block number {} out of {} in CASCADE pass number {}".format(
            i, pass_number_of_blocks, pass_number))

        current_indexes = list(alice_blocks[i].keys())  # same as Bob's
        print("Indexes of bits from the raw key which are sampled for the current block in CASCADE are: {}".format(
            current_indexes
        ))

        alice_current_bits = list(alice_blocks[i].values())
        print("Alice's bits with these indexes are: {}".format(alice_current_bits))

        bob_current_bits = list(bob_blocks[i].values())
        print("Bob's bits with these indexes are: {}".format(bob_current_bits))

        alice_bit_values = []
        bob_bit_values = []

        for j in range(len(current_indexes)):
            alice_bit_values.append(int(alice_current_bits[j]))
            bob_bit_values.append(int(bob_current_bits[j]))

        alice_pass_parity_list.append(sum(alice_bit_values) % 2)
        print("[Alice] Bits in my block have parity {}".format(alice_pass_parity_list[-1]))

        bob_pass_parity_list.append(sum(bob_bit_values) % 2)
        print("[Bob] Bits in my block have parity {}".format(bob_pass_parity_list[-1]))

        if alice_pass_parity_list[i] != bob_pass_parity_list[i]:  # we check if we should perform BINARY
            print("[Alice] We have different parities. Let's perform binary search for the erroneous bit.")

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
                print("[Alice] Since this CASCADE pass number {} (with 0 being the first one) we should find blocks "
                      "with the corrected bit from previous passes")
                for n_pass in range(pass_number):  # we check all previous passes
                    for n_block in range(
                            len(history[0][n_pass][1])):  # we check all Bob's blocks in each previous pass
                        if binary_results[1] in history[n_pass][1][n_block]:
                            history[n_pass][1][n_block] = binary_results[0]

                            print("[Bob] I found a previous block. Let's go!")
                            print("[Alice] Mee too.")

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
                                print("AttributeError for binary_previous")

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

    """For the purposes of demonstration we check the error rate after each pass:"""
    alice_key_error_check = ''.join(list(alice_cascade.values()))
    bob_key_error_check = ''.join(list(bob_cascade.values()))

    key_error_rate = 0
    index = 0
    for bit in alice_key_error_check:
        if bit != bob_key_error_check[index]:
            key_error_rate += 1
        index += 1
    try:
        key_error_rate = key_error_rate / len(alice_key_error_check)
        error_rates.append(key_error_rate)  # its length is equivalent to no. CASCADE passes performed
    except ZeroDivisionError:
        print("ZeroDivisionError in key error rate calculation.")
        exit()
    print("After this CASCADE pass the ACTUAL error rate is equal to {}.".format(key_error_rate))

"""Time to create strings from cascade dictionaries into corrected keys"""
alice_correct_key = ''.join(list(alice_cascade.values()))
bob_correct_key = ''.join(list(bob_cascade.values()))

print("Alice's correct key:", "\n{}".format(alice_correct_key))
print("Bob's key after performing CASCADE error correction:", "\n{}".format(bob_correct_key))
print("Number of bits exchanged during error correction: {}".format(exchanged_bits_counter))

print("History:", "\n{}".format(history))

"""Finally we perform privacy amplification. In order to to that efficiently we need an estimate of how many bits
of the current key (after sifting & the error correction) does Eve know.

For now, let's use a naive estimator, i.e. number of bits of information (parity etc.) exchanged by Alice and Bob
while performing BINARY.
"""

alice_digest = sha1(alice_correct_key)
print("Alice's correct key's digest is: {}".format(alice_digest))

bob_digest = sha1(bob_correct_key)
print("Bob's correct key's digest is: {}".format(bob_digest))
