"""Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/"""


# TODO: this file will have the chat-like version of the function for the demonstrator; for now the actual method with no printable messages will be put in the cascade.py file as a Block's method


def binary(sender_block, receiver_block, indexes, receiver_name='Bob'):
    is_binary = True
    sender_current_block = sender_block
    receiver_current_block = receiver_block

    """Contrary to real-life applications in this simulation of the BINARY algorithm Alice (sender) and Bob (receiver) 
    do not exchange messages. Instead, we count how many bits should be exchanged between them so that the algorithm
    would end successfully. Afterwards we return this value together with the bit to be changed as a result of the
    algorithm.
    """
    bit_counter = 0

    while is_binary:
        """Sender starts by sending to the Receiver parity of the first half of her string"""
        half_index = len(sender_current_block) // 2  # same as Bob's
        first_half_indexes = indexes[0:half_index:1]  # same as Bob's
        sender_first_half_list = []

        for index in first_half_indexes:
            sender_first_half_list.append(int(sender_current_block[index]))

        sender_first_half_parity = sum(sender_first_half_list) % 2
        # print("[Alice] My string's first half has a parity: {}".format(sender_first_half_parity))
        bit_counter += 1  # At this point sender informs receiver about their 1st half's parity

        """Now Receiver determines whether an odd number of errors occurred in the first or in the
        second half by testing the parity of his string and comparing it to the parity sent
        by Sender
        """

        receiver_first_half_list = []

        for index in first_half_indexes:
            receiver_first_half_list.append(int(receiver_current_block[index]))

        receiver_first_half_parity = sum(receiver_first_half_list) % 2

        """Single (at least) error is in the 'half' of a different parity; we change current strings
        that are analysed into halves of different parities until one bit is left - the error
        """

        if receiver_first_half_parity != sender_first_half_parity:
            """print('[{}] I have an odd number of errors in my first half.'.format(
                receiver_name
            ))
            """
            bit_counter += 1  # At this point receiver would send a message about an odd number of errors in 1st half

            sender_subscription_block = {}
            receiver_subscription_block = {}

            for index in first_half_indexes:
                receiver_subscription_block[index] = receiver_current_block[index]
                sender_subscription_block[index] = sender_current_block[index]

            sender_current_block = sender_subscription_block
            receiver_current_block = receiver_subscription_block

            indexes = list(sender_current_block.keys())  # same as Bob's
        else:
            """print('[{}] I have an odd number of errors in my second half.'.format(
                receiver_name
            ))
            """
            bit_counter += 1  # At this point receiver would send a message about an odd number of errors in 2nd half

            """We have to repeat the whole procedure for the second halves"""
            second_half_indexes = indexes[half_index::1]
            sender_subscription_block = {}
            receiver_subscription_block = {}

            for index in second_half_indexes:
                receiver_subscription_block[index] = receiver_current_block[index]
                sender_subscription_block[index] = sender_current_block[index]

            sender_current_block = sender_subscription_block
            receiver_current_block = receiver_subscription_block

            indexes = list(sender_current_block.keys())  # same as Bob's

        if len(receiver_current_block) == 1:  # at some point this clause will be true
            """print("[{}] I have one bit left, I'm changing it.".format(
                receiver_name
            ))
            """
            bit_counter += 1  # At this point receiver would send a message (?) about one bit left and changing it

            """Firstly we change the error bit in Bob's original dictionary of all bits"""
            if receiver_current_block[indexes[0]] == '0':
                # bob_cascade[indexes[0]] = '1'
                return {'Correct bit value': '1', 'Corrected bit index': indexes[0], 'Bit counter': bit_counter}
            else:
                # bob_cascade[indexes[0]] = '0'
                return {'Correct bit value': '0', 'Corrected bit index': indexes[0], 'Bit counter': bit_counter}

            # Secondly we change the error bit in blocks' history
            # We need to perform BINARY on all blocks which we correct in history list
            # history[number of pass][owner][number of block]

            # is_binary = False  # we break the loop, end of BINARY
