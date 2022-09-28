"""
Author: Jakub Gnyp; contact: gnyp.jakub@gmail.com, LinkedIn: https://www.linkedin.com/in/gnypit/
Functions and constants based on FIPS PUB 180-4 from:
http://dx.doi.org/10.6028/NIST.FIPS.180-4
"""


def constant_sha1(index):
    if 0 <= index <= 19:
        return 0x5a827999
    elif 20 <= index <= 39:
        return 0x6eD9eba1
    elif 40 <= index <= 59:
        return 0x8f1bbcdc
    elif 60 <= index <= 79:
        return 0xca62c1d6
    else:
        return 'Error: index exceeds allowed range.'


def ch(x, y, z):  # one of internal hashing functions for SHA-1
    return (x & y) ^ (~x & z)


def parity(x, y, z):  # one of internal hashing functions for SHA-1
    return x ^ y ^ z


def maj(x, y, z):  # one of internal hashing functions for SHA-1
    return (x & y) ^ (x & z) ^ (y & z)


def hashing_function_sha1(x, y, z, j):
    if 0 <= j <= 19:
        return ch(x=x, y=y, z=z)
    elif 20 <= j <= 39 or 60 <= j <= 79:
        return parity(x=x, y=y, z=z)
    elif 40 <= j <= 59:
        return maj(x=x, y=y, z=z)
    else:
        return 'Error: j exceeds possible range.'


def rotl(n, n_bits, word):  # word will be given as an int, so we will return it as an int too
    """python's left shift doesn't discard any bits, just adds padding - we will discard bits manually"""
    first = bin(word << n)[2 + n:n_bits + n + 2]  # bin changes type to str
    first = int('0b' + first, base=2)

    """python's right shift only discards the bits - we will add the padding manually"""
    second = bin(word >> n_bits - n)[2:n + 2]  # bin changes type to str
    second = '0' * (n_bits - n) + second
    second = int('0b' + second, base=2)

    """Result is a bitwise OR on first and second"""
    result = first | second  # result = (word << n) | (word >> (n_bits - n)) with proper discarding and padding
    return result


def sha1(message):
    # message = str(input('Please give a message:'))  # should be a series of 0's and 1's

    """SHA-1: firstly we parse the message into equally long substrings - blocks:"""
    message_length, blocks_length = len(message), 512
    blocks = [message[i:i + blocks_length] for i in range(0, message_length, blocks_length)]
    if len(blocks[-1]) < blocks_length:  # in case the last block is too short, we add padding with 0's
        padding_length = blocks_length - len(blocks[-1])
        padding = '0' * padding_length
        blocks[-1] = padding + blocks[-1]

    """Now for the actual hash computation. We will use:
    words -> groups 32 bits
    hashing function -> as defined above for SHA-1
    ROTL function -> as defined above
    ...to create a 'message schedule' for every block of parsed message from the list 'blocks':
    """

    """SHA-1: setting initial hash value"""
    hash_value = [
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    ]

    for block in blocks:
        """Firstly we prepare the message schedule:"""
        for t in range(0, 80, 1):
            """First 16 words of the message schedule are words of the block of parsed message which we are working on
            in this iteration of the loop.
            """
            message_schedule = [block[i:i + 32] for i in range(0, 512, 32)]

            """Last 64 words of the message schedule are created based on a recurrent formula using ROTL function:"""
            for i in range(16, 80, 1):
                """We locally represent selected previous words as binary values"""
                schedule_i3 = int('0b' + message_schedule[i - 3], base=2)
                schedule_i8 = int('0b' + message_schedule[i - 8], base=2)
                schedule_i14 = int('0b' + message_schedule[i - 14], base=2)
                schedule_i16 = int('0b' + message_schedule[i - 16], base=2)

                """Next word of the message schedule is value of ROTL on above words:"""
                next_word = bin(rotl(
                    n=1,
                    n_bits=32,
                    word=schedule_i3 ^ schedule_i8 ^ schedule_i14 ^ schedule_i16
                ))[2:32]
                message_schedule.append(next_word)

            """We initialize the five working variables with last (possibly initial) hash value"""
            a = hash_value[0]
            b = hash_value[1]
            c = hash_value[2]
            d = hash_value[3]
            e = hash_value[4]

            """We process the working variables..."""
            for i in range(0, 80, 1):
                t = rotl(n=5, n_bits=32, word=a) + hashing_function_sha1(x=b, y=c, z=d, j=i) + e + constant_sha1(index=i) \
                    + int('0b' + message_schedule[i], base=2)
                e = d
                d = c
                c = rotl(n=30, n_bits=32, word=b)
                b = a
                a = t

            """...and compute the next hash value.
            As we always need just the last hash value, we won't store all of them in memory; instead we will
            save each next one in the list hash_value, which decomposes such a value
            into its' five words - the list elements. 
            """
            hash_value[0] = hash_value[0] + a
            hash_value[1] = hash_value[1] + b
            hash_value[2] = hash_value[2] + c
            hash_value[3] = hash_value[3] + d
            hash_value[4] = hash_value[4] + e

    """After the loop the resulting 160-bit message digest of the original message is:"""
    digest = ''
    for value in hash_value:
        word = bin(value)
        digest += word[2:len(word)]

    return digest
