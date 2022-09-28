28 IX 2022:
Currently there's a bb84 protocol implemented with CASCADE error correction and naive error rate estimator; bb84.py is the main file, the others contain hashing functions and some functions for CASCADE.

This main file works as an interactive console application for simulating QKD. It lets User define quantum gain of the quantum channel, basis and bits choices for Alice and basis choices for Bob (including random). User may also define probability of eavesdropping, quantum disturbances etc., ratio of the length of the published part of the key for error estimation to raw key's length - and others. After CASCADE bb84.py performs rudimentary privacy amplification with the SHA-1 hashing function.
