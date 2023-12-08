23 XI 2023:
This month I added all my codes used for my QKD optimisation research project. I reorganised them, having a prototype of a window app for the Demonstrator and my parallel genetic algorithm for the optimisation purposes. I'm testing the latter on a simple labyrinth.

24 VII 2023:
I've uploaded a Python notebook (.ipynb) with a full BB84 protocol: measurements of states, simulating disturbances and losses in quantum channel, performing naive or refined error estimation, CASCADE error correction algorithm and privacy amplification with hashing functions.

User can define gain of the quantum channel, the length of Alice's basis choicec, probabilities of using rectilinear (and so, diagonal) basis, probability of disturbances in the quantum channel, probability of publication of bits for both error estimation methods and number of CASCADE passes.

28 IX 2022:
Currently there's a bb84 protocol implemented with CASCADE error correction and naive error rate estimator; bb84.py is the main file, the others contain hashing functions and some functions for CASCADE.

This main file works as an interactive console application for simulating QKD. It lets User define quantum gain of the quantum channel, basis and bits choices for Alice and basis choices for Bob (including random). User may also define probability of eavesdropping, quantum disturbances etc., ratio of the length of the published part of the key for error estimation to raw key's length - and others. After CASCADE bb84.py performs rudimentary privacy amplification with the SHA-1 hashing function.
