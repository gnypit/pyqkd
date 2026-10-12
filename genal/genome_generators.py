def uniform_gene_generator(ga_args: dict, gene_position: int | None = None):
    """Uniformly generate either a complete genome or one gene for mutation.

    Parameters:
        ga_args (dict): This dictionary is stored within the GeneticAlgorithm class and contains info about args to be
            used by either genome generator, crossover operators or selection operators. For the genome generation,
            args are stored under the key ``"genome"`` as ``(gene_space, genome_length)``. ``gene_space`` may be a
            shared sequence for every position or a dictionary mapping each position to its own sequence.
        gene_position (int | None): Position whose replacement gene should be generated. If omitted, a complete genome
            is generated for initial-population creation and whole-member mutation.

    Returns:
        list | object: A complete genome when ``gene_position`` is ``None``; otherwise one gene sampled from the space
            associated with that position.

    Raises:
        IndexError: if ``gene_position`` is outside the configured genome.
    """
    gene_space, length = ga_args["genome"]

    if gene_position is not None:
        if not 0 <= gene_position < length:
            raise IndexError(
                f"gene_position must be between 0 and {length - 1}; got {gene_position}."
            )
        position_space = gene_space[gene_position] if isinstance(gene_space, dict) else gene_space
        return random.choice(position_space)

    if isinstance(gene_space, dict):
        return [random.choice(gene_space[position]) for position in range(length)]
    return list(np.random.choice(gene_space, length))


def uniform_gene_generator_for_position(ga_args: dict, gene_position: int):
    """Uniform gene generator for a single gene mutation operator. Designed for the labyrinth test."""
    gene_space, _ = ga_args["genome"]

    if isinstance(gene_space, dict):
        position_space = gene_space[gene_position]
        return random.choice(position_space)

    return random.choice(gene_space)
