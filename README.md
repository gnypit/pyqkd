**Welcome to the PyQKD package for Quantum Key Distribution protocols simulation and optimisation with Parallel Genetic Algorithms.**

This repository was created as a side result of various research projects conducted at the International Centre for Theory of Quantum Technologies. The problem which led to writing a universal framework of a parallel genetic algorithm was a multi-variable optimisation problem of the QKD in quantum cryptography. Thus, some files for simulating BB84 protocol can be found in here, including an computationally optimised approach to the post-processing of data, especially the error correction stage. However, the main scope of this project now and in the coming weeks is creating a `pyqkd.genal` package which will provide Users with a framework for applying a Parallel Genetic Algorithm to any problem they may have.

I'm happy to collaborate with more people, so please feel free to join!
Cheers,
Jakub

## Genetic algorithm operator arguments

`genal.interfaces.GAArgumentsInterface` is the common configuration contract for
genome generation, selection, crossover, and custom mutation. `GeneticAlgorithm`
accepts a mapping or an implementation of this interface. Mapping input is adapted
once to `genal.arguments.GAArguments` and installed once in each pool worker.

```python
from genal.arguments import GAArguments
from genal.interfaces import GAArgumentsInterface

args = GAArguments({
    "genome": ([0, 1], 20),
    "selection": {"pool size": 3},
    "crossover": {"point": 9},
    "mutation": {"type": "custom", "increment": 1},
})

def custom_mutation(genome, args: GAArgumentsInterface):
    return [gene + args.mutation["increment"] for gene in genome]
```

Every operator receives the **complete** arguments object. Selection reads
`args.selection`, crossover reads `args.crossover`, generators read `args.genome`,
and mutation reads `args.mutation`. Extra custom keys remain available through
mapping access. `args.mutation_type` defaults to `"member"`; supported modes are
`"member"`, `"gene"`, `"gaussian"`, `"swap"`, and `"custom"`. Mutation probability
remains the separate `GeneticAlgorithm.mutation_prob` constructor option.

Built-in crossover settings are `point` for single-point crossover, `choice_prob`
for uniform crossover, and `transit_point`, `alfa`, `beta` for PLCO. Omitted named
settings use the existing defaults. A `None` crossover section and legacy
positional crossover sequences remain accepted, as do mutation strings such as
`"gene"` (normalized to `{"type": "gene"}`). Sections are shared among the configured
operators; independent per-operator configurations are not implemented.

Custom selection and crossover functions must migrate from section-only arguments
to the complete interface: replace selection's `args["pool size"]` with
`args.selection["pool size"]`, and crossover's `args[0]` with a named setting or
`args.crossover_parameter("point", 0, None)`. All selection operators accept
`(generation, args)`; crossovers accept `(parent1_genome, parent2_genome, args)`.
Built-in crossover functions now expect the interface even when called directly.

The outer configuration mapping is read-only through its public API and is
picklable. Nested configuration objects are **not deeply immutable** and must be
treated as read-only by operators. Interface implementations and configuration
values used with multiprocessing must be picklable. General section validation
occurs at adaptation; individual operators own validation of their specific
settings.
