"""Central configuration adapter for genetic algorithm operators."""
from collections.abc import Mapping, Sequence

try:
    from .interfaces import GAArgumentsInterface
except ImportError:  # Support existing standalone genal scripts.
    from interfaces import GAArgumentsInterface


class GAArguments(GAArgumentsInterface):
    """Picklable, read-only mapping with named operator sections.

    Copies the outer mapping, preserving custom keys and the existing genome
    tuple format. Nested values remain shared: operators must treat them as
    configuration and must not mutate them. An omitted genome is allowed for
    custom generators which do not use a gene space.
    """

    def __init__(self, values: Mapping[str, object]):
        self._values = dict(values)
        selection = self._values.get("selection", {})
        if not isinstance(selection, Mapping):
            raise TypeError("args.selection must be a mapping.")
        crossover = self._values.get("crossover")
        if crossover is not None and (
                not isinstance(crossover, (Mapping, Sequence))
                or isinstance(crossover, (str, bytes))):
            raise TypeError("args.crossover must be a mapping, a positional sequence, or None.")
        mutation = self._values.get("mutation")
        if mutation is None:
            mutation = {}
        elif isinstance(mutation, str):
            mutation = {"type": mutation}
        elif not isinstance(mutation, Mapping):
            raise TypeError("args.mutation must be a settings mapping, a mode string, or None.")
        if mutation.get("type", "member") not in ("gene", "member", "gaussian", "swap", "custom"):
            raise ValueError("args.mutation type must be gene, member, gaussian, swap, or custom.")
        self._values["mutation"] = dict(mutation)
        if "genome" in self._values:
            genome = self._values["genome"]
            if not isinstance(genome, (tuple, list)) or len(genome) != 2:
                raise TypeError("args.genome must contain (gene_space, genome_length).")
            length = genome[1]
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError("args.genome length must be a positive integer.")
            self._values["genome"] = tuple(genome)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    @property
    def genome(self) -> tuple[object, int]:
        return self._values["genome"]

    @property
    def selection(self) -> Mapping[str, object]:
        return self._values.get("selection", {})

    @property
    def crossover(self) -> Mapping[str, object] | Sequence | None:
        return self._values.get("crossover")

    @property
    def mutation(self) -> Mapping[str, object]:
        return self._values["mutation"]


def as_ga_arguments(args: GAArgumentsInterface | Mapping[str, object]) -> GAArgumentsInterface:
    """Adapt dictionary input once; preserve custom interface implementations."""
    if isinstance(args, GAArgumentsInterface):
        return args
    if not isinstance(args, Mapping):
        raise TypeError("args must implement GAArgumentsInterface or be a mapping.")
    return GAArguments(args)
