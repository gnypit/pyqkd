class LineageTracker:
    """Compact, append-only genealogy for members of accepted generations."""

    def __init__(self, mode: Literal["parents", "full"], initial_capacity: int = 1024):
        self.mode = mode
        self.size = 0
        self.capacity = max(1, initial_capacity)
        self.member_ids = np.empty(self.capacity, dtype=np.int64)
        self.parent_ids = np.full((self.capacity, 2), -1, dtype=np.int64)
        self.generations = np.empty(self.capacity, dtype=np.int32)
        self.fitness_values = np.empty(self.capacity, dtype=np.float64)
        self.id_to_row: dict[int, int] = {}
        self.genomes: np.ndarray | None = None
        self.serialized_genomes: list[bytes | None] | None = None

    def _ensure_capacity(self, required_size: int) -> None:
        if required_size <= self.capacity:
            return
        new_capacity = max(required_size, self.capacity * 2)

        def grow(array: np.ndarray, shape: tuple[int, ...], fill_value=None) -> np.ndarray:
            grown = np.empty(shape, dtype=array.dtype)
            if fill_value is not None:
                grown.fill(fill_value)
            grown[:self.size] = array[:self.size]
            return grown

        self.member_ids = grow(self.member_ids, (new_capacity,))
        self.parent_ids = grow(self.parent_ids, (new_capacity, 2), -1)
        self.generations = grow(self.generations, (new_capacity,))
        self.fitness_values = grow(self.fitness_values, (new_capacity,))
        if self.genomes is not None:
            self.genomes = grow(self.genomes, (new_capacity, *self.genomes.shape[1:]))
        if self.serialized_genomes is not None:
            self.serialized_genomes.extend([None] * (new_capacity - self.capacity))
        self.capacity = new_capacity

    def _initialize_genome_storage(self, generation: Generation) -> None:
        genomes = [member.genome for member in generation.members]
        dense_genomes = np.asarray(genomes)
        if dense_genomes.dtype != object and dense_genomes.ndim >= 2:
            self.genomes = np.empty((self.capacity, *dense_genomes.shape[1:]), dtype=dense_genomes.dtype)
        else:
            self.serialized_genomes = [None] * self.capacity

    def record_generation(self, generation: Generation, generation_number: int) -> None:
        """Append one accepted generation without retaining its Python object graph."""
        required_size = self.size + generation.size
        self._ensure_capacity(required_size)
        if self.mode == "full" and self.genomes is None and self.serialized_genomes is None:
            self._initialize_genome_storage(generation)

        for member in generation.members:
            if member.id in self.id_to_row:
                raise ValueError(f"Member ID {member.id} has already been recorded in the lineage.")
            row = self.size
            self.id_to_row[member.id] = row
            self.member_ids[row] = member.id
            self.parent_ids[row] = member.parent_ids if member.parent_ids else (-1, -1)
            self.generations[row] = generation_number
            self.fitness_values[row] = member.fit_val
            if self.mode == "full":
                if self.genomes is not None:
                    try:
                        self.genomes[row] = member.genome
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Full lineage requires consistent numeric genome shapes and dtypes within a run."
                        ) from error
                else:
                    self.serialized_genomes[row] = pickle.dumps(member.genome, protocol=pickle.HIGHEST_PROTOCOL)
            self.size += 1

    def get_record(self, member_id: int) -> dict:
        """Return lineage metadata for one recorded member."""
        row = self.id_to_row[member_id]
        return {
            "member_id": int(self.member_ids[row]),
            "parent_ids": [int(parent_id) for parent_id in self.parent_ids[row] if parent_id >= 0],
            "generation": int(self.generations[row]),
            "fitness": float(self.fitness_values[row]),
        }

    def get_genome(self, member_id: int):
        """Return an independent ancestral genome copy in full-tracking mode."""
        if self.mode != "full":
            raise RuntimeError("Genomes are available only when lineage_tracking='full'.")
        row = self.id_to_row[member_id]
        if self.genomes is not None:
            return self.genomes[row].copy()
        return pickle.loads(self.serialized_genomes[row])

    def ancestors(self, member_id: int, include_member: bool = True) -> set[int]:
        """Return all recorded ancestors reachable from a member's two parent links."""
        if member_id not in self.id_to_row:
            raise KeyError(member_id)
        discovered = set()
        pending = [member_id]
        while pending:
            current_id = pending.pop()
            if current_id in discovered:
                continue
            discovered.add(current_id)
            row = self.id_to_row.get(current_id)
            if row is None:
                continue
            pending.extend(int(parent_id) for parent_id in self.parent_ids[row] if parent_id >= 0)
        if not include_member:
            discovered.discard(member_id)
        return discovered
