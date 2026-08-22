import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import genetic_algorithm


def fixed_genome(_args):
    return [1, 1]


def deterministic_selection(generation, _args):
    return [generation.members[index % generation.size] for index in range(generation.num_parents_pairs * 2)]


def crossover_with_failing_child(parent1, parent2, _args):
    return [-1, parent1[1]], [parent2[0], parent2[1]]


def sometimes_failing_fitness(genome):
    if genome[0] < 0:
        raise RuntimeError("expected fitness failure")
    return sum(genome)


class ParallelGeneticAlgorithmTests(unittest.TestCase):
    @staticmethod
    def create_ga(parallel_workers=2, creation_parallelism="auto"):
        return genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=1,
            elite_size=0,
            args={},
            fitness_function=sometimes_failing_fitness,
            genome_generator=fixed_genome,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            parallel_workers=parallel_workers,
            creation_parallelism=creation_parallelism,
        )

    def test_non_positive_parallel_worker_count_is_rejected(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.create_ga(parallel_workers=value)

    def test_non_integer_parallel_worker_count_is_rejected(self):
        for value in (True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.create_ga(parallel_workers=value)

    def test_fitness_batches_limit_task_count_and_preserve_members(self):
        members = {member_id: [member_id] for member_id in range(100)}
        batches = genetic_algorithm.GeneticAlgorithm._create_fitness_batches(members, no_workers=2)

        self.assertLessEqual(len(batches), 8)
        self.assertEqual(list(members.items()), [job for batch in batches for job in batch])

    def test_invalid_creation_parallelism_is_rejected(self):
        with self.assertRaises(ValueError):
            self.create_ga(creation_parallelism="threads")

    def test_auto_creation_is_local_for_one_operator_combination(self):
        ga = self.create_ga(creation_parallelism="auto")
        self.assertEqual(ga._resolve_creation_parallelism(no_workers=2), "local")

    def test_parallel_fitness_failure_matches_serial_zero_fallback(self):
        ga = self.create_ga()
        ga.run()

        self.assertEqual([member.fit_val for member in ga.current_generation.members], [0.0, 2, 0.0, 2])
        self.assertEqual(ga.best_solution()[1], 2)

    def test_operator_creation_mode_remains_available(self):
        ga = self.create_ga(creation_parallelism="operators")
        ga.run()

        self.assertEqual([member.genome for member in ga.current_generation.members], [[-1, 1], [1, 1]] * 2)

    def test_parent_pair_creation_preserves_member_order_and_ids(self):
        ga = self.create_ga(creation_parallelism="parent_pairs")
        ga.run()

        member_ids = [member.id for member in ga.current_generation.members]
        self.assertEqual(member_ids, list(range(member_ids[0], member_ids[0] + 4)))
        self.assertEqual([member.genome for member in ga.current_generation.members], [[-1, 1], [1, 1]] * 2)


if __name__ == "__main__":
    unittest.main()
