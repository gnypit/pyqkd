import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import genetic_algorithm


class SynchronousPool:
    """Small Pool stand-in used to count evaluations without cross-process state."""

    def __init__(self, processes, initializer, initargs):
        self.processes = processes
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    @staticmethod
    def map(function, inputs, chunksize=1):
        return [function(item) for item in inputs]

    @staticmethod
    def starmap(function, inputs):
        return [function(*item) for item in inputs]


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


def expensive_custom_mutation(genome, args):
    increment = args["custom increment"]
    return [gene + increment for gene in genome]


class ParallelGeneticAlgorithmTests(unittest.TestCase):
    @staticmethod
    def create_ga(
            parallel_workers=2,
            creation_parallelism="auto",
            mutation_prob=0.0,
            number_of_generations=1,
            snapshot_interval=None,
    ):
        return genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=number_of_generations,
            elite_size=0,
            args={"mutation": "gene"},
            fitness_function=sometimes_failing_fitness,
            genome_generator=fixed_genome,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            parallel_workers=parallel_workers,
            creation_parallelism=creation_parallelism,
            mutation_prob=mutation_prob,
            snapshot_interval=snapshot_interval,
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

    def test_uniform_generator_supports_full_genome_and_single_gene_modes(self):
        args = {"genome": ([0, 1, 2], 4)}

        genome = genetic_algorithm.uniform_gene_generator(args)
        gene = genetic_algorithm.uniform_gene_generator(args, gene_position=2)

        self.assertEqual(len(genome), 4)
        self.assertTrue(all(value in args["genome"][0] for value in genome))
        self.assertIn(gene, args["genome"][0])

    def test_uniform_generator_supports_position_specific_spaces(self):
        args = {"genome": ({0: [10], 1: [20], 2: [30]}, 3)}

        self.assertEqual(genetic_algorithm.uniform_gene_generator(args), [10, 20, 30])
        self.assertEqual(genetic_algorithm.uniform_gene_generator(args, gene_position=1), 20)

        with self.assertRaises(IndexError):
            genetic_algorithm.uniform_gene_generator(args, gene_position=3)

    def test_gene_mutation_uses_single_gene_generator_mode(self):
        generator_calls = []

        def generator(_args, gene_position=None):
            generator_calls.append(gene_position)
            return [0, 0] if gene_position is None else 1

        ga = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=0,
            elite_size=0,
            args={},
            fitness_function=sum,
            genome_generator=generator,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            mutation_prob=1.0,
        )
        ga._create_initial_generation()
        generator_calls.clear()

        affected_indexes = ga._mutate_genes()

        self.assertEqual(affected_indexes, [0, 1, 2, 3])
        self.assertEqual(sorted(generator_calls), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual([member.genome for member in ga.current_generation.members], [[1, 1]] * 4)

    def test_gene_mutation_calls_legacy_generator_once_per_affected_member(self):
        generator_calls = []

        def legacy_generator(_args):
            generator_calls.append(1)
            return [2, 2]

        ga = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=0,
            elite_size=0,
            args={},
            fitness_function=sum,
            genome_generator=legacy_generator,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            mutation_prob=1.0,
        )
        ga._create_initial_generation()
        generator_calls.clear()

        affected_indexes = ga._mutate_genes()

        self.assertEqual(affected_indexes, [0, 1, 2, 3])
        self.assertEqual(len(generator_calls), 4)
        self.assertEqual([member.genome for member in ga.current_generation.members], [[2, 2]] * 4)

    def test_invalid_creation_parallelism_is_rejected(self):
        with self.assertRaises(ValueError):
            self.create_ga(creation_parallelism="threads")

    def test_invalid_snapshot_interval_is_rejected(self):
        for value in (True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.create_ga(snapshot_interval=value)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.create_ga(snapshot_interval=value)

    def test_default_retains_fitness_history_without_population_snapshots(self):
        ga = self.create_ga(number_of_generations=3)
        ga.run()

        self.assertEqual(len(ga.best_fit_history), 4)
        self.assertEqual(ga.generation_snapshots, {})

    def test_snapshot_interval_retains_only_requested_generations(self):
        ga = self.create_ga(number_of_generations=3, snapshot_interval=2)
        ga.run()

        self.assertEqual(len(ga.best_fit_history), 4)
        self.assertEqual(list(ga.generation_snapshots), [0, 2])
        self.assertIsNot(ga.generation_snapshots[0], ga.generation_snapshots[2])
        self.assertIsNot(ga.generation_snapshots[2], ga.current_generation)

    def test_auto_creation_uses_parent_pair_batches_for_one_operator_combination(self):
        ga = self.create_ga(creation_parallelism="auto")
        self.assertEqual(ga._resolve_creation_parallelism(no_workers=2), "parent_pairs")
        self.assertEqual(ga._resolve_creation_parallelism(no_workers=1), "local")

    def test_parallel_fitness_failure_matches_serial_zero_fallback(self):
        ga = self.create_ga()
        ga.run()

        self.assertEqual([member.fit_val for member in ga.current_generation.members], [0.0, 2, 0.0, 2])
        self.assertEqual(ga.best_solution()[1], 2)

    def test_post_mutation_fitness_is_updated_by_worker_pool(self):
        ga = self.create_ga(mutation_prob=1.0)
        ga.run()

        self.assertEqual([member.genome for member in ga.current_generation.members], [[1, 1]] * 4)
        self.assertEqual([member.fit_val for member in ga.current_generation.members], [2] * 4)

    def test_single_rival_is_evaluated_only_after_mutation(self):
        fitness_calls = []

        def counting_fitness(genome):
            fitness_calls.append(tuple(genome))
            return sum(genome)

        ga = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=1,
            elite_size=0,
            args={"mutation": "gene"},
            fitness_function=counting_fitness,
            genome_generator=fixed_genome,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            mutation_prob=1.0,
            parallel_workers=2,
        )

        with patch.object(genetic_algorithm, "Pool", SynchronousPool):
            ga.run()

        # Four initial members plus four post-mutation children; no four-child pre-mutation pass.
        self.assertEqual(len(fitness_calls), 8)
        self.assertEqual(fitness_calls[-4:], [(1, 1)] * 4)

    def test_expensive_custom_mutation_is_fused_with_worker_evaluation(self):
        ga = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=1,
            elite_size=0,
            args={"mutation": "custom", "custom increment": 10},
            fitness_function=sum,
            genome_generator=fixed_genome,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            mutation_prob=1.0,
            parallel_workers=2,
            custom_mutation_operator=expensive_custom_mutation,
        )
        ga.run()

        self.assertEqual(
            [member.genome for member in ga.current_generation.members],
            [[9, 11], [11, 11]] * 2,
        )
        self.assertEqual([member.fit_val for member in ga.current_generation.members], [20, 22] * 2)

    def test_custom_mutation_requires_an_operator(self):
        ga = genetic_algorithm.GeneticAlgorithm(
            initial_pop_size=4,
            number_of_generations=1,
            elite_size=0,
            args={"mutation": "custom"},
            fitness_function=sum,
            genome_generator=fixed_genome,
            selection=deterministic_selection,
            crossover=crossover_with_failing_child,
            mutation_prob=1.0,
            parallel_workers=2,
        )

        with self.assertRaises(ValueError):
            ga.run()

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
