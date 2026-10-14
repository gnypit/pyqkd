"""Configuration contract checks, including real spawned pool workers."""
import multiprocessing
import pickle
import unittest

from genal.arguments import GAArguments, as_ga_arguments
from genal.crossover_operators import single_point_crossover, uniform_crossover, plco
from genal.genetic_algorithm import GeneticAlgorithm
from genal.genome_generators import uniform_gene_generator
from genal.interfaces import GAArgumentsInterface
from genal.selection_operators import tournament_selection, ranking_selection


def fitness(genome):
    return sum(genome)


def custom_mutation(genome, args: GAArgumentsInterface):
    return [value + args.mutation["increment"] for value in genome]


class ArgumentsTests(unittest.TestCase):
    def test_mapping_adapter_preserves_extras_and_normalizes_mutation(self):
        original = {"genome": ([1], 2), "mutation": "gene", "extra": 42}
        args = as_ga_arguments(original)
        original["extra"] = 0
        self.assertEqual(args["extra"], 42)
        self.assertEqual(args.mutation, {"type": "gene"})
        self.assertIs(as_ga_arguments(args), args)
        with self.assertRaises(TypeError):
            args["extra"] = 12

    def test_invalid_configuration_is_rejected(self):
        for values in ({"genome": ([1], 0)}, {"mutation": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GAArguments(values)
        for values in ({"selection": []}, {"crossover": "bad"}, {"mutation": 1}):
            with self.subTest(values=values), self.assertRaises(TypeError):
                GAArguments(values)

    def test_pickle_and_generator_modes(self):
        args = pickle.loads(pickle.dumps(GAArguments({"genome": ({0: [1], 1: [2]}, 2)})))
        self.assertIsInstance(args, GAArgumentsInterface)
        self.assertEqual(uniform_gene_generator(args), [1, 2])
        self.assertEqual(uniform_gene_generator(args, gene_position=1), 2)

    def test_named_and_legacy_crossover_settings(self):
        parents = ([1, 2, 3, 4], [5, 6, 7, 8])
        named = GAArguments({"crossover": {"point": 0}})
        legacy = GAArguments({"crossover": [0]})
        expected = [[1, 6, 7, 8], [5, 2, 3, 4]]
        self.assertEqual(single_point_crossover(*parents, named), expected)
        self.assertEqual(single_point_crossover(*parents, legacy), expected)
        self.assertEqual(uniform_crossover(*parents, GAArguments({"crossover": {"choice_prob": 1}})), list(parents))
        self.assertEqual(plco(*parents, GAArguments({"crossover": {"transit_point": 2, "alfa": 1, "beta": 0}})), list(parents))

    @staticmethod
    def create_ga(mode, selection=tournament_selection):
        return GeneticAlgorithm(
            initial_pop_size=4, number_of_generations=1, elite_size=0,
            args={"genome": ([1], 2), "selection": {"pool size": 1},
                  "crossover": {"point": 0}, "mutation": {"type": "custom", "increment": 2}},
            fitness_function=fitness, genome_generator=uniform_gene_generator,
            selection=selection, crossover=single_point_crossover,
            custom_mutation_operator=custom_mutation,
            parallel_workers=2, creation_parallelism=mode,
        )

    def test_same_contract_in_local_and_spawned_creation(self):
        for mode in ("local", "operators", "parent_pairs"):
            with self.subTest(mode=mode):
                ga = self.create_ga(mode)
                ga._create_initial_generation()
                with multiprocessing.get_context("spawn").Pool(
                    2, initializer=ga._initialize_parallel_worker,
                    initargs=(ga.fit_fun, ga.operators, ga.args, custom_mutation),
                ) as pool:
                    members = ga._create_rival_members(pool, [0], 100, 4, 2)[0]
                    self.assertEqual([member.genome for member in members], [[1, 1]] * 4)
                    from genal.parallelisation import _mutate_and_evaluate_batch
                    result = pool.map(_mutate_and_evaluate_batch, [[(0, 100, [1, 1], True)]])[0]
                    self.assertEqual(result, [(0, 100, [3, 3], 6)])

    def test_selection_without_settings_accepts_common_contract(self):
        ga = self.create_ga("local", selection=ranking_selection)
        ga._create_initial_generation()
        pairs = ga._select_parent_genome_pairs(0, ga.current_generation, ga.operators, ga.args)
        self.assertEqual(len(pairs), 2)

    def test_complete_run_with_custom_mutation_configuration(self):
        for mode in ("local", "operators", "parent_pairs"):
            with self.subTest(mode=mode):
                ga = self.create_ga(mode)
                ga.mutation_prob = 1.0
                ga.run()
                self.assertEqual(ga.best_fit_history, [2, 6])
                self.assertEqual([member.genome for member in ga.current_generation.members], [[3, 3]] * 4)


if __name__ == "__main__":
    unittest.main()
