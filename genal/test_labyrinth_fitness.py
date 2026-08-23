import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import labyrinth_test


class LabyrinthFitnessTests(unittest.TestCase):
    def test_repeated_position_penalty_is_preserved(self):
        route = [2, 1]  # right, then left to the already-visited entrance

        self.assertAlmostEqual(labyrinth_test.fitness_fun_pyqkd(route), 3 / 60)
        self.assertEqual(
            labyrinth_test.fitness_fun_pygad(None, route, 0),
            labyrinth_test.fitness_fun_pyqkd(route),
        )

    def test_wall_penalty_is_preserved(self):
        route = [1]  # left from the entrance into a wall
        self.assertAlmostEqual(labyrinth_test.fitness_fun_pyqkd(route), 2.75 / 60)


if __name__ == "__main__":
    unittest.main()
