import unittest
from math import sqrt

from shotweave import allocate
from shotweave.experiment import run_experiment, summarize


class ExperimentTests(unittest.TestCase):
    def test_estimate_and_variance(self):
        result = summarize([2, -1], [4, 2], [2, 2])
        self.assertEqual(result.value, 0)
        self.assertAlmostEqual(result.standard_error, sqrt(1/3))

    def test_single_shot_uncertainty(self):
        self.assertIsNone(summarize([1], [1], [0]).standard_error)

    def test_pilot_excluded_from_estimate(self):
        answers = iter([(0, 0), (40, 40)])
        result = run_experiment(100, [1, 1], lambda shots: next(answers), pilot_size=10)
        self.assertEqual(result.value, 2)
        self.assertEqual(result.total_shots, 100)
        self.assertEqual(result.batches, 2)
        self.assertEqual(result.pilot_successes, (0, 0))

    def test_invalid_budget_before_submission(self):
        with self.assertRaises(ValueError):
            run_experiment(10, [1], self.fail)

    def test_invalid_backend_counts(self):
        for counts in ([], [65], [1.0]):
            with self.assertRaises(ValueError):
                run_experiment(100, [1], lambda shots: counts)

    def test_static_batches(self):
        for strategy in ("weighted", "uniform"):
            result = run_experiment(100, [1, -2], lambda shots: shots, strategy)
            self.assertEqual((result.batches, result.total_shots, result.value), (1, 100, -1))

    def test_integer_budget_precision(self):
        budget = 2**60 + 7
        self.assertEqual(sum(allocate(budget, [1, 2, 3])), budget)
