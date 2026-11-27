import unittest

from shotweave import allocate, adaptive_allocation


class AllocationTests(unittest.TestCase):
    def test_equal_weights_and_tie_order(self):
        self.assertEqual(allocate(10, [1, 1, 1]), [4, 3, 3])

    def test_zero_weight_receives_minimum(self):
        self.assertEqual(allocate(10, [0, 1]), [1, 9])

    def test_extreme_pilot_counts(self):
        counts = adaptive_allocation(100, [-2, 2], [0, 10], 10)
        self.assertEqual(counts, [40, 40])

    def test_pilot_budget_exhausted(self):
        with self.assertRaises(ValueError):
            adaptive_allocation(20, [1, 1], [2, 3], 10)

    def test_invalid_weights(self):
        for weights in ([0], [-1], [float("nan")], []):
            with self.assertRaises(ValueError):
                allocate(10, weights)
