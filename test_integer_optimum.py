from fractions import Fraction
from itertools import product
import unittest

from shotweave import optimal_allocation, allocate


class IntegerOptimality(unittest.TestCase):
    def test_against_exhaustive_search(self):
        for weights in ((0, 1, 4), (1, 1, 1), (1, 2, 5)):
            for budget in range(3, 13):
                def objective(ns):
                    return sum(Fraction(w*w, n) for w, n in zip(weights, ns))
                feasible = (ns for ns in product(range(1, budget), repeat=3) if sum(ns) == budget)
                counts = optimal_allocation(budget, weights)
                self.assertEqual(sum(counts), budget)
                self.assertEqual(objective(counts), min(map(objective, feasible)))
                self.assertLessEqual(objective(counts), objective(allocate(budget, weights)))
