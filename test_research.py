import unittest
from fractions import Fraction
from math import comb
from random import Random

from shotweave.experiment import automatic_pilot, confidence_interval, run_experiment, summarize


class PolicyAndIntervalTests(unittest.TestCase):
    def test_pilot_limits(self):
        for size in (1, 4, 100):
            for budget in (2*size, 17*size, 10000):
                k = automatic_pilot(budget, size)
                self.assertLessEqual(k, max(1, budget//(10*size)))
                self.assertGreaterEqual(budget-k*size, size)
        with self.assertRaises(ValueError):
            automatic_pilot(7, 4)

    def test_boundary_signed(self):
        result = summarize([1, -2], [100, 100], [0, 100])
        lo, hi = confidence_interval(result, [1, -2])
        self.assertEqual(lo, -2)
        self.assertGreater(hi, -2)
        with self.assertRaises(ValueError):
            confidence_interval(result, [1, -2], 1)

    def test_exact_coverage(self):
        for p in (Fraction(1, 100), Fraction(1, 2), Fraction(9, 10)):
            mass = Fraction(0)
            for k in range(21):
                result = summarize([-2], [20], [k])
                lo, hi = confidence_interval(result, [-2])
                if lo <= -2*p <= hi:
                    mass += comb(20, k)*p**k*(1-p)**(20-k)
            self.assertGreaterEqual(mass, Fraction(95, 100))

    def test_seeded_coverage(self):
        for method in ("uniform", "weighted", "adaptive"):
            hits = 0
            for seed in range(250):
                rng = Random(seed)
                def submit(shots):
                    return [sum(rng.random() < p for _ in range(n))
                            for p, n in zip((0.02, 0.6), shots)]
                result = run_experiment(200, [1, -2], submit, method, "auto")
                lo, hi = confidence_interval(result, [1, -2])
                hits += lo <= -1.18 <= hi
            self.assertGreaterEqual(hits, 230)
