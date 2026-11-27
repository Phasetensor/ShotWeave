import itertools
import unittest

from shotweave.design import (
    Posterior, Resources, continuous_cost, integer_cost, pareto, pilot_risks,
    precision_plan, robust_allocation,
)
from shotweave.measurements import Readout, basis, compatible, covariance, group_variance, groups
from shotweave.sequential import confidence_sequence, sequential


class AdvancedTests(unittest.TestCase):
    def test_knapsack_against_enumeration(self):
        for costs in itertools.product((1, 2, 3), repeat=2):
            for weights in [(0, 0), (0, 4), (3, 1), (1, 1)]:
                for budget in range(2*sum(costs), 18):
                    plan = integer_cost(budget, weights, costs, minimum=2)
                    exact = min(sum(w*w/n for w,n in zip(weights,counts))
                                for counts in itertools.product(range(2,budget+1),repeat=2)
                                if sum(k*n for k,n in zip(costs,counts)) <= budget)
                    self.assertAlmostEqual(exact,plan.variance)
                    self.assertEqual(plan.spent+plan.unspent,budget)
                    continuous = continuous_cost(budget,weights,costs,2)
                    self.assertLessEqual(sum(w*w/n for w,n in zip(weights,continuous)),exact+1e-12)

    def test_statistics(self):
        prior = Posterior(1,1)
        self.assertAlmostEqual(prior.parameter_variance,1/12)
        self.assertAlmostEqual(prior.measurement_variance,1/6)
        self.assertEqual(prior.update(2,3),Posterior(3,2))
        self.assertEqual(robust_allocation(6,[1,2],[(0,1),(1,1)]),[2,4])
        self.assertEqual(precision_plan([1],.25,8).shots,(4,))
        self.assertEqual(precision_plan([0],.25,8).variance,0)
        a,b = Resources(1,2,3,1),Resources(2,2,3,1)
        self.assertEqual(pareto([b,a,a]),[a])
        self.assertEqual(pilot_risks(8,[1],[.5],[2],lambda p,n:1,2),{2:.25/6})

    def test_grouped(self):
        self.assertTrue(compatible("XX","ZZ",general=True))
        self.assertEqual(groups(["XI","IX","ZZ"]),[["XI","IX"],["ZZ"]])
        self.assertEqual(basis(["XI","IY"]),"XY")
        with self.assertRaises(ValueError):
            basis(["XX","ZZ"])
        matrix = covariance([[1,-1],[-1,1]])
        self.assertEqual(group_variance([1,1],matrix),0)
        self.assertEqual(group_variance([1,1],covariance([[1,-1],[-1,1]],1)),4)
        for bad in ([[0,1],[1,0]], [[1,2],[2,1]], [[1,1],[0,1]]):
            with self.assertRaises(ValueError):
                group_variance([1,1],bad)
        value,variance = Readout(.1,.2).correct(.45,.049)
        self.assertAlmostEqual(value,.5)
        self.assertAlmostEqual(variance,.1)

    def test_stopping(self):
        self.assertEqual(confidence_sequence([2,-1],[0,0],[0,0]),(-1,2))
        submit = lambda plan: [n//2 for n in plan]
        for jobs,budget,width,expected in [(3,99,0,3),(9,15,0,2),(9,99,4,1)]:
            history = sequential([1,-1],submit,budget,8,jobs,width)
            self.assertEqual(len(history),expected)
            self.assertLessEqual(sum(history[-1].shots),budget)
            for look in history:
                self.assertLessEqual(look.interval[0],0)
                self.assertGreaterEqual(look.interval[1],0)
        with self.assertRaises(ValueError):
            sequential([1],lambda n:[-1],20,2,1,0)

class ApplicationTests(unittest.TestCase):
    def test_energy_and_mitigation(self):
        from shotweave.applications import grouped_energy, extrapolate
        self.assertEqual(grouped_energy(-1, [[1,1]], [[[1,-1],[-1,1]]]),(-1,0))
        self.assertEqual(extrapolate([1,2],[3,4],[.25,1]),(2,2))
        with self.assertRaises(ValueError):
            extrapolate([1,1],[3,4],[.25,1])
        with self.assertRaises(ValueError):
            grouped_energy(0,[[1]],[[[1],[2]]])

    def test_backend_accounting(self):
        from shotweave.applications import aggregate, BackendRecord
        def row(name,job):
            return BackendRecord(name,"cal1","Z",job,4,1,.5)
        self.assertEqual(aggregate([row("a","1"),row("a","2")]),(2,8,1))
        for pair in [[row("a","1"),row("b","2")],[row("a","1")]*2]:
            with self.assertRaises(ValueError):
                aggregate(pair)
