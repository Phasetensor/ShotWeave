import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("qiskit"), "install the qiskit extra")
class AdapterIntegration(unittest.TestCase):
    def test_real_jobs_and_counts(self):
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler
        from shotweave.experiment import run_experiment
        from shotweave.qiskit import SamplerAdapter

        zero = QuantumCircuit(1)
        zero.measure_all()
        one = QuantumCircuit(1)
        one.x(0)
        one.measure_all()
        adapter = SamplerAdapter(StatevectorSampler(), [zero, one])
        result = run_experiment(10000, [1, -3], adapter, pilot_size="auto")
        self.assertEqual(result.successes, (0, result.final_shots[1]))
        self.assertEqual(result.pilot_successes, (0, result.pilot_shots[1]))
        self.assertEqual([j.shots for j in adapter.jobs], [result.pilot_shots, result.final_shots])
        self.assertNotEqual(*result.final_shots)
        self.assertEqual(len({j.job_id for j in adapter.jobs}), 2)
        self.assertEqual(result.total_shots, 10000)
