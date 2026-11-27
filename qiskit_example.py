"""Run after installing ShotWeave with the qiskit extra."""
from numpy.random import default_rng
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from shotweave.experiment import run_experiment
from shotweave.qiskit import SamplerAdapter


def example():
    circuits = []
    for angle in (0, 0.7, 2.3):
        circuit = QuantumCircuit(1)
        circuit.ry(angle, 0)
        circuit.measure_all()
        circuits.append(circuit)
    adapter = SamplerAdapter(StatevectorSampler(seed=default_rng(47)), circuits)
    result = run_experiment(10000, [1, -2, 4], adapter, pilot_size="auto")
    return result, adapter.jobs


if __name__ == "__main__":
    result, jobs = example()
    assert result.total_shots == 10000 and len({job.job_id for job in jobs}) == 2
    assert len(set(jobs[1].shots)) > 1 and result.successes[0] == 0
    for job in jobs:
        print(job.job_id, job.shots)
    print(result)
