"""Adapter for unparameterized circuits and Qiskit V2 samplers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Submission:
    job_id: str
    shots: tuple


class SamplerAdapter:
    """Read a single named one-bit classical register per circuit."""
    def __init__(self, sampler, circuits, register="meas"):
        self.sampler = sampler
        self.circuits = tuple(circuits)
        self.register = register
        self.jobs = []
        if not self.circuits:
            raise ValueError("Provide at least one circuit")
        for circuit in self.circuits:
            if circuit.num_parameters or not any(
                reg.name == register and reg.size == 1 for reg in circuit.cregs
            ):
                raise ValueError("Bind parameters and provide a one-bit output register")

    def __call__(self, shots):
        shots = tuple(shots)
        if len(shots) != len(self.circuits) or any(type(n) is not int or n < 1 for n in shots):
            raise ValueError("Expected one positive shot count per circuit")
        pubs = [(circuit, None, n) for circuit, n in zip(self.circuits, shots)]
        job = self.sampler.run(pubs)
        self.jobs.append(Submission(job.job_id(), shots))
        results = list(job.result())
        if len(results) != len(shots):
            raise ValueError("Sampler returned a different number of circuit results")
        successes = []
        for result, expected in zip(results, shots):
            counts = result.data[self.register].get_counts()
            if set(counts) - {"0", "1"} or sum(counts.values()) != expected:
                raise ValueError("Sampler returned unexpected outcomes or shot total")
            successes.append(counts.get("1", 0))
        return successes
