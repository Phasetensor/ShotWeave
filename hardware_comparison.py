"""Randomized hardware blocks; each block uses 30,000 shots and four jobs."""
import argparse
from datetime import datetime, timezone
import json
from math import cos, fsum
from random import Random

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler import generate_preset_pass_manager

from hardware_example import ANGLES, BUDGET, COEFFICIENTS, PILOT, make_pubs, read_counts
from shotweave.experiment import run_experiment


def compare(backend, sampler, blocks, output, seed):
    if blocks < 2:
        raise ValueError("Use at least two blocks for replication")
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    circuit.measure_all()
    compiled = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=47
    ).run(circuit)
    truth = fsum(c * (1-cos(a))/2 for c, a in zip(COEFFICIENTS, ANGLES))
    rng = Random(seed)
    def record(**fields):
        print(json.dumps(fields), file=output, flush=True)
    record(backend=backend.name, blocks=blocks, seed=seed, ideal=truth)
    for block in range(blocks):
        methods = ["uniform", "weighted", "adaptive"]
        rng.shuffle(methods)
        for method in methods:
            def submit(shots):
                job = sampler.run(make_pubs(compiled, shots))
                record(block=block, method=method, job_id=job.job_id(), shots=shots,
                       submitted_utc=datetime.now(timezone.utc).isoformat())
                counts = read_counts(job, shots)
                record(job_id=job.job_id(), counts=counts, usage=job.metrics().get("usage"))
                return counts
            estimate = run_experiment(BUDGET, COEFFICIENTS, submit, method, PILOT[0])
            record(block=block, method=method, estimate=estimate.value,
                   signed_error=estimate.value-truth, shots=estimate.total_shots)


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=args.instance)
    backend = service.backend(args.backend)
    with open(args.output, "x", encoding="utf8") as output:
        compare(backend, SamplerV2(mode=backend), args.blocks, output, args.seed)
