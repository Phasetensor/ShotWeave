"""Run a 10,000-shot experiment on a selected IBM Quantum instance."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from time import monotonic

from shotweave.experiment import run_experiment
from shotweave.qiskit import SamplerAdapter


def run(backend, sampler, output):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager

    compiler = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                           seed_transpiler=47)
    circuits = []
    for angle in (0, 0.7, 2.3):
        circuit = QuantumCircuit(1)
        circuit.ry(angle, 0)
        circuit.measure_all()
        circuits.append(circuit)
    adapter = SamplerAdapter(sampler, compiler.run(circuits))
    def record(**fields):
        print(json.dumps(fields), file=output, flush=True)
    record(backend=backend.name, started_utc=datetime.now(timezone.utc).isoformat(),
           budget=10000, coefficients=[1, -2, 4], angles=[0, 0.7, 2.3])
    start = monotonic()
    def submit(shots):
        before = monotonic()
        count = len(adapter.jobs)
        try:
            return adapter(shots)
        finally:
            if len(adapter.jobs) > count:
                record(**asdict(adapter.jobs[-1]), elapsed_s=monotonic()-before)
    result = run_experiment(10000, [1, -2, 4], submit, pilot_size="auto")
    if len({j.job_id for j in adapter.jobs}) != 2 or result.total_shots != 10000:
        raise RuntimeError("Unexpected job count or shot budget")
    record(**asdict(result), total_shots=result.total_shots, elapsed_s=monotonic()-start)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, help="IBM Quantum instance CRN")
    parser.add_argument("--output", required=True, help="New JSON Lines report")
    args = parser.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=args.instance)
    backend = service.least_busy(operational=True, simulator=False)
    with open(args.output, "x", encoding="utf8") as stream:
        run(backend, SamplerV2(mode=backend), stream)
