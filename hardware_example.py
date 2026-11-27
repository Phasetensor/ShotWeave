"""Two IBM jobs for an adaptive three-angle RY experiment."""
import argparse
from datetime import datetime, timezone
import json
from math import gcd
from pathlib import Path
from time import monotonic

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler import generate_preset_pass_manager

from shotweave import adaptive_allocation
from shotweave.experiment import summarize

ANGLES = (0, 0.7, 2.3)
COEFFICIENTS = (1, -2, 4)
PILOT = (57, 57, 57)
BUDGET = 10000


def make_pubs(circuit, shots):
    # Every binding gets the same shot count. Repeat each angle the exact
    # number of times needed to realize its allocation in one sampler job.
    common = gcd(*shots)
    return [(circuit, np.full((n // common, 1), angle), common)
            for angle, n in zip(ANGLES, shots)]


def read_counts(job, shots):
    results = list(job.result())
    if len(results) != len(shots):
        raise ValueError("Unexpected number of PUB results")
    successes = []
    for result, n in zip(results, shots):
        counts = result.data.meas.get_counts()  # pool over this angle's bindings
        if sum(counts.values()) != n or set(counts) - {"0", "1"}:
            raise ValueError("Unexpected binary outcomes or shot count")
        successes.append(counts.get("1", 0))
    return successes


def execute(service, path):
    from fcntl import flock, LOCK_EX, LOCK_NB
    from qiskit_ibm_runtime import SamplerV2

    with Path(path).open("a+", encoding="utf8") as out:
        flock(out, LOCK_EX | LOCK_NB)
        out.seek(0)
        rows = [json.loads(line) for line in out if line.strip()]
        def record(**fields):
            print(json.dumps(fields), file=out, flush=True)

        if any("total_shots" in row for row in rows):
            print("Esperimento già completato.", flush=True)
            return
        if rows:
            header = rows[0]
            if (header["budget"], tuple(header["coefficients"]), tuple(header["angles"])) != (
                BUDGET, COEFFICIENTS, ANGLES
            ):
                raise ValueError("Report belongs to another experiment")
            backend = service.backend(header["backend"])
        else:
            backend = service.least_busy(operational=True, simulator=False)
            record(backend=backend.name, started_utc=datetime.now(timezone.utc).isoformat(),
                   budget=BUDGET, coefficients=COEFFICIENTS, angles=ANGLES)
        circuit = QuantumCircuit(1)
        circuit.ry(Parameter("theta"), 0)
        circuit.measure_all()
        compiled = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                                seed_transpiler=47).run(circuit)
        sampler = SamplerV2(mode=backend)
        jobs = []
        started = monotonic()
        def measure(stage, shots):
            saved = [row for row in rows if "job_id" in row and "shots" in row
                     and row.get("stage", "pilot") == stage]
            if saved:
                if len(saved) != 1 or tuple(saved[0]["shots"]) != tuple(shots):
                    raise ValueError("Saved job allocation differs")
                job = service.job(saved[0]["job_id"])
            else:
                if any(row.get("submitting") == stage for row in rows):
                    raise RuntimeError("Submission uncertain: reconcile IBM workloads before retry")
                pubs = make_pubs(compiled, shots)
                record(submitting=stage, shots=shots)
                job = sampler.run(pubs)
                record(stage=stage, job_id=job.job_id(), shots=shots)
            print(stage, job.job_id(), flush=True)
            jobs.append(job)
            return read_counts(job, shots)

        pilot = measure("pilot", PILOT)
        final = adaptive_allocation(BUDGET, COEFFICIENTS, pilot, PILOT[0])
        counts = measure("final", final)
        result = summarize(COEFFICIENTS, final, counts)
        if len({job.job_id() for job in jobs}) != 2 or sum(PILOT) + sum(final) != BUDGET:
            raise RuntimeError("Job or shot accounting mismatch")
        usage = [job.metrics().get("usage") for job in jobs]
        record(total_shots=BUDGET, pilot_successes=pilot, final_shots=final,
               successes=counts, value=result.value, standard_error=result.standard_error,
               usage=usage, elapsed_this_invocation_s=monotonic()-started)
        print("Report completato:", path, flush=True)


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(QiskitRuntimeService(channel="ibm_quantum_platform", instance=args.instance), args.output)
