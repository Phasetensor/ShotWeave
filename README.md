# ShotWeave

ShotWeave is a PhaseTensor Python library for allocating a fixed measurement budget across circuits. The first experiment estimates a signed weighted sum of independent binary outcome probabilities.

## Quick start

From the repository root, run with Python 3.9 or later. The library and benchmark use the standard library.

```python
from shotweave import allocate, adaptive_allocation

counts = allocate(10_000, [1, 2, 4, 1])
# After 64 pilot shots per circuit, allocate the remaining 9,744 shots.
remaining = adaptive_allocation(10_000, [1, -2, 4, 1], [32, 16, 63, 45], 64)
```

## Installation and measurement data

Install from the repository root with `python -m pip install .`. No runtime dependencies are required.

A backend receives a tuple of requested shot counts and returns one integer count of outcome 1 per circuit in the same order. This example uses a seeded local sampler. Replace the callback with your own batch submission function to use other measurement data.

```python
from random import Random
from shotweave.experiment import run_experiment

rng = Random(42)
probabilities = (0.3, 0.8)

def submit(shots):
    return [sum(rng.random() < p for _ in range(n))
            for p, n in zip(probabilities, shots)]

result = run_experiment(10_000, [1, -2], submit, strategy="adaptive")
print(result.value, result.standard_error, result.total_shots)
```

For previously collected independent final counts, use the CLI:

```sh
echo '{"coefficients":[1,-2],"shots":[100,100],"successes":[30,80]}' | shotweave
```

The target is the weighted sum of probabilities of outcome 1. For a Pauli measurement with outcome 0 mapped to +1, the expectation is 1 − 2p. Each circuit must supply independent shots with a stable probability. The reported standard error uses sample variances, can be zero for boundary counts, and is unavailable when a nonzero term has only one final shot. It excludes systematic device error. A backend exception propagates to the caller without an automatic retry.

## Reproduce the benchmarks

```sh
python -m unittest -v
python benchmark.py
```

Each of 200 seeded trials spends 10,000 shots on four simulated Bernoulli circuits. Uniform allocation splits shots equally. Weighted allocation uses absolute observable coefficients. Adaptive allocation spends 256 shots in a pilot batch and allocates its second batch in proportion to each absolute coefficient times its estimated standard deviation. Half-count smoothing keeps the estimate positive at both boundaries. Only the independent second batch contributes to the adaptive point estimate.

The script reports root mean squared error against the exact weighted sum, simulated batch calls per trial, and elapsed local time across all trials. A batch can contain different shot counts per circuit. Hardware support for this batching assumption and queue latency require separate evaluation.

### Reference error by scenario

The simulator knows the true probabilities. For a fixed allocation, the exact mean squared error is the sum of c_i² p_i (1 − p_i) / n_i over circuits. For adaptive allocation, we average this conditional variance across the 200 pilot allocations before taking the square root. Independent final measurements make the conditional estimate unbiased.

| Scenario | Uniform | Weighted | Adaptive |
| --- | ---: | ---: | ---: |
| Original | 0.023396 | 0.026449 | 0.022751 |
| Equal variance | 0.046904 | 0.040000 | 0.040523 |
| Rare outcomes | 0.020721 | 0.021192 | 0.015505 |

With equal circuit variances, coefficient weighting gives the lowest reference error. Empirical rankings from 200 trials can differ from the reference ranking.

## Status

Version 0.1.0, licensed under Apache 2.0. The benchmarks cover three synthetic scenarios. No hardware or PyPI release has been validated.

## Qiskit V2 sampler

On Python 3.12 install with `python -m pip install '.[qiskit]'` and run
`python qiskit_example.py`. Pass bound circuits with final measurements into a
one-bit classical register (default `meas`) to `SamplerAdapter(sampler, circuits)`.
The adapter sends a single list of (circuit, bindings, shots) tuples per call.
Its jobs list records the sampler job IDs and requested circuit shots. The
example uses StatevectorSampler locally and a seeded NumPy generator shared
across calls. Provider-side scheduling and hardware execution are outside this example.

## Pilot policy and confidence intervals

Pass `pilot_size="auto"` to run_experiment for a budget-aware heuristic.
Given b = floor(budget / circuits), the pilot per circuit is
max(1, min(floor(sqrt(b)), floor(b / 10))). Adaptive mode needs at least two
shots per circuit. The fixed default of 64 preserves existing calls.

Use `confidence_interval(result, coefficients, level=0.95)` from
shotweave.experiment with the original coefficients. The interval uses the
two-sided Hoeffding bound with radius sqrt(log(2 / alpha) sum(c_i²/n_i) / 2),
clipped to the attainable range. The final outcomes are independent
conditional on the pilot. This gives at least the requested coverage under
the independent, stationary Bernoulli model, including boundary outcomes.
Intervals can be conservative. The normal standard error remains available.

## Extended benchmark

Run `python3.12 benchmark_grid.py > results.csv` to reproduce
[the reference table](benchmark_grid.csv). It includes 300 seeded trials per
feasible cell, two budgets, three circuit counts, two probability profiles and
four policies. Reference errors average the conditional variance over pilots.
Coverage is the observed fraction of intervals containing the exact target.
Time columns model 10 microseconds per shot plus either zero or one second per
job. The original benchmark separately measures local wall-clock time.

With 10,000 shots across 100 circuits in the rare-outcome scenario, automatic
allocation leaves 9,000 final shots and gives reference RMSE 0.0045562.
A 64-shot pilot leaves 3,600 and gives 0.0063765. Uniform allocation gives
0.0051802 with one job. For equal variances at the same budget and size,
coefficient weighting gives 0.0100003 with one job, versus 0.0105714 for auto.

## IBM Quantum hardware and public distribution

Install the Qiskit extra and qiskit-ibm-runtime in Python 3.12. With an
authenticated IBM Quantum account run
`python hardware_example.py --instance YOUR_INSTANCE_CRN --output hardware-run.jsonl`.
This submits jobs on the selected instance. The new JSON Lines file records
backend, UTC start, job IDs, per-circuit shot counts, observed successes and
wall time including queueing. Keep the report when diagnosing a failed run.

For the first PyPI publication, configure a pending Trusted Publisher for
project `shotweave`, owner `Phasetensor`, repository `ShotWeave`, workflow
`publish.yml`, environment `pypi`. Dispatch the Publish ShotWeave to PyPI
workflow from main. It builds and validates both distributions, uploads them
using GitHub OIDC and checks installation from the public index.
