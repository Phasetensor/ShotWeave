"""Run with python benchmark.py from the repository root."""
from math import fsum, sqrt
from random import Random
from time import perf_counter

from shotweave import allocate, adaptive_allocation

BUDGET = 10_000
PILOT = 64
TRIALS = 200
COEFFICIENTS = (1, -2, 4, 1)
PROBABILITIES = (0.5, 0.25, 0.99, 0.7)


def submit(rng, counts):
    return [sum(rng.random() < p for _ in range(n))
            for p, n in zip(PROBABILITIES, counts)]


def benchmark(strategy):
    truth = fsum(c * p for c, p in zip(COEFFICIENTS, PROBABILITIES))
    squared_errors = []
    submissions = 0
    start = perf_counter()
    for seed in range(TRIALS):
        rng = Random(seed)
        spent = 0
        if strategy == "adaptive":
            pilot = submit(rng, [PILOT] * len(COEFFICIENTS))
            submissions += 1
            spent = PILOT * len(pilot)
            counts = adaptive_allocation(BUDGET, COEFFICIENTS, pilot, PILOT)
        else:
            weights = [1] * len(COEFFICIENTS) if strategy == "uniform" else list(map(abs, COEFFICIENTS))
            counts = allocate(BUDGET, weights)
        outcomes = submit(rng, counts)
        submissions += 1
        assert spent + sum(counts) == BUDGET
        estimate = fsum(c * k / n for c, k, n in zip(COEFFICIENTS, outcomes, counts))
        squared_errors.append((estimate - truth) ** 2)
    elapsed = perf_counter() - start
    print(f"{strategy:9} RMSE={sqrt(fsum(squared_errors)/TRIALS):.6f} "
          f"batches/trial={submissions/TRIALS:g} total_seconds={elapsed:.3f}")


if __name__ == "__main__":
    for method in ("uniform", "weighted", "adaptive"):
        benchmark(method)
