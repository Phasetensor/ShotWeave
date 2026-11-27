"""Print the benchmark grid as CSV. Reference results use Python 3.12."""
import csv
from math import fsum, sqrt
from random import Random
import sys

from shotweave.experiment import confidence_interval, run_experiment

TRIALS = 300


def simulate(budget, size, regime, method):
    c = tuple((1, -2, 4, 1)[i % 4] / size for i in range(size))
    pattern = (0.5,) if regime == "equal" else (0.01, 0.5, 0.001, 0.95)
    p = tuple(pattern[i % len(pattern)] for i in range(size))
    truth = fsum(w*q for w, q in zip(c, p))
    fixed = method == "fixed64"
    pilot = 64 if fixed else "auto"
    strategy = "adaptive" if method in ("auto", "fixed64") else method
    feasible = not fixed or budget >= 65 * size
    if not feasible:
        return ["infeasible"] + [""] * 8
    variances, errors = [], []
    covered = 0
    for seed in range(TRIALS):
        rng = Random(seed)
        def submit(shots):
            if hasattr(rng, "binomialvariate"):
                return [rng.binomialvariate(n, q) for n, q in zip(shots, p)]
            return [sum(rng.random() < q for _ in range(n)) for n, q in zip(shots, p)]
        result = run_experiment(budget, c, submit, strategy, pilot)
        assert result.total_shots == budget
        lo, hi = confidence_interval(result, c)
        covered += lo <= truth <= hi
        errors.append((result.value - truth)**2)
        variances.append(fsum(w*w*q*(1-q)/n for w, q, n in zip(c, p, result.final_shots)))
    jobs = result.batches
    # Cost model in seconds: 10 microseconds per shot, plus latency per job.
    shot_time = budget * 1e-5
    return ["ok", sum(result.pilot_shots), sum(result.final_shots), jobs,
            round(sqrt(fsum(errors)/TRIALS), 7), round(sqrt(fsum(variances)/TRIALS), 7),
            round(covered/TRIALS, 4), round(shot_time, 5), round(shot_time + jobs, 5)]


if __name__ == "__main__":
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow("budget circuits regime method status pilot final jobs rmse reference coverage95 modeled_s_latency0 modeled_s_latency1".split())
    for budget in (1000, 10000):
        for size in (4, 16, 100):
            for regime in ("equal", "rare"):
                for method in ("uniform", "weighted", "fixed64", "auto"):
                    writer.writerow([budget, size, regime, method] + simulate(budget, size, regime, method))
