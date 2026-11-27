"""Matched-block Bernoulli benchmark. Run with Python 3.9+."""
import csv
from math import fsum, sqrt
from random import Random
from statistics import mean, stdev
import sys

from shotweave.design import Posterior, bayesian_allocation, robust_allocation
from shotweave.experiment import confidence_interval, run_experiment, summarize


def compare(blocks=150, budget=320, seed=47):
    coefficients = (1, -2, 4)
    methods = ("uniform", "weighted", "adaptive", "adaptive-optimal", "bayesian", "robust")
    for scenario, probabilities in (("equal", (.5,.5,.5)), ("rare", (.01,.2,.98))):
        truth = fsum(c*p for c,p in zip(coefficients,probabilities))
        metrics = {key: [] for key in methods}
        for block in range(blocks):
            streams = []
            for stage in range(2):
                rng = Random(seed+2*block+stage)
                streams.append([[rng.random()<p for _ in range(budget)] for p in probabilities])
            for method in methods:
                adaptive = method in ("adaptive", "adaptive-optimal", "bayesian")
                stage = 0 if adaptive else 1
                def submit(shots):
                    nonlocal stage
                    counts = [sum(row[:n]) for row,n in zip(streams[stage],shots)]
                    stage += 1
                    return counts
                if method == "bayesian":
                    pilot = (8,)*3
                    post = [Posterior(.5,.5).update(k,8) for k in submit(pilot)]
                    plan = bayesian_allocation(budget-sum(pilot),coefficients,post)
                    result = summarize(coefficients,plan,submit(plan))
                elif method == "robust":
                    plan = robust_allocation(budget,coefficients,[(0,.25)]*3)
                    result = summarize(coefficients,plan,submit(plan))
                else:
                    result = run_experiment(budget,coefficients,submit,method,8)
                lo,hi = confidence_interval(result,coefficients)
                metrics[method].append((result.value-truth, float(lo<=truth<=hi)))
        for method,records in metrics.items():
            errors,covered = zip(*records)
            squared = [e*e for e in errors]
            rmse = sqrt(mean(squared))
            yield [scenario,method,blocks,budget,mean(errors),stdev(errors)/sqrt(blocks),
                   rmse,stdev(squared)/(2*rmse*sqrt(blocks)) if rmse else 0,mean(covered)]


if __name__ == "__main__":
    writer = csv.writer(sys.stdout,lineterminator="\n")
    writer.writerow("scenario method blocks budget bias bias_mcse rmse rmse_mcse coverage".split())
    writer.writerows(compare())
