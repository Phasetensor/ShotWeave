"""Matched-block Bernoulli benchmark. Run with Python 3.9+."""
import csv
from math import fsum, sqrt
from random import Random
from statistics import mean, stdev
import sys

from shotweave.measurements import Readout
from shotweave.experiment import confidence_interval, run_experiment


def compare(blocks=150, budget=10000, seed=47):
    coefficients = (1, -2, 4)
    methods = ("uniform", "weighted", "adaptive", "adaptive-optimal", "bayesian", "robust", "optimal", "noise-aware")
    for scenario, probabilities in (("equal", (.5,.5,.5)), ("rare", (.01,.2,.98))):
        models = (Readout(.02,.03),Readout(.12,.07),Readout(.23,.18))
        observed = [m.false_one+(1-m.false_one-m.false_zero)*p for p,m in zip(probabilities,models)]
        truth = fsum(c*p for c,p in zip(coefficients,probabilities))
        metrics = {key: [] for key in methods}
        for block in range(blocks):
            streams = []
            for stage in range(2):
                rng = Random(seed+2*block+stage)
                streams.append([[rng.random()<p for _ in range(budget)] for p in observed])
            for method in methods:
                adaptive = method in ("adaptive", "adaptive-optimal", "bayesian","noise-aware")
                stage = 0 if adaptive else 1
                def submit(shots):
                    nonlocal stage
                    counts = [sum(row[:n]) for row,n in zip(streams[stage],shots)]
                    stage += 1
                    return counts
                result = run_experiment(budget,coefficients,submit,method,"auto",
                                        readout=models)
                assert result.total_shots == budget
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
