"""Simultaneous coverage across all inspected batches, including optional stopping."""
import csv
from math import sqrt
from random import Random
import sys

from shotweave.sequential import sequential


def compare(paths=500):
    for probabilities in ((.5,.5),(.01,.98),(0.,1.)):
        truth = probabilities[0]-2*probabilities[1]
        hits = 0
        looks = 0
        for seed in range(paths):
            rng = Random(seed)
            def draw(shots):
                return [sum(rng.random()<p for _ in range(n)) for n,p in zip(shots,probabilities)]
            history = sequential([1,-2],draw,1000,50,20,1.35)
            hits += all(lo <= truth <= hi for lo,hi in (r.interval for r in history))
            looks += len(history)
        rate = hits/paths
        z = 1.959963984540054
        denominator = 1+z*z/paths
        center = (rate+z*z/(2*paths))/denominator
        radius = z*sqrt(rate*(1-rate)/paths+z*z/(4*paths*paths))/denominator
        yield [*probabilities,paths,hits,rate,max(0,center-radius),min(1,center+radius),looks/paths]


if __name__ == "__main__":
    writer = csv.writer(sys.stdout,lineterminator="\n")
    writer.writerow("p1 p2 paths simultaneous_hits coverage wilson_low wilson_high mean_looks".split())
    writer.writerows(compare())
