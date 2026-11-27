# ShotWeave methods

## Estimand and assumptions

Each circuit produces independent Bernoulli observations with a stable probability
p_i. For signed coefficients c_i, the target is the sum of c_i p_i. The final
estimate substitutes the observed final frequency for each probability. With
fixed final allocations, its variance is the sum of c_i² p_i(1−p_i)/n_i.
Conditioning on an independent pilot gives the same expression for adaptive
allocation. This argument excludes drift and correlated device error.

## Integer allocation

For weights w_i = |c_i| sqrt(p_i(1−p_i)), optimal_allocation minimizes
sum(w_i²/n_i) with at least one shot per circuit and an exact integer budget.
Starting from one shot each, the next shot reduces the objective by
w_i²/[n_i(n_i+1)]. These gains decrease within each circuit, so repeatedly
taking the greatest available gain selects the largest feasible prefix of gains
and attains the global optimum. Ties use circuit order. Exact rational arithmetic
avoids ties caused by rounded marginal gains. The heap requires O(k) space and
O((B−k) log k) update operations for k circuits and budget B, excluding rational
arithmetic costs. Largest remainder remains the default. Optimality with estimated
weights applies to that plug-in objective only.

| Budget | Weights | Largest remainder | Objective | Discrete optimum | Objective |
|---:|---|---|---:|---|---:|
| 5 | 1, 2, 5 | 1, 2, 2 | 15.5 | 1, 1, 3 | 13.3333 |
| 8 | 1, 2, 5 | 2, 2, 4 | 8.75 | 1, 2, 5 | 8 |
| 12 | 1, 2, 5 | 2, 3, 7 | 5.4048 | 2, 3, 7 | 5.4048 |

## Uncertainty and replication

The standard error reported for an individual estimate uses final sample
variances. The Hoeffding confidence interval instead uses the bounded range
of each shot's contribution and is conservative near boundary probabilities.
The benchmark CSV separates sampling uncertainty from Monte Carlo uncertainty.
For R independent seeded trials with signed errors e, bias is mean(e), RMSE is
sqrt(mean(e²)), and their Monte Carlo standard errors are sd(e)/sqrt(R) and
sd(e²)/(2 RMSE sqrt(R)) respectively. The latter uses the delta method and is
undefined for zero RMSE. Compare within the same scenario. Strategies share
seed labels, so separate standard errors do not directly quantify a paired
difference. Regenerate the table with Python 3.12 and benchmark_grid.py.

## Hardware comparison

Use the same backend, qubit layout, circuit family and 10,000-shot budget per
strategy. Randomize execution order within each replicate block to limit
confounding with drift. Report each job, timestamp, estimate and execution
usage. Compare all strategies to the ideal circuit target and label this as
total deviation including hardware noise. Device bias relative to an unknown
instantaneous hardware expectation cannot be identified from the ideal target.
One completed adaptive run verifies execution. Comparative performance requires
repeated matched blocks.
