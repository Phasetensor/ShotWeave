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

Development preview under Apache 2.0. The benchmarks cover three synthetic scenarios. No hardware or PyPI release has been validated.
