"""Execution of independent binary measurement batches."""
from dataclasses import dataclass
from math import fsum, isfinite, isqrt, log, sqrt
from typing import Optional, Tuple

from . import adaptive_allocation, allocate


@dataclass(frozen=True)
class Estimate:
    value: float
    standard_error: Optional[float]
    final_shots: Tuple[int, ...]
    successes: Tuple[int, ...]
    pilot_shots: Tuple[int, ...] = ()
    pilot_successes: Tuple[int, ...] = ()

    @property
    def total_shots(self):
        return sum(self.final_shots) + sum(self.pilot_shots)

    @property
    def batches(self):
        return 2 if self.pilot_shots else 1


def _check_counts(shots, successes):
    if len(shots) != len(successes) or any(
        type(k) is not int or not 0 <= k <= n for n, k in zip(shots, successes)
    ):
        raise ValueError("Expected one integer success count in [0, shots] per circuit")


def summarize(coefficients, shots, successes):
    """Estimate a weighted sum of probabilities from independent final counts."""
    coefficients, shots, successes = map(tuple, (coefficients, shots, successes))
    if not shots or len(coefficients) != len(shots) or any(
        type(n) is not int or n < 1 for n in shots
    ):
        raise ValueError("Expected one positive shot count per coefficient")
    if not all(map(isfinite, coefficients)):
        raise ValueError("Coefficients must be finite")
    _check_counts(shots, successes)
    value = fsum(c * (k / n) for c, k, n in zip(coefficients, successes, shots))
    standard_error = None
    if all(n > 1 or c == 0 for c, n in zip(coefficients, shots)):
        variance = fsum(
            c*c * (k/n) * ((n-k)/n) / (n-1)
            for c, k, n in zip(coefficients, successes, shots) if c
        )
        standard_error = sqrt(variance)
    return Estimate(value, standard_error, shots, successes)


def confidence_interval(result, coefficients, level=0.95):
    """Two-sided Hoeffding interval conditional on the independent final allocation."""
    coefficients = tuple(coefficients)
    if not 0 < level < 1 or len(coefficients) != len(result.final_shots):
        raise ValueError("Invalid confidence level or coefficient count")
    estimate = summarize(coefficients, result.final_shots, result.successes)
    scale = fsum(c*c/n for c, n in zip(coefficients, result.final_shots))
    radius = sqrt(0.5 * log(2 / (1 - level)) * scale)
    return (max(fsum(min(0, c) for c in coefficients), estimate.value - radius),
            min(fsum(max(0, c) for c in coefficients), estimate.value + radius))


def automatic_pilot(budget, circuits):
    """Square-root pilot per circuit, capped at 10% except for the one-shot floor."""
    if type(circuits) is not int or circuits < 1 or type(budget) is not int or budget < 2 * circuits:
        raise ValueError("Adaptive sampling needs two shots per circuit")
    per_circuit = budget // circuits
    return max(1, min(isqrt(per_circuit), per_circuit // 10))


def run_experiment(budget, coefficients, submit, strategy="adaptive", pilot_size=64):
    """Call submit(shots) in circuit order and return the estimate and both batches."""
    coefficients = tuple(coefficients)
    if strategy not in ("uniform", "weighted", "adaptive"):
        raise ValueError("Unknown allocation strategy")
    if not coefficients or not all(map(isfinite, coefficients)) or not any(coefficients):
        raise ValueError("Provide finite coefficients with a nonzero term")
    if type(budget) is not int or budget < len(coefficients):
        raise ValueError("Budget must cover every circuit")
    pilot = pilot_counts = ()
    if strategy == "adaptive":
        if pilot_size == "auto":
            pilot_size = automatic_pilot(budget, len(coefficients))
        if type(pilot_size) is not int or not 1 <= pilot_size <= budget // len(coefficients) - 1:
            raise ValueError("Budget must cover the pilot and at least one final shot per circuit")
        pilot = (pilot_size,) * len(coefficients)
        pilot_counts = tuple(submit(pilot))
        _check_counts(pilot, pilot_counts)
        final = adaptive_allocation(budget, coefficients, pilot_counts, pilot_size)
    else:
        weights = [1] * len(coefficients) if strategy == "uniform" else map(abs, coefficients)
        final = allocate(budget, weights)
    result = summarize(coefficients, final, submit(tuple(final)))
    return Estimate(result.value, result.standard_error, result.final_shots,
                    result.successes, pilot, pilot_counts)
