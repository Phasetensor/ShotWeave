"""Integer shot allocation for independent binary measurements."""
from fractions import Fraction
from heapq import heapify, heapreplace
from typing import Iterable
from math import isfinite, sqrt


def allocate(budget: int, weights: Iterable[float]) -> list[int]:
    """Reserve one shot per circuit, then apportion by largest remainder."""
    weights = tuple(weights)
    if type(budget) is not int or not weights or budget < len(weights):
        raise ValueError("Provide an integer budget covering every circuit")
    if any(not isfinite(w) or w < 0 for w in weights) or not max(weights):
        raise ValueError("Weights must be finite, nonnegative and include a positive value")
    exact = [Fraction(w) for w in weights]
    total = sum(exact)
    quotas = [(budget - len(weights)) * w / total for w in exact]
    counts = [1 + int(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: quotas[i] % 1, reverse=True)
    for i in order[:budget - sum(counts)]:
        counts[i] += 1
    return counts


def adaptive_allocation(budget: int, coefficients: Iterable[float], successes: Iterable[int],
                        pilot_size: int, *, optimal: bool = False) -> list[int]:
    """Allocate the remaining shots after an equal-size pilot per circuit."""
    coefficients, successes = tuple(coefficients), tuple(successes)
    if type(pilot_size) is not int or pilot_size < 1 or len(successes) != len(coefficients):
        raise ValueError("Provide one pilot count per coefficient")
    if any(type(k) is not int or not 0 <= k <= pilot_size for k in successes):
        raise ValueError("Invalid pilot success count")
    probabilities = [(k + 0.5) / (pilot_size + 1) for k in successes]
    weights = [abs(c) * sqrt(p * (1 - p)) for c, p in zip(coefficients, probabilities)]
    allocator = optimal_allocation if optimal else allocate
    return allocator(budget - pilot_size * len(weights), weights)


def optimal_allocation(budget: int, weights: Iterable[float]) -> list[int]:
    """Minimize sum(w_i**2 / n_i) over positive integer shot counts."""
    values = tuple(weights)
    allocate(len(values), values)  # same domain as the proportional allocator
    if type(budget) is not int or budget < len(values):
        raise ValueError("Budget must cover every circuit")
    squares = [Fraction(w)**2 for w in values]
    counts = [1] * len(values)
    gains = [(-a/2, i) for i, a in enumerate(squares)]
    heapify(gains)
    for _ in range(budget - len(values)):
        _, i = gains[0]
        counts[i] += 1
        n = counts[i]
        heapreplace(gains, (-squares[i] / (n * (n+1)), i))
    return counts
