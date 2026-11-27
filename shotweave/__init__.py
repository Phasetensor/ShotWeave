"""Integer shot allocation for independent binary measurements."""
from math import floor, fsum, isfinite, sqrt


def allocate(budget, weights):
    """Reserve one shot per circuit, then apportion by largest remainder."""
    weights = tuple(weights)
    if type(budget) is not int or not weights or budget < len(weights):
        raise ValueError("Provide an integer budget covering every circuit")
    if any(not isfinite(w) or w < 0 for w in weights) or not max(weights):
        raise ValueError("Weights must be finite, nonnegative and include a positive value")
    scaled = [w / max(weights) for w in weights]
    total = fsum(scaled)
    quotas = [(budget - len(weights)) * w / total for w in scaled]
    counts = [1 + floor(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: quotas[i] % 1, reverse=True)
    for i in order[:budget - sum(counts)]:
        counts[i] += 1
    return counts


def adaptive_allocation(budget, coefficients, successes, pilot_size):
    """Allocate the remaining shots after an equal-size pilot per circuit."""
    coefficients, successes = tuple(coefficients), tuple(successes)
    if type(pilot_size) is not int or pilot_size < 1 or len(successes) != len(coefficients):
        raise ValueError("Provide one pilot count per coefficient")
    if any(type(k) is not int or not 0 <= k <= pilot_size for k in successes):
        raise ValueError("Invalid pilot success count")
    probabilities = [(k + 0.5) / (pilot_size + 1) for k in successes]
    weights = [abs(c) * sqrt(p * (1 - p)) for c, p in zip(coefficients, probabilities)]
    return allocate(budget - pilot_size * len(weights), weights)
