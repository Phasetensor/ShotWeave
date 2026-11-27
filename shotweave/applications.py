"""Analysis of grouped energies, independent noise scales and backend records."""

from dataclasses import dataclass
from math import fsum, isfinite, prod
from typing import Sequence

from .measurements import covariance, group_variance


def grouped_energy(
    offset: float,
    coefficients: Sequence[Sequence[float]],
    observations: Sequence[Sequence[Sequence[float]]],
) -> tuple[float, float]:
    """Estimate a Pauli energy and its variance from independent measurement groups."""
    if (
        not isfinite(offset)
        or not coefficients
        or len(coefficients) != len(observations)
    ):
        raise ValueError("Invalid Hamiltonian")
    value, variance = offset, 0.0
    for c, rows in zip(coefficients, observations):
        matrix = covariance(rows)
        if any(x not in (-1, 1) for row in rows for x in row):
            raise ValueError("Expected joint Pauli eigenvalues")
        variance += group_variance(c, matrix) / len(rows)
        value += fsum(a * fsum(col) / len(rows) for a, col in zip(c, zip(*rows)))
    return value, variance


def extrapolate(
    scales: Sequence[float], values: Sequence[float], variances: Sequence[float]
) -> tuple[float, float]:
    """Zero-noise polynomial extrapolation with independent estimates per scale."""
    if (
        len(scales) < 2
        or len(scales) != len(set(scales))
        or len(scales) != len(values)
        or len(values) != len(variances)
        or not all(isfinite(s) and s > 0 for s in scales)
        or not all(map(isfinite, values))
        or not all(isfinite(v) and v >= 0 for v in variances)
    ):
        raise ValueError("Invalid noise scale experiment")
    factors = [prod(t / (t - s) for t in scales if t != s) for s in scales]
    return fsum(a * v for a, v in zip(factors, values)), fsum(
        a * a * v for a, v in zip(factors, variances)
    )


@dataclass(frozen=True)
class BackendRecord:
    backend: str
    calibration: str
    observable: str
    job: str
    shots: int
    ones: int
    seconds: float

    def __post_init__(self) -> None:
        if (
            not all((self.backend, self.calibration, self.observable, self.job))
            or type(self.shots) is not int
            or type(self.ones) is not int
            or self.shots < 1
            or not 0 <= self.ones <= self.shots
            or not isfinite(self.seconds)
            or self.seconds < 0
        ):
            raise ValueError("Invalid backend measurement")


def aggregate(records: Sequence[BackendRecord]) -> tuple[int, int, float]:
    """Pool counts only within one backend, calibration and observable."""
    keys = {(r.backend, r.calibration, r.observable) for r in records}
    if len(keys) != 1 or len({r.job for r in records}) != len(records):
        raise ValueError("Incompatible or duplicate measurements")
    return (
        sum(r.ones for r in records),
        sum(r.shots for r in records),
        fsum(r.seconds for r in records),
    )


def bornlens_rows(
    pilot_job: str,
    final_job: str,
    pilot_shots: Sequence[int],
    final_shots: Sequence[int],
    ones: Sequence[int],
) -> list[dict]:
    """JSONL rows for BornLens compare_final from independent pilot and final jobs."""
    if (
        not pilot_job
        or not final_job
        or pilot_job == final_job
        or not pilot_shots
        or len(pilot_shots) != len(final_shots)
        or len(ones) != len(final_shots)
        or any(type(n) is not int or n < 1 for n in (*pilot_shots, *final_shots))
        or any(type(k) is not int or not 0 <= k <= n for k, n in zip(ones, final_shots))
    ):
        raise ValueError("Invalid fixed-final experiment")
    return [
        {"stage": "pilot", "job_id": pilot_job, "shots": list(pilot_shots)},
        {"stage": "final", "job_id": final_job, "shots": list(final_shots)},
        {
            "total_shots": sum(pilot_shots) + sum(final_shots),
            "final_shots": list(final_shots),
            "successes": list(ones),
        },
    ]
