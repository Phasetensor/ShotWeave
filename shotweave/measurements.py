"""Grouped Pauli measurements and calibrated single-qubit readout."""
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Sequence


def compatible(a: str, b: str, general: bool = False) -> bool:
    if not a or len(a) != len(b) or set(a+b) - set("IXYZ"):
        raise ValueError("Expected equal-length Pauli strings")
    conflicts = sum(x != y and x != "I" and y != "I" for x,y in zip(a,b))
    return conflicts % 2 == 0 if general else conflicts == 0


def groups(terms: Sequence[str], general: bool = False) -> list[list[str]]:
    """Stable first-fit grouping. General commuting groups can need entangling rotations."""
    result: list[list[str]] = []
    for term in terms:
        if terms:
            compatible(term, terms[0])
        for group in result:
            if all(compatible(term,t,general) for t in group):
                group.append(term)
                break
        else:
            result.append([term])
    return result


def basis(group: Sequence[str]) -> str:
    if not group or len(groups(group)) != 1:
        raise ValueError("Local measurement requires a nonempty QWC group")
    return "".join(next((x for x in col if x != "I"), "Z") for col in zip(*group))


def covariance(samples: Sequence[Sequence[float]], shrinkage: float = 0) -> list[list[float]]:
    """Sample covariance of jointly measured outcomes with diagonal shrinkage."""
    n = len(samples)
    if n < 2 or not samples[0] or not 0 <= shrinkage <= 1:
        raise ValueError("Need two or more observations and valid shrinkage")
    if any(len(row)!=len(samples[0]) or not all(map(isfinite,row)) for row in samples):
        raise ValueError("Invalid observations")
    columns = list(zip(*samples))
    centered = [[x-fsum(col)/n for x in col] for col in columns]
    return [[fsum(x*y for x,y in zip(a,b))/(n-1)*(1 if i==j else 1-shrinkage)
             for j,b in enumerate(centered)] for i,a in enumerate(centered)]


def group_variance(coefficients: Sequence[float], matrix: Sequence[Sequence[float]]) -> float:
    """Validate symmetry and positive semidefiniteness using diagonal pivoting."""
    n = len(coefficients)
    if not n or len(matrix)!=n or not all(map(isfinite,coefficients)) or any(len(row)!=n or not all(map(isfinite,row)) for row in matrix):
        raise ValueError("Invalid covariance dimensions or values")
    m = [list(row) for row in matrix]
    tol = 1e-12*max(abs(x) for row in m for x in row)
    if any(abs(m[i][j]-m[j][i])>tol for i in range(n) for j in range(i)):
        raise ValueError("Asymmetric covariance")
    for i in range(n):
        p = max(range(i,n),key=lambda j:m[j][j])
        m[i],m[p] = m[p],m[i]
        for row in m:
            row[i],row[p] = row[p],row[i]
        pivot = m[i][i]
        if pivot <= tol:
            if any(abs(m[j][k])>tol for j in range(i,n) for k in range(i,n)):
                raise ValueError("Indefinite covariance")
            break
        for j in range(i+1,n):
            for k in range(i+1,n):
                m[j][k] -= m[j][i]*m[i][k]/pivot
    return max(0.,fsum(a*b*matrix[i][j] for i,a in enumerate(coefficients) for j,b in enumerate(coefficients)))


@dataclass(frozen=True)
class Readout:
    false_one: float
    false_zero: float

    def __post_init__(self) -> None:
        if not all(0 <= p <= 1 for p in (self.false_one,self.false_zero)) or self.false_one+self.false_zero >= 1:
            raise ValueError("Unidentifiable or inverted readout model")

    def correct(self, observed: float, variance: float) -> tuple[float,float]:
        """Linear inverse; conditional variance treats calibration rates as known."""
        if not 0<=observed<=1 or not isfinite(variance) or variance<0:
            raise ValueError("Invalid measurement")
        slope = 1-self.false_one-self.false_zero
        return (observed-self.false_one)/slope,variance/slope**2
