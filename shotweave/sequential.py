"""Anytime-valid intervals for predictable batches of independent Bernoulli shots."""
from dataclasses import dataclass
from math import fsum, isfinite, log, pi, sqrt
from typing import Callable, Sequence

from .design import Posterior, bayesian_allocation


@dataclass(frozen=True)
class Look:
    shots: tuple[int,...]
    ones: tuple[int,...]
    interval: tuple[float,float]


def confidence_sequence(coefficients: Sequence[float], shots: Sequence[int],
                        ones: Sequence[int], alpha: float = .05) -> tuple[float,float]:
    """Union bound across arms and all intrinsic sample sizes using 6/(pi*n)^2."""
    arms = len(shots)
    if not arms or len(coefficients)!=arms or len(ones)!=arms or not 0<alpha<1:
        raise ValueError("Invalid dimensions or error probability")
    lows, highs = [], []
    for c,n,k in zip(coefficients,shots,ones):
        if type(n) is not int or type(k) is not int or not 0<=k<=n or not isfinite(c):
            raise ValueError("Invalid measurements")
        lo,hi = 0.,1.
        if n:
            radius = sqrt(log(2*arms*pi**2*n**2/(6*alpha))/(2*n))
            lo,hi = max(0.,k/n-radius),min(1.,k/n+radius)
        lows.append(min(c*lo,c*hi))
        highs.append(max(c*lo,c*hi))
    return fsum(lows),fsum(highs)


def sequential(coefficients: Sequence[float], submit: Callable[[tuple[int,...]],Sequence[int]],
               budget: int, batch: int, max_jobs: int, width: float, alpha: float=.05) -> list[Look]:
    size = len(coefficients)
    if not size or not all(map(isfinite,coefficients)) or not any(coefficients) or not 0<alpha<1 or not isfinite(width) or width<0 or any(type(v) is not int for v in (budget,batch,max_jobs)) or min(batch,budget)<size or max_jobs<1:
        raise ValueError("Invalid sequential design")
    history: list[Look] = []
    shots,ones = [0]*size,[0]*size
    for _ in range(max_jobs):
        available = min(batch,budget-sum(shots))
        if available<size:
            break
        post = [Posterior(.5,.5).update(k,n) for k,n in zip(ones,shots)]
        plan = tuple(bayesian_allocation(available,coefficients,post))
        observed = tuple(submit(plan))
        confidence_sequence(coefficients,plan,observed,alpha)
        shots = [a+b for a,b in zip(shots,plan)]
        ones = [a+b for a,b in zip(ones,observed)]
        interval = confidence_sequence(coefficients,shots,ones,alpha)
        history.append(Look(tuple(shots),tuple(ones),interval))
        if interval[1]-interval[0]<=width:
            break
    return history
