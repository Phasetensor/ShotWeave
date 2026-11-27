"""Resource designs for independent measurement groups."""
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isfinite, sqrt
from typing import Callable, Iterable, Sequence

from . import allocate, optimal_allocation


def _weights(weights: Sequence[float]) -> None:
    if not weights or any(w < 0 or not isfinite(w) for w in weights):
        raise ValueError("Expected nonnegative finite weights")


def _budget(budget: int, costs: Sequence[int], minimum: int) -> None:
    if (type(budget) is not int or type(minimum) is not int or minimum < 1 or not costs
            or any(type(k) is not int or k < 1 for k in costs) or budget < minimum*sum(costs)):
        raise ValueError("Infeasible integer resource budget")


@dataclass(frozen=True)
class CostPlan:
    shots: tuple[int, ...]
    variance: float
    spent: int
    unspent: int


def continuous_cost(budget: int, weights: Sequence[float], costs: Sequence[int],
                    minimum: int = 1) -> tuple[float, ...]:
    """Continuous lower bound with active lower constraints."""
    _weights(weights)
    _budget(budget, costs, minimum)
    if len(weights) != len(costs):
        raise ValueError("Length mismatch")
    n = [float(minimum)] * len(weights)
    active = {i for i,w in enumerate(weights) if w}
    while active:
        remaining = budget - sum(costs[i]*minimum for i in range(len(n)) if i not in active)
        scale = remaining/fsum(weights[i]*sqrt(costs[i]) for i in active)
        blocked = {i for i in active if scale*weights[i]/sqrt(costs[i]) < minimum}
        if not blocked:
            for i in active:
                n[i] = scale*weights[i]/sqrt(costs[i])
            break
        active -= blocked
    return tuple(n)


def integer_cost(budget: int, weights: Sequence[float], costs: Sequence[int],
                 minimum: int = 1) -> CostPlan:
    """Exact multiple-choice knapsack DP for small integer cost budgets."""
    continuous_cost(budget, weights, costs, minimum)
    if minimum == 1 and all(k == 1 for k in costs) and any(weights):
        counts = tuple(optimal_allocation(budget, weights))
    else:
        states: dict[int, tuple[Fraction, tuple[int,...]]] = {0:(Fraction(0), ())}
        remaining = sum(costs)*minimum
        for w,k in zip(weights,costs):
            remaining -= k*minimum
            nxt: dict[int, tuple[Fraction,tuple[int,...]]] = {}
            for spent,(objective,previous) in states.items():
                for n in range(minimum,(budget-remaining-spent)//k+1):
                    key = spent+k*n
                    candidate = (objective+Fraction(w)**2/n,previous+(n,))
                    if key not in nxt or candidate < nxt[key]:
                        nxt[key] = candidate
            states = nxt
        counts = min(states.values())[1]
    spent = sum(k*n for k,n in zip(costs,counts))
    return CostPlan(counts, fsum(w*w/n for w,n in zip(weights,counts)), spent, budget-spent)


def robust_allocation(budget: int, coefficients: Sequence[float],
                      intervals: Sequence[tuple[float,float]]) -> list[int]:
    """Minimax under rectangular uncertainty on independent group variances."""
    if len(coefficients)!=len(intervals) or any(not 0<=lo<=hi or not isfinite(hi) for lo,hi in intervals):
        raise ValueError("Invalid variance intervals")
    weights = [abs(c)*sqrt(hi) for c,(_,hi) in zip(coefficients,intervals)]
    return optimal_allocation(budget,weights if any(weights) else [1.]*len(weights))


@dataclass(frozen=True)
class Resources:
    variance: float
    cost: float
    seconds: float
    jobs: int

    def __post_init__(self) -> None:
        if type(self.jobs) is not int or self.jobs < 1 or any(not isfinite(v) or v<0 for v in (self.cost,self.seconds,self.variance)):
            raise ValueError("Invalid resource dimensions")


def pareto(plans: Iterable[Resources]) -> list[Resources]:
    """Nondominated candidates; comparisons keep native units separate."""
    items = list(dict.fromkeys(plans))
    def dims(p: Resources) -> tuple[float,...]:
        return p.variance,p.cost,p.seconds,p.jobs
    return [p for p in items if not any(q!=p and all(a<=b for a,b in zip(dims(q),dims(p))) for q in items)]


def precision_plan(weights: Sequence[float], target_variance: float, max_shots: int) -> CostPlan:
    """Smallest feasible total for unit shot costs and supplied variance weights."""
    _weights(weights)
    if target_variance<=0 or not isfinite(target_variance):
        raise ValueError("Invalid target variance")
    costs = [1]*len(weights)
    if not any(weights):
        _budget(max_shots,costs,1)
        return CostPlan(tuple(costs),0.,len(costs),0)
    lo,hi = len(weights),max_shots
    if integer_cost(hi,weights,costs).variance>target_variance:
        raise ValueError("Target infeasible")
    while lo<hi:
        mid=(lo+hi)//2
        if integer_cost(mid,weights,costs).variance<=target_variance:
            hi=mid
        else:
            lo=mid+1
    return integer_cost(lo,weights,costs)


@dataclass(frozen=True)
class Posterior:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) and x>0 for x in (self.alpha,self.beta)):
            raise ValueError("Beta shapes must be finite and positive")

    def update(self, ones: int, shots: int) -> "Posterior":
        if type(ones) is not int or type(shots) is not int or not 0<=ones<=shots:
            raise ValueError("Invalid counts")
        return Posterior(self.alpha+ones,self.beta+shots-ones)

    @property
    def mean(self) -> float:
        return self.alpha/(self.alpha+self.beta)

    @property
    def measurement_variance(self) -> float:
        return self.mean*self.beta/(self.alpha+self.beta+1)

    @property
    def parameter_variance(self) -> float:
        return self.measurement_variance/(self.alpha+self.beta)


def bayesian_allocation(budget: int, coefficients: Sequence[float], posterior: Sequence[Posterior]) -> list[int]:
    if len(coefficients)!=len(posterior):
        raise ValueError("Length mismatch")
    return optimal_allocation(budget,[abs(c)*sqrt(p.measurement_variance) for c,p in zip(coefficients,posterior)])


def pilot_risks(budget: int, coefficients: Sequence[float], probabilities: Sequence[float],
                candidates: Sequence[int], draw: Callable[[float,int], int], trials: int) -> dict[int,float]:
    """Scenario-conditional simulation risk for candidate equal pilots, including their cost."""
    if len(probabilities)!=len(coefficients) or not all(0<=p<=1 for p in probabilities) or trials<1:
        raise ValueError("Invalid pilot design scenario")
    risks = {}
    for k in candidates:
        if type(k) is not int or k<1 or budget<(k+1)*len(coefficients):
            raise ValueError("Infeasible pilot")
        total = 0.
        for _ in range(trials):
            post = [Posterior(.5,.5).update(draw(p,k),k) for p in probabilities]
            weights = [abs(c)*sqrt(d.measurement_variance) for c,d in zip(coefficients,post)]
            n = allocate(budget-k*len(post),weights)
            total += fsum(c*c*p*(1-p)/m for c,p,m in zip(coefficients,probabilities,n))
        risks[k]=total/trials
    return risks
