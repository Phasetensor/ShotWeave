"""Public allocation, measurement and inference API for ShotWeave."""

from .allocation import adaptive_allocation, allocate, optimal_allocation
from .applications import (
    BackendRecord,
    aggregate,
    bornlens_rows,
    extrapolate,
    grouped_energy,
)
from .design import (
    CostPlan,
    Posterior,
    Resources,
    bayesian_allocation,
    continuous_cost,
    integer_cost,
    pareto,
    pilot_risks,
    precision_plan,
    robust_allocation,
)
from .experiment import (
    Estimate,
    automatic_pilot,
    confidence_interval,
    run_experiment,
    summarize,
)
from .measurements import Readout, basis, compatible, covariance, group_variance, groups
from .sequential import Look, confidence_sequence, sequential

__all__ = [
    "BackendRecord",
    "CostPlan",
    "Estimate",
    "Look",
    "Posterior",
    "Readout",
    "Resources",
    "adaptive_allocation",
    "aggregate",
    "allocate",
    "automatic_pilot",
    "basis",
    "bayesian_allocation",
    "bornlens_rows",
    "compatible",
    "confidence_interval",
    "confidence_sequence",
    "continuous_cost",
    "covariance",
    "extrapolate",
    "group_variance",
    "grouped_energy",
    "groups",
    "integer_cost",
    "optimal_allocation",
    "pareto",
    "pilot_risks",
    "precision_plan",
    "robust_allocation",
    "run_experiment",
    "sequential",
    "summarize",
]
