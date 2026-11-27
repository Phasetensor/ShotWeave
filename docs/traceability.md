# Advanced module traceability

| Module | Public entry point or runner | Evidence |
| --- | --- | --- |
| Covariance | covariance, group_variance | test_advanced.py, correlated and singular observations |
| Pauli grouping | groups, basis, compatible | test_advanced.py, QWC and general commutation |
| Cost-aware allocation | continuous_cost, integer_cost | test_advanced.py, exhaustive discrete comparisons |
| Robust allocation | robust_allocation | test_release.py, rectangular variance bounds |
| Pilot design | pilot_risks | test_advanced.py, candidate design risk |
| Bayesian allocation | Posterior, bayesian_allocation | test_advanced.py, analytic moments |
| Sequential allocation | sequential, confidence_sequence | benchmark_sequential.csv, simultaneous path coverage |
| Multi-objective design | Resources, pareto | test_advanced.py, dominance |
| Resource estimation | precision_plan | test_advanced.py, minimum shot budget |
| Noise-aware optimization | run_experiment, Readout | test_release.py, allocation sensitivity to calibration |
| VQE | grouped_energy | h2_example.py, two-qubit reference |
| Error mitigation | Readout.correct, extrapolate | test_advanced.py, propagated variances |
| Multi-backend accounting | BackendRecord, aggregate | test_advanced.py, incompatible or duplicate records |
| Benchmark suite | benchmark_advanced.py | benchmark_advanced.csv, matched simulated trials |
| Hardware validation | hardware_example.py, hardware_comparison.py | hardware-run.jsonl, historical adaptive run only |
