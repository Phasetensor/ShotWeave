# Resource allocation and inference

For independent groups with variance weights w, the objective is the sum of w²/n. With unequal integer costs, continuous_cost enforces the lower bounds and gives a relaxation. integer_cost minimizes over feasible integer counts by dynamic programming. Its worst-case time grows quadratically with the integer budget per group. Equal unit costs use a marginal-gain heap. Unspent cost is reported explicitly.

Use measurements.covariance on simultaneous eigenvalue observations from a group. Off-diagonal shrinkage trades bias for stability. group_variance computes the quadratic form after a pivoted PSD check. Pass the square roots of group variances to optimal_allocation. QWC groups admit local basis changes. General commuting groups can require a joint diagonalizer, which the grouping function does not synthesize.

robust_allocation uses the upper endpoints of rectangular variance bounds. Correlated uncertainty sets require a different optimization. Posterior.measurement_variance is the posterior mean of Bernoulli outcome variance. parameter_variance instead measures uncertainty about the probability. pilot_risks compares specified pilot sizes conditional on a supplied probability scenario. Its ranking depends on the scenario and random draws.

sequential chooses each batch using only previous observations. confidence_sequence combines Hoeffding bounds over arms and intrinsic sample sizes using summable error spending. Independent stationary Bernoulli outcomes per arm are assumed. Inspect Look records for realized counts, bounds and stopping. A fixed-final interval is inappropriate after outcome-dependent stopping.

Readout.correct propagates sampling variance through a linear inverse conditional on known calibration rates. extrapolate propagates independent variances through polynomial zero-noise extrapolation. Both methods can amplify variance. aggregate enforces matching backend, observable and calibration identifiers. Caller identifiers must encode circuit and measurement semantics.

Run python benchmark_advanced.py to regenerate the matched-block CSV. Every method gets 320 shots per block including the pilot. A block reuses random streams within a stage across strategies. Pilot and final streams are independent. RMSE standard errors use the delta method. Equal pilot sizes and a common symmetric Beta prior can make Bayesian and adaptive-optimal allocations coincide. Simulation coverage records empirical inclusion of the known value.

Run python h2_example.py for a two-qubit H2 variational-state measurement simulation. The exact reference diagonalizes both parity sectors of the rounded Hamiltonian. The reported variance includes within-Z-group covariance. The simulation uses exact-state weights as an oracle for measurement design.

For BornLens, serialize each dictionary returned by bornlens_rows as one JSON line and pass the file to bornlens.shotweave.compare_final. Use independent final measurements of single-qubit Z. The existing IBM evidence in hardware-run.jsonl covers the original adaptive strategy.
