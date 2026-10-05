# Signed coverage summary

Signed coverage = (empirical coverage - nominal level).
Positive = over-covering; negative = under-covering.

| condition | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | +0.141 | +0.257 | +0.334 | +0.364 | +0.352 | +0.309 | +0.245 | +0.168 | +0.084 |
| **dynamics_high** | +0.123 | +0.225 | +0.293 | +0.322 | +0.314 | +0.278 | +0.221 | +0.150 | +0.073 |
| **dynamics_low** | +0.138 | +0.250 | +0.324 | +0.354 | +0.342 | +0.301 | +0.238 | +0.162 | +0.080 |
| **observation_high** | -0.064 | -0.129 | -0.192 | -0.253 | -0.312 | -0.369 | -0.420 | -0.463 | -0.488 |
| **observation_low** | -0.040 | -0.081 | -0.120 | -0.158 | -0.193 | -0.227 | -0.256 | -0.277 | -0.280 |
| **policy_high** | +0.146 | +0.265 | +0.341 | +0.369 | +0.355 | +0.309 | +0.243 | +0.165 | +0.082 |
| **policy_low** | +0.142 | +0.260 | +0.335 | +0.363 | +0.350 | +0.306 | +0.241 | +0.164 | +0.080 |

### Key observations

* Baseline over-covers at all levels (signed_cov > 0).
* Observation shift: low noise may reduce over-coverage; high noise
  pushes through nominal into under-coverage.
* Policy and dynamics shift have smaller, direction-consistent effects.