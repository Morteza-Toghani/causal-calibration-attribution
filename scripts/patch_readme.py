from pathlib import Path

p = Path("README.md")
s = p.read_text(encoding="utf-8")
orig = s

new_block = """## Key results

We trained 5-member Gaussian MLP ensembles on Minari Hopper and Walker2d
(**10 seeds each**) and evaluated three evaluation-time interventions
(dynamics, observation, policy) at two severities each. The primary
estimand is the paired average treatment effect (ATE) on instance-level
regression calibration error. All numbers are means over **n = 10 seeds**;
confidence intervals use Student's t with `df = 9` (`t_crit = 2.262`).

### 1. Baseline over-dispersion

The 5-member ensemble **over-covers at every nominal level**. Signed
coverage at nominal 0.50 is `+0.35` (empirical 0.85 instead of 0.50).
A single per-seed variance scaling factor fitted on the **calibration
split** (nominal 0.90) brings coverage to nominal:

- Hopper: `alpha = 0.2361` (variance ~4.2x too large)
- Walker2d: `alpha = 0.3743` (variance ~2.7x too large)

### 2. Observation shift dominates after recalibration

Recalibrated ATE (main result), n = 10 per environment:

| Condition | Hopper | Walker2d |
|---|---|---|
| `observation_low`  | **+0.20125** +/- 0.00429 | **+0.07749** +/- 0.00393 |
| `observation_high` | **+0.26588** +/- 0.00319 | **+0.15268** +/- 0.00383 |
| `dynamics_low`     | -0.00113 +/- 0.00068 | -0.00178 +/- 0.00042 |
| `dynamics_high`    | -0.00445 +/- 0.00212 | -0.01194 +/- 0.00125 |
| `policy_low`       | +0.00086 +/- 0.00029 | +0.00486 +/- 0.00071 |
| `policy_high`      | +0.00247 +/- 0.00094 | +0.00630 +/- 0.00248 |

95% t-intervals for all twelve conditions are reported in
[`results/tables/final_n10_summary_recalibrated.csv`](results/tables/final_n10_summary_recalibrated.csv).

Observation effects are **13x to 178x larger** than dynamics or policy
at matched severity labels. All twelve effects exclude zero at 95%.

### 3. Sensitivity to the variance-scaling factor alpha

We swept alpha on a 20-point grid (alpha in [0.05, 1.00]) and recomputed
the ATE for all twelve conditions at each value. The observation effect
remains the dominant pathway across the swept range:

| alpha | Hopper O_high/D_high | Walker2d O_high/D_high |
|---|---|---|
| fitted (0.25 / 0.35) | 45.7x | 15.6x |
| 0.50 | 9.1x | 5.9x |
| 0.75 | 4.6x | 1.1x |
| 1.00 (no recalibration) | 1.9x | 1.4x |

The sign flip in `observation_low` as alpha increases is
**over-dispersion-driven, not environment-driven** -- see
[`docs/paper1_findings.md`](docs/paper1_findings.md).

![Alpha sensitivity](results/figures/alpha_sensitivity.png)

### 4. Re-simulation bias in dynamics and policy

An identity intervention (`mass_scale=1.0` for dynamics,
`action_scale=1.0` for policy) does **not** reproduce the dataset target
bit-for-bit. The cause is MuJoCo `set_state` + `step` not resetting
`qacc_warmstart` internally, plus the x=0 reconstruction of state from
the observation. Correcting the paired baseline to use
`y_resim(scale=1.0)` changes the D and P ATEs substantially.

### Reproduce
python scripts/final_n10_analysis.py # n=10 corrected ATE
python scripts/final_n10_recalibrated.py # n=10 recalibrated ATE
python scripts/alpha_sensitivity.py --env both # alpha sweep
python scripts/dom_ratio.py # O/D ratio table
python scripts/fill_placeholders.py # sign-flip p-values

See [`docs/paper1_findings.md`](docs/paper1_findings.md) for the full
findings summary and [`docs/paper1_draft/`](docs/paper1_draft/) for the
manuscript draft.

---

"""

new_status = """> **Status: REPRODUCIBLE**

The research question and experimental protocol are defined and
completed. The full pipeline (Minari data, 10-seed evaluation for both
Hopper and Walker2d, mechanism-specific interventions, CRN-paired
evaluation, baseline recalibration, and alpha-sensitivity sweep) is
implemented and reproducible. Bit-exact reproduction of the original
FROZEN pipeline is verified for baseline and all three interventions.

The project follows an evidence-based maturity model:

`IN DESIGN -> PROTOTYPE -> PRELIMINARY -> VALIDATED -> REPRODUCIBLE`

The current state is **REPRODUCIBLE**. A manuscript draft is available
in `docs/paper1_draft/`, and the key findings are summarized in
`docs/paper1_findings.md`."""

i_start = s.find("## Key findings")
i_end = s.find("## Current Status")
if i_start >= 0 and i_end > i_start:
    s = s[:i_start] + new_block + s[i_end:]
    print("Key findings -> Key results: replaced")
else:
    print("ERROR: markers not found")

i_start = s.find("> **Status: IN DESIGN / PILOT PHASE**")
if i_start >= 0:
    i_end = s.find("\n---", i_start)
    s = s[:i_start] + new_status + s[i_end:]
    print("Current Status: updated")
else:
    print("ERROR: old status marker not found")

p.write_text(s, encoding="utf-8", newline="\n")
print(f"README.md written ({len(s)} bytes)")

p6 = Path("docs/paper1_draft/06_results.md")
s6 = p6.read_text(encoding="utf-8")
old_t67 = """| Environment | Severity | $O/\\lvert D\\rvert$ | $O/\\lvert P\\rvert$ |
|---|---|---|---|
| Hopper | low | 178 | 234 |
| Hopper | high | 60 | 108 |
| Walker2d | low | 48 | 16 |
| Walker2d | high | 14 | 22 |"""
new_t67 = """| Environment | Severity | $O/\\lvert D\\rvert$ | $O/\\lvert P\\rvert$ |
|---|---|---|---|
| Hopper | low | 178 | 234 |
| Hopper | high | 60 | 108 |
| Walker2d | low | 44 | 16 |
| Walker2d | high | 13 | 24 |"""
if old_t67 in s6:
    s6 = s6.replace(old_t67, new_t67)
    print("Table 6.7: patched")
else:
    print("Table 6.7: PATTERN NOT FOUND")
s6 = s6.replace(
    "The observation effect was between about 15 and 155 times larger",
    "The observation effect was between about 13 and 178 times larger",
)
p6.write_text(s6, encoding="utf-8", newline="\n")
print("06_results.md: written")
