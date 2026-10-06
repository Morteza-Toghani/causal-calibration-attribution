"""Fill placeholders in Paper 1 draft.

- Computes sign-flip permutation p-values for recalibrated ATE
  (exact, 2^n over n=10 seeds) + Holm-Bonferroni correction.
- Replaces the 7 [FILL] placeholders in 04, 05, 06 drafts.
"""
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 1. Sign-flip p-values for recalibrated ATE ----
df = pd.read_csv("results/tables/final_n10_per_seed_recalibrated.csv")
rows = []
for (env, mech, lvl), grp in df.groupby(["env", "mechanism", "intensity"]):
    ates = grp.ate.values
    n = len(ates)
    observed = float(np.mean(ates))
    signs = np.array(list(product([-1, 1], repeat=n)))
    means = (signs * ates).mean(axis=1)
    p = float(np.mean(np.abs(means) >= np.abs(observed) - 1e-12))
    rows.append({"env": env, "mechanism": mech, "intensity": lvl,
                 "n": n, "ate_mean": observed, "p_raw": p})

res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
m = len(res)
res["p_holm"] = np.minimum(1.0, res.p_raw * (m - res.index.values))
res["p_holm"] = np.maximum.accumulate(res.p_holm)
res["p_holm"] = np.minimum(1.0, res.p_holm)
res = res.sort_values(["env", "mechanism", "intensity"]).reset_index(drop=True)
res.to_csv("results/tables/recalibrated_signflip_pvalues.csv", index=False)
print("=== Sign-flip p-values (recalibrated ATE) ===")
print(res.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

table_md_lines = [
    "| Env | Mechanism | Intensity | n | ATE | p (raw) | p (Holm) |",
    "|---|---|---|---|---|---|---|",
]
for _, r in res.iterrows():
    table_md_lines.append(
        f"| {r.env} | {r.mechanism} | {r.intensity} | {r.n} | "
        f"{r.ate_mean:+.5f} | {r.p_raw:.5f} | {r.p_holm:.5f} |"
    )
table_md = "\n".join(table_md_lines)

# ---- 2. Patch drafts ----
base = Path("docs/paper1_draft")

# 04_method.md
p = base / "04_method.md"
s = p.read_text(encoding="utf-8")
n_before = s.count("[FILL")
s = s.replace(
    "[FILL: bootstrap resampling unit, number of resamples.]",
    "The bootstrap is a secondary check; the primary confidence intervals "
    "are the seed-level $t$-intervals reported in the results tables. "
    "Where shown, bootstrap confidence intervals resample paired "
    "instances (not seeds) and use 5000 resamples for the observation "
    "and policy conditions and 2000 for the dynamics condition.",
)
p.write_text(s, encoding="utf-8", newline="\n")
print(f"\n04_method.md: {n_before} -> {s.count('[FILL')} FILLs")

# 05_experimental_setup.md
p = base / "05_experimental_setup.md"
s = p.read_text(encoding="utf-8")
n_before = s.count("[FILL")
s = s.replace(
    "[FILL: Minari version and dataset hash or commit.]",
    "Minari 0.5.3, MuJoCo 3.2.3, Gymnasium 1.3.0.",
)
s = s.replace(
    "run on [FILL: device, GPU/CPU model, driver and library versions]",
    "run on CPU only (12 physical cores; PyTorch restricted to 6 threads) "
    "under Windows. PyTorch 2.7.1+cpu, NumPy 1.26.4, SciPy 1.15.3, "
    "pandas 2.3.1, Matplotlib 3.10.9.",
)
s = s.replace(
    "[FILL: determinism flags used, if any.]",
    "No determinism flags beyond per-seed and per-member seeding were "
    "enabled. All reported bit-exact reproductions were obtained on CPU.",
)
s = s.replace(
    "[FILL: repository URL, commit hash, and archive DOI.]",
    "https://github.com/Morteza-Toghani/causal-calibration-attribution "
    "(commit 6a17e80; archive DOI to be assigned at submission).",
)
p.write_text(s, encoding="utf-8", newline="\n")
print(f"05_setup.md: {n_before} -> {s.count('[FILL')} FILLs")

# 06_results.md
p = base / "06_results.md"
s = p.read_text(encoding="utf-8")
n_before = s.count("[FILL")
s = re.sub(
    r"\[FILL: sign-flip permutation[^\]]*\]",
    "Sign-flip permutation $p$-values and Holm-adjusted $p$-values for "
    "each of the twelve recalibrated conditions are reported in "
    "Table 6.2. At $n = 10$ the minimum attainable two-sided $p$ is "
    "0.00195.\n\n"
    "**Table 6.2 — Sign-flip permutation $p$-values (recalibrated ATE).**\n\n"
    + table_md,
    s,
)
s = re.sub(
    r"\[FILL: secondary-metric ATE tables[^\]]*\]",
    "Secondary metrics (NLL, CRPS, sharpness) were not analyzed at the "
    "mechanism level in this paper; the analysis is restricted to "
    "coverage-based calibration error. A mechanism-level analysis of "
    "sharpness and proper scoring rules is left to future work.",
    s,
)
p.write_text(s, encoding="utf-8", newline="\n")
print(f"06_results.md: {n_before} -> {s.count('[FILL')} FILLs")
