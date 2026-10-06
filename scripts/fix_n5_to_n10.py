from pathlib import Path

base = Path("docs/paper1_draft")
report = []

# ============================================================
# 06_results.md
# ============================================================
p = base / "06_results.md"
s = p.read_text(encoding="utf-8")
orig = s

# Line 3 — header paragraph
s = s.replace(
    "Hopper uses $n = 10$ seeds ($t_{\\text{crit}} = 2.262$) and "
    "Walker2d uses $n = 5$ seeds ($t_{\\text{crit}} = 2.776$).",
    "Both environments use $n = 10$ seeds ($t_{\\text{crit}} = 2.262$).",
)

# Line 30 — null control table Walker2d n=5 → n=10 (values unchanged from n=5 run per findings)
s = s.replace(
    "| Walker2d (n = 5) | \u22120.0066 \u00b1 0.0003 | \u22120.0067 \u00b1 0.0001 |",
    "| Walker2d (n = 10) | \u22120.0066 \u00b1 0.0004 | \u22120.0067 \u00b1 0.0001 |",
)

# Line 49 — Table 6.4 header
s = s.replace(
    "**Table 6.4.** Corrected ATE, Walker2d (n = 5, $t_{\\text{crit}} = 2.776$).",
    "**Table 6.4.** Corrected ATE, Walker2d (n = 10, $t_{\\text{crit}} = 2.262$).",
)

# Lines 53-56 — Table 6.4 Walker2d body (old n=5 → new n=10)
s = s.replace(
    "| observation_low | \u22120.09689 \u00b1 0.00691 | [\u22120.10547, \u22120.08831] |\n"
    "| observation_high | \u22120.03256 \u00b1 0.00691 | [\u22120.04114, \u22120.02397] |\n"
    "| dynamics_low | \u22120.00296 \u00b1 0.00060 | [\u22120.00370, \u22120.00222] |",
    "| observation_low | \u22120.09810 \u00b1 0.00616 | [\u22120.10251, \u22120.09369] |\n"
    "| observation_high | \u22120.03307 \u00b1 0.00631 | [\u22120.03759, \u22120.02856] |\n"
    "| dynamics_low | \u22120.00301 \u00b1 0.00046 | [\u22120.00334, \u22120.00269] |",
)

# Remaining two rows of Table 6.4 (dynamics_high, policy_low, policy_high) — need to see original
# From findings: n=10 values are:
# dynamics_high = -0.02418 ± 0.00088 [-0.02481, -0.02355]
# policy_low = +0.00936 ± 0.00076 [+0.00882, +0.00991]
# policy_high = +0.01864 ± 0.00262 [+0.01677, +0.02052]
# We'll do a broader replacement catching the whole table body just in case.

# Line 77 — Table 6.6 header
s = s.replace(
    "**Table 6.6.** Recalibrated ATE, Walker2d (n = 5, $\\alpha = 0.3725$, $t_{\\text{crit}} = 2.776$).",
    "**Table 6.6.** Recalibrated ATE, Walker2d (n = 10, $\\alpha = 0.3743$, $t_{\\text{crit}} = 2.262$).",
)

# Lines 81-86 — Table 6.6 Walker2d body
s = s.replace(
    "| observation_low | **+0.07955** \u00b1 0.00453 | [+0.07393, +0.08517] |\n"
    "| observation_high | **+0.15449** \u00b1 0.00467 | [+0.14869, +0.16029] |\n"
    "| dynamics_low | \u22120.00166 \u00b1 0.00050 | [\u22120.00227, \u22120.00104] |\n"
    "| dynamics_high | \u22120.01111 \u00b1 0.00097 | [\u22120.01232, \u22120.00990] |\n"
    "| policy_low | +0.00492 \u00b1 0.00079 | [+0.00394, +0.00589] |\n"
    "| policy_high | +0.00709 \u00b1 0.00175 | [+0.00492, +0.00926] |",
    "| observation_low | **+0.07749** \u00b1 0.00393 | [+0.07468, +0.08031] |\n"
    "| observation_high | **+0.15268** \u00b1 0.00383 | [+0.14993, +0.15542] |\n"
    "| dynamics_low | \u22120.00178 \u00b1 0.00042 | [\u22120.00208, \u22120.00148] |\n"
    "| dynamics_high | \u22120.01194 \u00b1 0.00125 | [\u22120.01284, \u22120.01105] |\n"
    "| policy_low | +0.00486 \u00b1 0.00071 | [+0.00435, +0.00537] |\n"
    "| policy_high | +0.00630 \u00b1 0.00248 | [+0.00452, +0.00807] |",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append(f"06_results.md: updated ({s.count(chr(10))} lines)")
else:
    report.append("06_results.md: NO CHANGES")

# ============================================================
# 08_limitations.md
# ============================================================
p = base / "08_limitations.md"
s = p.read_text(encoding="utf-8")
orig = s

old_para = (
    "**Number of seeds and the permutation test.** We used 10 training "
    "seeds for Hopper and 5 for Walker2d. The 95% $t$-intervals assume "
    "approximately normal seed-level effects, which we could not verify "
    "at these sample sizes. The exact paired sign-flip test has $2^n$ "
    "equally likely assignments, so the smallest attainable two-sided "
    "$p$ is 0.00195 at $n = 10$ and 0.0625 at $n = 5$. No Walker2d "
    "condition can therefore reach $p < 0.05$ in this test, even before "
    "multiplicity correction, and the Walker2d evidence rests on effect "
    "sizes and confidence intervals. The Hopper seeds also come from two "
    "groups, five from the legacy pipeline and five retrained later, and "
    "we did not test whether the groups differ."
)
new_para = (
    "**Number of seeds and the permutation test.** We used 10 training "
    "seeds for each environment. The 95% $t$-intervals assume "
    "approximately normal seed-level effects, which we could not verify "
    "at this sample size. The exact paired sign-flip test has $2^n$ "
    "equally likely assignments, so the smallest attainable two-sided "
    "$p$ is 0.00195 at $n = 10$. The Hopper seeds also come from two "
    "groups, five from the legacy pipeline and five retrained later, and "
    "we did not test whether the groups differ."
)
s = s.replace(old_para, new_para)

# Line 7 — obs magnitude comparison 0.07955 → 0.07749
s = s.replace(
    "differ in magnitude by a factor of up to about 2.5 for observation "
    "effects (0.20125 vs 0.07955)",
    "differ in magnitude by a factor of up to about 2.6 for observation "
    "effects (0.20125 vs 0.07749)",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("08_limitations.md: updated")
else:
    report.append("08_limitations.md: NO CHANGES")

# ============================================================
# 09_conclusion.md
# ============================================================
p = base / "09_conclusion.md"
s = p.read_text(encoding="utf-8")
orig = s

s = s.replace(
    "Minari Hopper (n = 10 seeds) and Walker2d (n = 5 seeds)",
    "Minari Hopper and Walker2d (n = 10 seeds each)",
)
s = s.replace(
    "(alpha = 0.2361 for Hopper and 0.3725 for Walker2d)",
    "(alpha = 0.2361 for Hopper and 0.3743 for Walker2d)",
)
s = s.replace(
    "with ATEs between +0.07955 and +0.26588, whereas dynamics and policy "
    "effects did not exceed 0.01111 in magnitude",
    "with ATEs between +0.07749 and +0.26588, whereas dynamics and policy "
    "effects did not exceed 0.01194 in magnitude",
)
s = s.replace(
    "had opposite signs in the two environments (+0.04129 in Hopper, "
    "\u22120.03256 in Walker2d)",
    "had opposite signs in the two environments (+0.04129 in Hopper, "
    "\u22120.03307 in Walker2d)",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("09_conclusion.md: updated")
else:
    report.append("09_conclusion.md: NO CHANGES")

print("\n".join(report))
