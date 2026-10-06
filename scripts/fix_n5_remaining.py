from pathlib import Path

base = Path("docs/paper1_draft")
report = []

# ============================================================
# 01_abstract.md
# ============================================================
p = base / "01_abstract.md"
s = p.read_text(encoding="utf-8")
orig = s
s = s.replace(
    "Minari `hopper/medium-v0` (n = 10 training seeds) and "
    "`walker2d/medium-v0` (n = 5)",
    "Minari `hopper/medium-v0` and `walker2d/medium-v0` "
    "(n = 10 training seeds each)",
)
if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("01_abstract.md: updated")
else:
    report.append("01_abstract.md: NO CHANGE")

# ============================================================
# 02_introduction.md
# ============================================================
p = base / "02_introduction.md"
s = p.read_text(encoding="utf-8")
orig = s
s = s.replace(
    "in both Hopper (n = 10 seeds) and Walker2d (n = 5 seeds)",
    "in both Hopper and Walker2d (n = 10 seeds each)",
)
if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("02_introduction.md: updated")
else:
    report.append("02_introduction.md: NO CHANGE")

# ============================================================
# 04_method.md
# ============================================================
p = base / "04_method.md"
s = p.read_text(encoding="utf-8")
orig = s

# Line 103 — t_crit line
s = s.replace(
    "with $t_{\\text{crit}} = 2.262$ for $n = 10$ and 2.776 for $n = 5$.",
    "with $t_{\\text{crit}} = 2.262$ for $n = 10$.",
)

# Line 105 — p-floor paragraph
s = s.replace(
    "For $n$ seeds, the exact test has $2^n$ equally likely sign assignments, "
    "so the smallest attainable two-sided $p$ is $2/2^n$: 0.0625 at $n = 5$ "
    "and 0.00195 at $n = 10$. At $n = 5$ the test therefore cannot reach "
    "$p < 0.05$ even when all seed effects share a sign, and any "
    "multiplicity correction makes this stricter.",
    "For $n$ seeds, the exact test has $2^n$ equally likely sign assignments, "
    "so the smallest attainable two-sided $p$ is $2/2^n$: 0.00195 at "
    "$n = 10$.",
)

# Line 105 — remove duplicate bootstrap paragraph (the one added earlier
# now appears twice: once as an inserted sentence, once as the original)
s = s.replace(
    "Bootstrap intervals over instances, where computed, were used as a "
    "secondary check. The bootstrap is a secondary check; the primary "
    "confidence intervals are the seed-level $t$-intervals reported in "
    "the results tables. Where shown, bootstrap confidence intervals "
    "resample paired instances (not seeds) and use 5000 resamples for "
    "the observation and policy conditions and 2000 for the dynamics "
    "condition.",
    "Bootstrap intervals over instances, where computed, were used as a "
    "secondary check; the primary confidence intervals are the seed-level "
    "$t$-intervals reported in the results tables. Where shown, bootstrap "
    "confidence intervals resample paired instances (not seeds) and use "
    "5000 resamples for the observation and policy conditions and 2000 "
    "for the dynamics condition.",
)

# Line 107 — "with n = 5 or n = 10"
s = s.replace(
    "which we cannot verify with $n = 5$ or $n = 10$.",
    "which we cannot verify at $n = 10$.",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("04_method.md: updated")
else:
    report.append("04_method.md: NO CHANGE")

print("\n".join(report))
