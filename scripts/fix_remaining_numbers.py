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
    "(alpha = 0.2361 for Hopper, 0.3725 for Walker2d)",
    "(alpha = 0.2361 for Hopper, 0.3743 for Walker2d)",
)
s = s.replace(
    "ATES were +0.20125 and +0.26588 (Hopper) and +0.07955 and +0.15449 "
    "(Walker2d), whereas dynamics and policy effects were at most 0.01111 "
    "in magnitude.",
    "ATES were +0.20125 and +0.26588 (Hopper) and +0.07749 and +0.15268 "
    "(Walker2d), whereas dynamics and policy effects were at most 0.01194 "
    "in magnitude.",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("01_abstract.md: updated")
else:
    report.append("01_abstract.md: NO CHANGE")

# ============================================================
# 06_results.md
# ============================================================
p = base / "06_results.md"
s = p.read_text(encoding="utf-8")
orig = s

s = s.replace(
    "and $\\alpha = 0.3725$ for Walker2d, implying about 2.7 times too large",
    "and $\\alpha = 0.3743$ for Walker2d, implying about 2.7 times too large",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("06_results.md: updated")
else:
    report.append("06_results.md: NO CHANGE")

# ============================================================
# paper1_findings.md — mark n=5 script as historical in reproduction list
# ============================================================
p = Path("docs/paper1_findings.md")
s = p.read_text(encoding="utf-8")
orig = s

s = s.replace(
    "python scripts/recalibrate_and_rerun.py                # recalibration (n=5)",
    "python scripts/recalibrate_and_rerun.py                # recalibration (n=5, historical)",
)

if s != orig:
    p.write_text(s, encoding="utf-8", newline="\n")
    report.append("paper1_findings.md: updated")
else:
    report.append("paper1_findings.md: NO CHANGE")

print("\n".join(report))
