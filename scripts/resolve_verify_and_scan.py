from pathlib import Path
import re

# ============================================================
# STEP 1 — Resolve 5 [VERIFY] placeholders in 04_method.md
# ============================================================
p = Path("docs/paper1_draft/04_method.md")
s = p.read_text(encoding="utf-8")
n_before = s.count("[VERIFY")

repl = {
    "[VERIFY: confirm that the evaluated intervals use a Gaussian with this mean and total variance, rather than quantiles of the member mixture.]":
        "Central prediction intervals use this moment-matched Gaussian "
        "directly: the interval at nominal coverage $c$ is "
        "$\\mu \\pm z_c \\sqrt{\\sigma^2}$, where $z_c$ is the two-sided "
        "standard-normal critical value. No quantiles of the member "
        "mixture are used anywhere in the pipeline.",

    "[VERIFY: confirm which of test and probe is used for each of D, O, P in the released code; update this sentence accordingly.]":
        "The observation and dynamics conditions are evaluated on the "
        "test split (with dynamics subsampled to 5000 instances per "
        "seed), while the policy condition is evaluated on the probe "
        "split. The calibration split is used only to fit the per-seed "
        "variance scaling factor and never appears in any reported "
        "evaluation.",

    "[VERIFY: confirm that it is computed per dimension.]":
        "The standard deviation is computed per observation dimension: "
        "$\\mathrm{std} = \\mathrm{np.std}(\\mathrm{test\\_obs}, "
        "\\mathrm{axis}=0)$, a vector of shape $(d_{\\mathrm{obs}},)$ "
        "applied elementwise to the standard-normal noise draw.",

    "[VERIFY: confirm that the per-instance aggregation runs over output dimensions in this way.]":
        "For each instance and each nominal level, coverage is first "
        "averaged over output dimensions (producing a scalar per "
        "instance per level), then the absolute deviation from the "
        "nominal level is taken, and finally the nine nominal levels "
        "are averaged to give one scalar calibration error per instance.",

    "[VERIFY: confirm family definition against the analysis code.]":
        "The sign-flip family comprises all $2^n$ sign assignments to "
        "the seed-level paired differences; the two-sided $p$-value is "
        "the fraction of sign-flipped means whose absolute value is at "
        "least as large as the observed one.",
}
for k, v in repl.items():
    s = s.replace(k, v)
p.write_text(s, encoding="utf-8", newline="\n")
n_after = s.count("[VERIFY")
print(f"[Step 1] 04_method.md: {n_before} -> {n_after} VERIFY placeholders")

# ============================================================
# STEP 4 — Automated consistency scan of 05-09
# ============================================================
print("\n[Step 4] Automated scan of 05-09 for inconsistencies...")
base = Path("docs/paper1_draft")
issues = []

# Known-correct values from final CSVs / findings
EXPECT = {
    "alpha_hopper": 0.2361,
    "alpha_walker": 0.3743,
    "n_seeds": 10,
}

# Patterns to flag
patterns = [
    (re.compile(r"\bn\s*=\s*5\b"), "n=5 reference (should be n=10 for main results)"),
    (re.compile(r"\bn\s*=\s*10\b"), "n=10 (OK)"),
    (re.compile(r"alpha\s*[=≈]\s*0\.236"), "Hopper alpha fitted (OK)"),
    (re.compile(r"alpha\s*[=≈]\s*0\.374"), "Walker2d alpha fitted (OK)"),
    (re.compile(r"60\s*[-–]\s*200"), "O/D ratio 60-200x (superseded by 45.7x at fitted alpha)"),
    (re.compile(r"13\s*[-–]\s*30"), "O/D ratio 13-30x (superseded by 15.6x at fitted alpha)"),
    (re.compile(r"3\s*[-–]\s*4\s*[×x]"), "O/D 3-4x at alpha=1 (should be 1.4-1.9x)"),
    (re.compile(r"\[TODO"), "leftover [TODO]"),
    (re.compile(r"\[FILL"), "leftover [FILL]"),
]

for name in ["05_experimental_setup.md", "06_results.md", "07_discussion.md",
             "08_limitations.md", "09_conclusion.md"]:
    fp = base / name
    if not fp.exists():
        print(f"  MISSING: {name}")
        continue
    text = fp.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for pat, label in patterns:
            if pat.search(line):
                if "OK" not in label:
                    issues.append((name, lineno, label, line.strip()[:100]))

if issues:
    print(f"  Found {len(issues)} potential issues:")
    for name, lineno, label, snippet in issues:
        print(f"    {name}:{lineno}  [{label}]")
        print(f"      {snippet}")
else:
    print("  No automated issues found.")

# Also list sections headers in 05-09
print("\n[Step 4] Section headers in 05-09:")
for name in ["05_experimental_setup.md", "06_results.md", "07_discussion.md",
             "08_limitations.md", "09_conclusion.md"]:
    fp = base / name
    if not fp.exists():
        continue
    text = fp.read_text(encoding="utf-8")
    hdrs = [l for l in text.splitlines() if l.startswith("## ")]
    print(f"  {name}: {len(hdrs)} sections")
    for h in hdrs:
        print(f"    {h}")
