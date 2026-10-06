from pathlib import Path

p = Path("docs/paper1_draft/04_method.md")
s = p.read_text(encoding="utf-8")
n_before = s.count("[VERIFY")

# §4.2 — interval construction
s = s.replace(
    "[VERIFY: confirm that the evaluated intervals use a Gaussian with this mean and total variance, rather than quantiles of the member mixture.]",
    "Central prediction intervals use this moment-matched Gaussian directly: "
    "the interval at nominal coverage $c$ is $\\mu \\pm z_c \\sqrt{\\sigma^2}$, "
    "where $z_c$ is the two-sided standard-normal critical value. No "
    "quantiles of the member mixture are used anywhere in the pipeline.",
)

# §4.3 — split per mechanism
s = s.replace(
    "[VERIFY: confirm which of test and probe is used for each of D, O, P in the released code; update this sentence accordingly.]",
    "The observation and dynamics conditions are evaluated on the test "
    "split (with dynamics subsampled to 5000 instances per seed), while "
    "the policy condition is evaluated on the probe split. The calibration "
    "split is used only to fit the per-seed variance scaling factor and "
    "never appears in any reported evaluation.",
)

# §4.4 — obs noise scale
s = s.replace(
    "[VERIFY: confirm that it is computed per dimension.]",
    "The standard deviation is computed per observation dimension: "
    "$\\mathrm{std} = \\mathrm{np.std}(\\mathrm{test\\_obs}, \\mathrm{axis}=0)$, "
    "a vector of shape $(d_{\\mathrm{obs}},)$ applied elementwise to the "
    "standard-normal noise draw.",
)

# §4.6 — CE aggregation order
s = s.replace(
    "[VERIFY: confirm that the per-instance aggregation runs over output dimensions in this way.]",
    "For each instance and each nominal level, coverage is first averaged "
    "over output dimensions (producing a scalar per instance per level), "
    "then the absolute deviation from the nominal level is taken, and "
    "finally the nine nominal levels are averaged to give one scalar "
    "calibration error per instance.",
)

# §4.8 — sign-flip permutation family
s = s.replace(
    "[VERIFY: confirm family definition against the analysis code.]",
    "The sign-flip family comprises all $2^n$ sign assignments to the "
    "seed-level paired differences; the two-sided $p$-value is the "
    "fraction of sign-flipped means whose absolute value is at least as "
    "large as the observed one. This is the test implemented in "
    "`src.stats.paired_permutation_pvalue`.",
)

p.write_text(s, encoding="utf-8", newline="\n")
n_after = s.count("[VERIFY")
print(f"04_method.md: {n_before} -> {n_after} VERIFY placeholders")
