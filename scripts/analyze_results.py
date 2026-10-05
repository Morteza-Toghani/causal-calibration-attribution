"""Rebuild every table and figure in results/ from results/raw/.

    python scripts/analyze_results.py

Inputs : results/raw/hopper_seeds_0_4/{hopper_seeds_0_to_4_results.csv, seed_0?_summary.json}
Outputs: results/tables/*.csv|md, results/figures/*.png

Evidence status of everything produced here: PRELIMINARY (one environment, 5 seeds,
original experiment code not yet migrated into this repository).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.causal.attribution import seed_level_effects  # noqa: E402
from src.stats.multi_seed_runner import summarize_seed_effects  # noqa: E402

RAW = ROOT / "results" / "raw" / "hopper_seeds_0_4"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
ORDER = ["dynamics_low", "dynamics_high", "observation_low", "observation_high", "policy_low", "policy_high"]
COLORS = {"dynamics": "#1b6ca8", "observation": "#c0392b", "policy": "#2e8b57"}
STATUS = "PRELIMINARY - Hopper (medium), 5 seeds"


def verify_hashes() -> None:
    for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*").strip()
        actual = hashlib.sha256((RAW / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"hash mismatch for {name}")


def load_json() -> dict[int, dict]:
    return {
        int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in sorted(RAW.glob("seed_0?_summary.json"))
    }


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append(
            "| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    verify_hashes()
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / "hopper_seeds_0_to_4_results.csv")
    js = load_json()

    # 1. Definition check: calibration_error == mean_p |pooled empirical - nominal|
    for s, d in js.items():
        for curve, ce in [
            (d["baseline_metrics"]["calibration_curve"], d["baseline_metrics"]["calibration_error"])
        ] + [
            (i["metrics"]["calibration_curve"], i["metrics"]["calibration_error"]) for i in d["interventions"]
        ]:
            recomputed = np.mean([abs(c["empirical"] - c["nominal"]) for c in curve])
            assert abs(recomputed - ce) < 1e-9, (s, recomputed, ce)

    # 2. Seed-level ATE as reported by the original run (pair-level ATE) -> inference over seeds
    ate = df[df.condition != "baseline"][["seed_idx", "condition", "ate"]].copy()
    ate["mechanism"] = ate.condition.str.split("_").str[0]
    ate["intensity"] = ate.condition.str.split("_").str[1]
    ate = ate.rename(columns={"seed_idx": "seed", "ate": "effect"})
    ate.sort_values(["condition", "seed"]).to_csv(TAB / "seed_level_ate_reported.csv", index=False)
    summ = summarize_seed_effects(ate, n_boot=10_000, seed=0)
    summ["condition"] = pd.Categorical(summ["condition"], ORDER, ordered=True)
    summ = summ.sort_values("condition").reset_index(drop=True)
    summ.to_csv(TAB / "mechanism_attribution_summary.csv", index=False)

    # 3. Aggregate-metric differences (treatment - baseline, per seed) and signed coverage bias
    rows = []
    for s, d in js.items():
        b = d["baseline_metrics"]
        b_bias = np.mean([c["empirical"] - c["nominal"] for c in b["calibration_curve"]])
        rows.append(
            {
                "seed": s,
                "condition": "baseline",
                "calibration_error": b["calibration_error"],
                "signed_bias": b_bias,
                "coverage_50": b["coverage_50"],
                "coverage_90": b["coverage_90"],
                "nll": b["nll"],
            }
        )
        for i in d["interventions"]:
            m = i["metrics"]
            rows.append(
                {
                    "seed": s,
                    "condition": f"{i['mechanism']}_{i['intensity']}",
                    "calibration_error": m["calibration_error"],
                    "signed_bias": np.mean([c["empirical"] - c["nominal"] for c in m["calibration_curve"]]),
                    "coverage_50": m["coverage_50"],
                    "coverage_90": m["coverage_90"],
                    "nll": m["nll"],
                }
            )
    agg = pd.DataFrame(rows)
    agg.to_csv(TAB / "aggregate_metrics_by_seed.csv", index=False)
    mean_tab = agg.groupby("condition")[
        ["calibration_error", "signed_bias", "coverage_50", "coverage_90", "nll"]
    ].mean()
    sd_tab = agg.groupby("condition")[["calibration_error", "signed_bias"]].std(ddof=1)
    mean_tab["calibration_error_sd"] = sd_tab["calibration_error"]
    mean_tab["signed_bias_sd"] = sd_tab["signed_bias"]
    mean_tab = mean_tab.reindex(["baseline"] + ORDER)
    mean_tab.reset_index().to_csv(TAB / "aggregate_metrics_mean_over_seeds.csv", index=False)

    # consistency between pair-level ATE and aggregate difference (documented discrepancy)
    diff = seed_level_effects(agg.rename(columns={"seed": "seed_idx"}), "calibration_error")
    cmp_ = (
        diff.groupby("condition")["effect"]
        .mean()
        .rename("mean_aggregate_difference")
        .to_frame()
        .join(summ.set_index("condition")["mean_effect"].rename("mean_reported_ate"))
        .reindex(ORDER)
    )
    cmp_["gap"] = cmp_["mean_reported_ate"] - cmp_["mean_aggregate_difference"]
    cmp_.reset_index().to_csv(TAB / "ate_vs_aggregate_difference.csv", index=False)

    # ---- markdown tables (embedded verbatim in the README) ----
    t1 = summ[
        [
            "condition",
            "n_seeds",
            "mean_effect",
            "sd_across_seeds",
            "boot_ci_low",
            "boot_ci_high",
            "cohens_dz",
            "n_positive_seeds",
            "p_signflip",
            "p_holm",
        ]
    ].copy()
    t1["n_positive_seeds"] = (
        t1["n_positive_seeds"].astype(int).astype(str) + "/" + t1["n_seeds"].astype(int).astype(str)
    )
    t1 = t1.drop(columns="n_seeds")
    t1.columns = [
        "condition",
        "mean ATE",
        "SD (seeds)",
        "95% boot CI low",
        "95% boot CI high",
        "d_z",
        "ATE>0 seeds",
        "p (sign-flip)",
        "p (Holm)",
    ]
    (TAB / "mechanism_attribution_summary.md").write_text(md_table(t1))
    t2 = mean_tab.reset_index()[
        [
            "condition",
            "calibration_error",
            "calibration_error_sd",
            "signed_bias",
            "coverage_50",
            "coverage_90",
            "nll",
        ]
    ]
    t2.columns = ["condition", "cal. error", "SD (seeds)", "signed bias", "cov@50", "cov@90", "NLL"]
    (TAB / "aggregate_metrics_mean_over_seeds.md").write_text(md_table(t2))
    (TAB / "ate_vs_aggregate_difference.md").write_text(md_table(cmp_.reset_index(), "{:.4f}"))

    # ---- Figure A: forest plot of seed-level ATE ----
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for yi, cond in enumerate(ORDER[::-1]):
        r = summ[summ.condition == cond].iloc[0]
        vals = ate[ate.condition == cond]["effect"].to_numpy()
        col = COLORS[r.mechanism]
        ax.scatter(vals, np.full(vals.size, yi), s=14, color=col, alpha=0.45, zorder=2)
        ax.errorbar(
            r.mean_effect,
            yi,
            xerr=[[r.mean_effect - r.boot_ci_low], [r.boot_ci_high - r.mean_effect]],
            fmt="o",
            color=col,
            capsize=3,
            zorder=3,
            markersize=6,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([c.replace("_", " ") for c in ORDER[::-1]])
    ax.set_xlabel("ATE on calibration error (treatment - baseline), as reported per seed")
    ax.set_title("Mechanism-specific effect on calibration error\n" + STATUS, fontsize=10)
    ax.text(
        0.01,
        -0.28,
        "Dots: individual seeds. Marker + bar: mean and 95% percentile-bootstrap CI over seeds (n=5).\n"
        "Calibration error is unsigned: ATE<0 does not imply coverage closer to nominal in sign.",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig_mechanism_attribution_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure B: calibration curves (mean over seeds) ----
    levels = [c["nominal"] for c in js[0]["baseline_metrics"]["calibration_curve"]]

    def mean_curve(getter):
        return np.mean([[c["empirical"] for c in getter(js[s])] for s in js], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)
    base_curve = mean_curve(lambda d: d["baseline_metrics"]["calibration_curve"])
    for ax, mech in zip(axes, ["dynamics", "observation", "policy"]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        ax.plot(levels, base_curve, color="gray", marker="o", ms=3, label="baseline")
        for inten, ls in [("low", ":"), ("high", "-")]:
            cv = mean_curve(
                lambda d: next(
                    i for i in d["interventions"] if i["mechanism"] == mech and i["intensity"] == inten
                )["metrics"]["calibration_curve"]
            )
            ax.plot(levels, cv, color=COLORS[mech], ls=ls, marker="o", ms=3, label=f"{mech} {inten}")
        ax.set_title(mech, fontsize=10)
        ax.set_xlabel("nominal coverage")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("empirical coverage (pooled over dimensions)")
    fig.suptitle("Calibration curves, mean over seeds - " + STATUS, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("OK: tables ->", TAB, "| figures ->", FIG)
    print(t1.to_string(index=False))


if __name__ == "__main__":
    main()
