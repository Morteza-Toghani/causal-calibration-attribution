"""
Phase 3.3 — Select & Freeze baseline (Hopper, seed 1).

Protocol (declared BEFORE looking at numbers):
  1. Evaluate candidates A-D on the CALIBRATION split (never TEST).
  2. Sanity gate: lower AND upper log-var saturation < 1% in every dimension.
  3. Among gated configs: lowest calibration_error wins, but any config within
     NOISE_MARGIN of the best is considered tied. Ties -> fewest epochs,
     then lower CRPS, then lower NLL.
  4. Write frozen_baseline.json (config, hashes, git commit) BEFORE touching TEST.
  5. Run exactly ONE TEST evaluation -> metrics.json + calibration curve.
     Refuses to run if a TEST evaluation was already recorded.

Status is PRELIMINARY-FROZEN: A-C were seen on TEST during diagnostics.

Usage (from repo root):
    python experiments/pilot/select_and_freeze.py      # select + freeze + test
    python experiments/pilot/select_and_freeze.py --dry-run  # select only, no freeze/test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calibration.regression import (
    ensemble_moments,
    evaluate_regression_calibration,
    interval_coverage,
    prediction_interval,
)
from src.data.loader import TransitionBatch
from src.model.ensemble import EnsembleWorldModel

# ----------------------------------------------------------------------------
# EDIT THIS: checkpoint dirs for each candidate.
# min_log_var is declared here because ensemble.py's meta.json does not store it.
# ----------------------------------------------------------------------------
DATA_DIR = Path("data/processed/hopper_seed1")
OUT_DIR = Path("results/baseline/hopper_seed1_frozen")
MAX_LOG_VAR = 2.0

CONFIGS = {
    "A": {"dir": "results/models/hopper_seed1_A_mlv-10_ep20", "min_log_var": -10.0, "epochs": 20},
    "B": {"dir": "results/models/hopper_seed1_B_mlv-10_ep50", "min_log_var": -10.0, "epochs": 50},
    "C": {"dir": "results/models/hopper_seed1_C_mlv-20_ep20", "min_log_var": -20.0, "epochs": 20},
    "D": {"dir": "results/models/hopper_seed1_D_mlv-20_ep50", "min_log_var": -20.0, "epochs": 50},
}

NOISE_MARGIN = 0.005      # calibration_error differences below this = tie
SAT_THRESHOLD = 0.01      # 1% per dimension
LEVELS = (0.50, 0.90)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty_working_tree": dirty}
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "error": str(e)}


def load_model(cfg: dict) -> EnsembleWorldModel:
    """Load checkpoint and FORCE the declared clamp bounds on every member.

    ensemble.py does not save min_log_var, so load() silently uses the default
    (-10). For -20 checkpoints that would be a silent bug.
    """
    model = EnsembleWorldModel.load(cfg["dir"])
    for m in model.members:
        m.min_log_var = cfg["min_log_var"]
        m.max_log_var = MAX_LOG_VAR
        m.eval()
    return model


@torch.no_grad()
def member_moments_and_saturation(model, obs, action):
    """Return member mus/vars [K,N,D] and raw-log-var saturation [D] (lower, upper)."""
    o = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    a = torch.as_tensor(action, dtype=torch.float32, device=model.device)
    mus, vars_, lo_sat, hi_sat = [], [], [], []
    for m in model.members:
        h = m.trunk(torch.cat([o, a], dim=-1))
        raw = m.log_var_head(h)
        lo_sat.append((raw <= m.min_log_var).float().mean(0).cpu().numpy())
        hi_sat.append((raw >= m.max_log_var).float().mean(0).cpu().numpy())
        mu, s2 = m(o, a)
        mus.append(mu.cpu().numpy())
        vars_.append(s2.cpu().numpy())
    return (
        np.stack(mus), np.stack(vars_),
        np.mean(lo_sat, axis=0), np.mean(hi_sat, axis=0),
    )


def evaluate(model, batch: TransitionBatch) -> dict:
    mus, vars_, lo, hi = member_moments_and_saturation(model, batch.observations, batch.actions)
    y = batch.next_observations
    res = evaluate_regression_calibration(y, mus, vars_, LEVELS)
    mean, var = ensemble_moments(mus, vars_)
    z = (y - mean) / np.sqrt(var)
    res["std_z"] = float(np.std(z))
    res["std_z_per_dimension"] = np.std(z, axis=0).tolist()
    res["lower_saturation_per_dimension"] = lo.tolist()
    res["upper_saturation_per_dimension"] = hi.tolist()
    res["max_saturation"] = float(max(lo.max(), hi.max()))
    res["sanity_pass"] = bool(max(lo.max(), hi.max()) < SAT_THRESHOLD)
    res["_mean"], res["_var"] = mean, var  # stripped before saving
    return res


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def calibration_curve(y, mean, var, levels=np.linspace(0.05, 0.95, 19)):
    emp = []
    for q in levels:
        lo, hi = prediction_interval(mean, var, float(q))
        emp.append(float(np.mean(interval_coverage(y, lo, hi))))
    return levels.tolist(), emp


def save_curve(levels, emp, path_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="ideal")
    ax.plot(levels, emp, "o-", label="frozen baseline (TEST)")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def select(results: dict) -> tuple[str, list[str], str]:
    passed = [k for k, r in results.items() if r["sanity_pass"]]
    if not passed:
        raise SystemExit("No config passed the saturation gate. Do NOT freeze; investigate.")
    best_err = min(results[k]["calibration_error"] for k in passed)
    tied = [k for k in passed if results[k]["calibration_error"] - best_err <= NOISE_MARGIN]
    winner = min(
        tied,
        key=lambda k: (
            CONFIGS[k]["epochs"], results[k]["crps_mean"], results[k]["nll_mean"],
        ),
    )
    reason = (
        f"gate passed: {passed}; best cal_error={best_err:.4f}; "
        f"within {NOISE_MARGIN} of best: {tied}; "
        f"tie-break (fewest epochs, CRPS, NLL) -> {winner}"
    )
    return winner, passed, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT_DIR / "frozen_baseline.json"
    if frozen_path.exists() and json.load(open(frozen_path)).get("test_evaluated"):
        raise SystemExit("TEST already evaluated for this baseline. Refusing to run again.")

    cal = TransitionBatch.load(DATA_DIR / "calibration.npz")
    print(f"calibration split: {len(cal)} transitions\n")

    results, saved = {}, {}
    for name, cfg in CONFIGS.items():
        model = load_model(cfg)
        r = evaluate(model, cal)
        results[name] = r
        saved[name] = strip(r)
        print(
            f"[{name}] mlv={cfg['min_log_var']:>5} ep={cfg['epochs']:>2} | "
            f"cal_err={r['calibration_error']:.4f} crps={r['crps_mean']:.5f} "
            f"nll={r['nll_mean']:.4f} std_z={r['std_z']:.3f} "
            f"max_sat={r['max_saturation']*100:.2f}% gate={'PASS' if r['sanity_pass'] else 'FAIL'}"
        )

    winner, passed, reason = select(results)
    print(f"\nSELECTED: {winner}\n{reason}")

    with open(OUT_DIR / "selection_calibration_split.json", "w") as f:
        json.dump({"rule": {"noise_margin": NOISE_MARGIN, "sat_threshold": SAT_THRESHOLD},
                   "per_config": saved, "selected": winner, "reason": reason}, f, indent=2)

    if args.dry_run:
        print("\n--dry-run: stopping before freeze/TEST.")
        return

    # ---- LOCK -------------------------------------------------------------
    ckpt_dir = Path(CONFIGS[winner]["dir"])
    files = sorted(ckpt_dir.glob("member_*.pt")) + [ckpt_dir / "meta.json"]
    frozen = {
        "status": "PRELIMINARY-FROZEN",
        "note": ("Selected on calibration split, but configs A-C were previously "
                 "inspected on TEST during diagnostics (Phase 3.1/3.2)."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_config": winner,
        "config": CONFIGS[winner],
        "max_log_var": MAX_LOG_VAR,
        "selection_reason": reason,
        "training_seed": 1,
        "dataset_id": "mujoco/hopper/medium-v0",
        "data_manifest_sha256": sha256(DATA_DIR / "manifest.json"),
        "checkpoint_files": {str(p): sha256(p) for p in files},
        "git": git_commit(),
        "test_evaluated": False,
    }
    with open(frozen_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nLocked -> {frozen_path}")

    # ---- ONE TEST evaluation ---------------------------------------------
    test = TransitionBatch.load(DATA_DIR / "test.npz")
    model = load_model(CONFIGS[winner])
    r = evaluate(model, test)
    levels, emp = calibration_curve(test.next_observations, r["_mean"], r["_var"])

    report = {"evaluation_split": "test", "selected_config": winner,
              "status": "PRELIMINARY-FROZEN", "prediction_horizon": 1,
              **strip(r), "calibration_curve": {"nominal": levels, "empirical": emp}}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    save_curve(levels, emp, OUT_DIR / "calibration_curve.png")

    frozen["test_evaluated"] = True
    frozen["test_evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(frozen_path, "w") as f:
        json.dump(frozen, f, indent=2)

    print(f"TEST: cal_err={r['calibration_error']:.4f} nll={r['nll_mean']:.4f} "
          f"crps={r['crps_mean']:.5f} std_z={r['std_z']:.3f}")
    print(f"Saved metrics.json + calibration_curve.png in {OUT_DIR}")


if __name__ == "__main__":
    main()
