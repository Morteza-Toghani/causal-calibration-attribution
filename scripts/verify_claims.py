"""Verify the [VERIFY] claims in docs/paper1_draft/04_method.md.

Reads the actual source files and prints the code that resolves each
placeholder. This is a static inspection script; it does not run any
experiment. Use it to confirm what the code actually does.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEPARATOR = '=' * 78


def section(title: str):
    print()
    print(SEPARATOR)
    print(f'  {title}')
    print(SEPARATOR)


def print_source(func):
    """Print the source of a function."""
    try:
        src = inspect.getsource(func)
        print(src)
    except (OSError, TypeError) as e:
        print(f'  [could not inspect: {e}]')


def main():
    print('=' * 78)
    print('VERIFY CLAIMS — Paper 1 Method Section')
    print('=' * 78)

    # ======================================================================
    # §4.2 — Do evaluated intervals use Gaussian(mu, total var) or mixture quantiles?
    # ======================================================================
    section('§4.2 — Interval construction: Gaussian(mu, total_var) or mixture?')

    from src.calibration import regression as reg
    from src.model import ensemble_v2 as ens

    print('--- ensemble_v2.predict_ensemble (returns mu, var) ---')
    print_source(ens.predict_ensemble)

    print('--- calibration.regression.prediction_interval (how intervals are built) ---')
    print_source(reg.prediction_interval)

    print('--- calibration.regression.ensemble_moments (moment-matched aggregation) ---')
    print_source(reg.ensemble_moments)

    print()
    print('CONCLUSION:')
    print('  If prediction_interval uses `norm.ppf` on sqrt(var) and var is the')
    print('  moment-matched total variance from ensemble_moments, then intervals are')
    print('  Gaussian(mu, total_var) — NOT mixture quantiles.')

    # ======================================================================
    # §4.3 — Which split for D, O, P?
    # ======================================================================
    section('§4.3 — Split usage per mechanism (from final_n10_analysis.py)')

    from scripts import final_n10_analysis as fna
    print_source(fna.process_seed)

    print()
    print('CONCLUSION: see which of test_obs/cal_obs/probe_obs is used per mechanism.')

    # ======================================================================
    # §4.4 — Is std per-dim or scalar?
    # ======================================================================
    section('§4.4 — Observation noise scale: per-dim or scalar?')

    from src.interventions import observation as obs
    print_source(obs.compute_obs_scale)
    print_source(obs.make_observation_noise)

    print()
    print('CONCLUSION: `np.std(test_obs, axis=0)` is per-dim (axis=0 reduces over')
    print('  instances, keeping one std per feature). If axis=0 is used, it is')
    print('  per-dim; if no axis, it would be scalar.')

    # ======================================================================
    # §4.6 — Per-instance aggregation: over output dims first?
    # ======================================================================
    section('§4.6 — Instance-level calibration error: aggregation order')

    from src.calibration import instance as inst
    print_source(inst.instance_calibration_error)

    print()
    print('CONCLUSION: read the function. If `inside.mean(axis=1)` appears before')
    print('  `np.abs`, then per-instance aggregation averages over output dims first.')

    # ======================================================================
    # §4.8 — Bootstrap unit and n_resamples
    # ======================================================================
    section('§4.8 — Bootstrap resampling unit and n_resamples')

    from src.stats import intervals as st
    print_source(st.bootstrap_mean_ci)

    print()
    print('--- Calls to bootstrap_mean_ci in final_n10_analysis.py ---')
    src = Path('scripts/final_n10_analysis.py').read_text(encoding='utf-8')
    for i, line in enumerate(src.splitlines(), 1):
        if 'bootstrap_mean_ci(' in line:
            print(f'  line {i}: {line.strip()}')

    print()
    print('CONCLUSION: bootstrap operates on the input array `diffs`. The unit is')
    print('  whatever `diffs` contains. n_resamples = `n_boot` argument.')

    print()
    print(SEPARATOR)
    print('  END — manually read the printed code and confirm each [VERIFY]')
    print(SEPARATOR)


if __name__ == '__main__':
    main()