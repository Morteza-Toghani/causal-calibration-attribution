"""Append backward-compatibility aliases to src/interventions/observation.py.

The old test file tests/test_observation_shift.py imports the legacy names
apply_observation_shift and draw_base_noise. We keep those names as thin
wrappers around the new FROZEN-semantics functions.
"""
from pathlib import Path

path = Path('src/interventions/observation.py')
content = path.read_text(encoding='utf-8')

if 'def draw_base_noise' in content:
    print('aliases already present, skipping')
else:
    compat = '''


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (legacy API used by tests/test_observation_shift.py)
# ---------------------------------------------------------------------------

def draw_base_noise(obs_shape, rng):
    """Legacy alias: return a standard-normal draw of the given shape."""
    return rng.standard_normal(size=obs_shape).astype(np.float32)


def apply_observation_shift(obs, noise_fraction, reference_scale, rng):
    """Legacy wrapper: apply additive standardized noise to observations.

    Returns the perturbed observation. Equivalent to:
        eps = make_observation_noise(obs.shape, reference_scale, noise_fraction, rng)
        return apply_observation_noise(obs, eps)
    """
    eps = make_observation_noise(obs.shape, reference_scale, noise_fraction, rng)
    return apply_observation_noise(obs, eps)
'''
    path.write_text(content + compat, encoding='utf-8', newline='\n')
    print(f'added compatibility aliases to {path}')
