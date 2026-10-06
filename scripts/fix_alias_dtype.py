"""Fix apply_observation_shift to preserve obs.dtype (no forced float32)."""
from pathlib import Path

path = Path('src/interventions/observation.py')
content = path.read_text(encoding='utf-8')

# Find and replace the bad alias section
old_marker = '# ---------------------------------------------------------------------------\n# Backward-compatibility'
if old_marker in content:
    content = content.split(old_marker)[0].rstrip() + '\n'

aliases = '''

# ---------------------------------------------------------------------------
# Backward-compatibility aliases for tests/test_observation_shift.py
# ---------------------------------------------------------------------------

def draw_base_noise(obs_shape, seed):
    """Deterministic standard-normal draw given an integer seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=obs_shape)


def apply_observation_shift(obs, eps, intensity, reference_scale):
    """Add scaled noise to observations.

    shifted = obs + eps * intensity * reference_scale

    Preserves obs.dtype. Rejects shape mismatch and negative intensity.
    """
    obs = np.asarray(obs)
    eps = np.asarray(eps)
    if obs.shape != eps.shape:
        raise ValueError(f'shape mismatch: obs {obs.shape} vs eps {eps.shape}')
    if intensity < 0:
        raise ValueError('intensity must be non-negative')
    result = obs + eps.astype(obs.dtype) * float(intensity) * reference_scale
    return result.astype(obs.dtype)
'''

path.write_text(content + aliases, encoding='utf-8', newline='\n')
print(f'updated {path}')
