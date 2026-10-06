"""Replace the observation.py aliases with versions matching the test API."""
from pathlib import Path

path = Path('src/interventions/observation.py')
content = path.read_text(encoding='utf-8')

# Cut off any existing alias section
marker = '# ---------------------------------------------------------------------------\n# Backward-compatibility'
if marker in content:
    content = content.split(marker)[0].rstrip() + '\n'

# Append correct aliases
aliases = '''

# ---------------------------------------------------------------------------
# Backward-compatibility aliases for tests/test_observation_shift.py
# ---------------------------------------------------------------------------

def draw_base_noise(obs_shape, seed):
    """Deterministic standard-normal draw given an integer seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=obs_shape).astype(np.float32)


def apply_observation_shift(obs, eps, intensity, reference_scale):
    """Add scaled noise to observations.

    shifted = obs + eps * intensity * reference_scale

    Rejects shape mismatch and negative intensity.
    """
    obs = np.asarray(obs)
    eps = np.asarray(eps)
    if obs.shape != eps.shape:
        raise ValueError(f'shape mismatch: obs {obs.shape} vs eps {eps.shape}')
    if intensity < 0:
        raise ValueError('intensity must be non-negative')
    return (obs + eps * float(intensity) * reference_scale).astype(np.float32)
'''

path.write_text(content + aliases, encoding='utf-8', newline='\n')
print(f'updated {path}')
