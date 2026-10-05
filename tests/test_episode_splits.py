import numpy as np
import pytest

from src.data.splits import episode_split, transitions_for_episodes


def test_every_episode_in_exactly_one_split():
    s = episode_split(1327, seed=0)
    allv = np.concatenate(list(s.values()))
    assert len(allv) == 1327 and len(set(allv.tolist())) == 1327


def test_split_is_reproducible_and_seed_dependent():
    a, b, c = episode_split(200, seed=1), episode_split(200, seed=1), episode_split(200, seed=2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["train"], c["train"])


def test_no_transition_leakage_between_splits():
    lengths = np.random.default_rng(0).integers(5, 50, size=100)
    s = episode_split(100, seed=3)
    sets = [set(transitions_for_episodes(lengths, e).tolist()) for e in s.values()]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])
    assert sum(len(x) for x in sets) == lengths.sum()


def test_invalid_fractions_raise():
    with pytest.raises(ValueError):
        episode_split(100, {"a": 0.5, "b": 0.6})
    with pytest.raises(ValueError):
        episode_split(2, seed=0)
