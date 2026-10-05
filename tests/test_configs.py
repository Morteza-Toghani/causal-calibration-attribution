from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(p):
    return yaml.safe_load((ROOT / p).read_text())


def test_all_configs_parse():
    files = list((ROOT / "configs").rglob("*.yaml"))
    assert len(files) >= 6
    for f in files:
        assert isinstance(yaml.safe_load(f.read_text()), dict), f


def test_experiment_configs_reference_existing_files():
    for name in ["phase_A", "phase_B"]:
        cfg = _load(f"configs/experiment/{name}.yaml")
        for key in ("model_config", "raw_results", "shift_configs"):
            for ref in [cfg[key]] if isinstance(cfg.get(key), str) else cfg.get(key, []):
                assert (ROOT / ref).exists(), (name, ref)


def test_phase_A_splits_add_up_to_documented_totals():
    s = _load("configs/experiment/phase_A.yaml")["splits"]
    assert sum(v["episodes"] for v in s.values()) == 1327
    assert sum(v["transitions"] for v in s.values()) == 999404


def test_observation_intensities_match_raw_results():
    import json

    cfg = _load("configs/shift/observation.yaml")["intensities"]
    d = json.loads((ROOT / "results/raw/hopper_seeds_0_4/seed_00_summary.json").read_text())
    raw = {i["intensity"]: i["noise_fraction"] for i in d["interventions"] if i["mechanism"] == "observation"}
    assert raw == {k: v["noise_fraction"] for k, v in cfg.items()}
