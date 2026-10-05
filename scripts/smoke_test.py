"""
Phase 0 — Environment + Data Smoke Test

Verifies that the core scientific-computing and RL stack required for the
project (Counterfactual Causal Attribution of Calibration Failure in
Uncertainty-Aware World Models) is correctly installed and importable.

Run:
    python scripts/smoke_test.py
"""

import sys
import traceback


def check(name, fn):
    try:
        result = fn()
        extra = f" ({result})" if result else ""
        print(f"{name:<14}OK{extra}")
        return True
    except Exception as e:
        print(f"{name:<14}FAIL - {e}")
        traceback.print_exc()
        return False


def check_python():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, f"Python 3.10+ required, found {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def check_torch():
    import torch
    x = torch.randn(3, 3)
    y = x @ x
    assert y.shape == (3, 3)
    return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}"


def check_gymnasium():
    import gymnasium
    return f"gymnasium {gymnasium.__version__}"


def check_mujoco():
    import mujoco
    return f"mujoco {mujoco.__version__}"


def check_minari():
    import minari
    return f"minari {minari.__version__}"


def check_scipy():
    import scipy
    return f"scipy {scipy.__version__}"


def check_pandas():
    import pandas
    return f"pandas {pandas.__version__}"


def check_matplotlib():
    import matplotlib
    return f"matplotlib {matplotlib.__version__}"


def check_env(env_id):
    def _fn():
        import gymnasium as gym
        env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        env.close()
        return f"obs_dim={obs.shape}, action_dim={env.action_space.shape}"
    return _fn


def main():
    print("=== Environment Smoke Test ===\n")

    results = []
    results.append(check("Python:", check_python))
    results.append(check("PyTorch:", check_torch))
    results.append(check("Gymnasium:", check_gymnasium))
    results.append(check("MuJoCo:", check_mujoco))
    results.append(check("Minari:", check_minari))
    results.append(check("SciPy:", check_scipy))
    results.append(check("Pandas:", check_pandas))
    results.append(check("Matplotlib:", check_matplotlib))

    print()
    results.append(check("Hopper-v5:", check_env("Hopper-v5")))
    results.append(check("Walker2d-v5:", check_env("Walker2d-v5")))

    print()
    if all(results):
        print("Environment smoke test passed.")
        sys.exit(0)
    else:
        print("Environment smoke test FAILED. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
