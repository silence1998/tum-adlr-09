"""
Plain single-task SAC baseline (GPU vectorised env), sharing the implementation of training_SAC-X.py:
only the main intention (--tasks 0), 1-step soft TD targets (--seq-len 1, i.e. no Retrace), twin critics,
automatic entropy tuning, target critics soft-updated every gradient step, same networks / input features /
replay / update schedule as the SAC-X runs.  Any extra CLI argument is forwarded (e.g. --seed, --total-steps).

    python training_SAC.py --seed 0 --total-steps 300000 --log ../results/new_env/runs/sac_s0.csv
"""
import os
import runpy
import sys

if __name__ == "__main__":
    # Defaults validated with debug_min_sac.py (see results/new_env/RESULTS_new_env.md, "SAC overestimation"):
    # replay ratio = batch * updates / num_envs = 32 sampled transitions per collected transition (the earlier
    # 2048 * 4 / 32 = 256 made the critic overfit and the actor exploit its errors -> Q(s0) ~10 vs true ~0),
    # distance-sorted obstacle slots, env reward scale 1, initial alpha 0.05.
    defaults = ["--tasks", "0", "--seq-len", "1", "--batch", "256", "--num-envs", "8", "--updates-per-step", "1",
                "--features", "relative_sorted", "--main-reward-scale", "1.0", "--alpha-init", "0.05",
                "--warmup", "5000"]
    sys.argv = [sys.argv[0]] + defaults + sys.argv[1:]  # later CLI args override the defaults
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_SAC-X.py"), run_name="__main__")
