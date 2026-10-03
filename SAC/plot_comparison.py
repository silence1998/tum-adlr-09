"""
Comparison plot + summary table from the CSV logs written by training_SAC-X.py, training_PPO.py and
baseline_original/run_original_sacx.py.

    python plot_comparison.py --runs ../results/runs --out ../results/comparison.png
Each algorithm is a file prefix: <prefix>_s<seed>.csv (e.g. ppo_s0.csv, sacx_q_s1.csv, orig_s2.csv).
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "orig": "original 'SAC-X' script (plain SAC, unfixed)",
    "sacx_q": "SAC-X fixed (SAC-Q scheduler, 6 intentions)",
    "sacx_u": "SAC-X fixed (SAC-U scheduler, 6 intentions)",
    "sac_main": "SAC + Retrace, main task only (ablation)",
    "ppo": "PPO (GPU vectorised env)",
    "sac": "SAC (single task, twin critics, auto-alpha)",
    "ppo_shaped": "PPO, modified env (shaping + time penalty + collision -10)",
    "sacx_q_shaped": "SAC-X fixed (SAC-Q), modified env",
    "ppo_sorted": "PPO, distance-sorted obstacle input",
}


def load(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows = [list(map(lambda x: float(x) if x not in ("", "nan") else np.nan, l.strip().split(",")))
                for l in f if l.strip()]
    a = np.array(rows)
    return {h: a[:, i] for i, h in enumerate(header)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/runs")
    ap.add_argument("--out", default="../results/comparison.png")
    ap.add_argument("--table", default="../results/summary_table.md")
    ap.add_argument("--return-label", default="eval return (original reward: +10 goal, -50 collision)")
    ap.add_argument("--at", type=int, nargs="*", default=[100_000],
                    help="extra table columns: deterministic success at these env steps (matching budgets)")
    ap.add_argument("--logx", action="store_true", help="also write a *_logx.png copy with a log x-axis")
    ap.add_argument("--xmax", type=int, default=None, help="optional x-axis limit of the plot")
    args = ap.parse_args()

    groups = {}
    for p in sorted(glob.glob(os.path.join(args.runs, "*_s*.csv"))):
        m = re.match(r"(.+)_s(\d+)\.csv$", os.path.basename(p))
        if m:
            groups.setdefault(m.group(1), []).append(load(p))

    fig, axes = plt.subplots(1, 4, figsize=(23, 5))
    lines = ["| algorithm | seeds | env steps (max) | final eval success | final eval return | "
             + " | ".join(f"eval success @{a // 1000}k" for a in args.at) +
             " | best eval success | final stochastic-policy success | final collision rate | wall time / run |",
             "|---|---|---|---|---|" + "---|" * len(args.at) + "---|---|---|---|"]
    for name, runs in groups.items():
        n = min(len(r["env_steps"]) for r in runs)
        x = runs[0]["env_steps"][:n]
        label = f"{LABELS.get(name, name)} (n={len(runs)})"
        for ax, key in zip(axes, ["eval_success", "eval_return", "eval_collision", "eval_success_stoch"]):
            if not all(key in r for r in runs):
                continue
            y = np.stack([r[key][:n] for r in runs])
            mu, sd = y.mean(0), y.std(0)
            ax.plot(x, mu, label=label)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        succ = np.stack([r["eval_success"][:n] for r in runs])
        if all("eval_success_stoch" in r for r in runs):
            ss = np.stack([r["eval_success_stoch"][:n] for r in runs])[:, -1]
            stoch = f"{ss.mean():.3f} ± {ss.std():.3f}"
        else:
            stoch = "n/a"
        ret = np.stack([r["eval_return"][:n] for r in runs])
        def at(step):
            i = int(np.argmin(np.abs(x - step)))
            return f"{fmt(succ[:, i])} (at {int(x[i])})" if abs(x[i] - step) <= 0.1 * step else "n/a"
        wt = np.array([r["wall_time"][n - 1] for r in runs]) if all("wall_time" in r for r in runs) else None
        wts = f"{wt.mean():.0f} ± {wt.std():.0f} s" if wt is not None else "n/a"
        fmt = lambda v: f"{v.mean():.3f} ± {v.std():.3f}"
        col = np.stack([r["eval_collision"][:n] for r in runs])[:, -1]
        lines.append(f"| {LABELS.get(name, name)} | {len(runs)} | {int(x[-1])} | {fmt(succ[:, -1])} | "
                     f"{ret[:, -1].mean():.2f} ± {ret[:, -1].std():.2f} | " + " | ".join(at(a) for a in args.at) + " | "
                     f"{fmt(succ.max(1))} | {stoch} | {fmt(col)} | {wts} |")
    for ax, t in zip(axes, ["eval success rate (deterministic policy, 512 episodes)",
                            args.return_label, "eval collision rate",
                            "eval success rate, stochastic policy"]):
        ax.set_title(t, fontsize=10); ax.set_xlabel("environment steps"); ax.grid(alpha=0.3)
        if args.xmax:
            ax.set_xlim(0, args.xmax)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    if args.logx:  # optional log-x copy (off by default since the 2026-10-02 cleanup)
        for ax in axes:
            ax.set_xscale("symlog", linthresh=1e4)
        fig.savefig(args.out.replace(".png", "_logx.png"), dpi=130)
    with open(args.table, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
