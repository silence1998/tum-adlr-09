"""Plot PPO learning curves on env v3 (results/v3/runs/*.csv) with the A* level from results/v3/eval/<mode>.json
and write a markdown table. python report_v3.py"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results/v3"


def load(p):
    with open(p) as f:
        h = f.readline().strip().split(","); a = np.array([[float(x) for x in l.split(",")] for l in f if l.strip()])
    return {k: a[:, i] for i, k in enumerate(h)}


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(R, "runs", "ppo_*_s*.csv"))):
        m = re.match(r"ppo_(.+)_s(\d+)\.csv", os.path.basename(p)); runs.setdefault(m.group(1), []).append(load(p))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    rows = ["| env config | seeds | det success | det collision | det timeout | stoch success | ep. length (det) | A* success / collision / timeout | A* plan failures |",
            "|---|---|---|---|---|---|---|---|---|"]
    for i, (cfg, rs) in enumerate(sorted(runs.items())):
        n = min(len(r["env_steps"]) for r in rs); x = rs[0]["env_steps"][:n]
        col = f"C{i}"
        for a, k in zip(ax, ("eval_success", "eval_collision", "eval_timeout")):
            y = np.stack([r[k][:n] for r in rs]); a.plot(x, y.mean(0), color=col, label=f"PPO {cfg} (n={len(rs)})")
            if len(rs) > 1: a.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.15)
        mode = cfg.split("_")[0]
        ev = os.path.join(R, "eval", f"{cfg}.json")
        ast = json.load(open(ev)).get("astar") if os.path.exists(ev) else None
        if ast:
            ax[0].axhline(ast["eval_success"], color=col, ls=":", lw=1)
        f = lambda k: np.array([r[k][n - 1] for r in rs])
        fm = lambda v: f"{v.mean():.3f}" + (f" ± {v.std():.3f}" if len(v) > 1 else "")
        steps_done = int(x[-1])
        a_s = (f"{ast['eval_success']:.3f} / {ast['eval_collision']:.3f} / {ast['eval_timeout']:.3f}" if ast else "n/a")
        a_f = (f"{ast['initial_plan_fail']} initial, {ast['plan_fail_steps']} step-level" if ast else "n/a")
        rows.append(f"| {cfg} ({steps_done // 1000}k) | {len(rs)} | {fm(f('eval_success'))} | {fm(f('eval_collision'))} | "
                    f"{fm(f('eval_timeout'))} | {fm(f('eval_success_stoch'))} | {fm(f('eval_length'))} | {a_s} | {a_f} |")
    for a, t in zip(ax, ("success (deterministic PPO; dotted = A*)", "collision", "timeout")):
        a.set_title(t); a.set_xlabel("env steps"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("env v3: PPO on 512 fixed eval layouts (env seed 2026)")
    fig.tight_layout(); fig.savefig(os.path.join(R, "ppo_v3_curves.png"), dpi=120)
    open(os.path.join(R, "summary_v3.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
