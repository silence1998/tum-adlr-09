"""Aggregate results/sacx_aux/eval/*.json (+ training CSVs) into a markdown table and plots.
    python sacx_aux_report.py   (writes ../results/sacx_aux/eval_table.md, sacx_aux_comparison.png)"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results/sacx_aux"
LAB = {"sac": "SAC (plain, reference)", "sacx_full": "SAC-X old (main+approach+4 axis, SAC-Q, xi=20)",
       "aux_main": "SAC-X code, main only", "aux_approach": "main+approach", "aux_axis": "main+4 axis-move",
       "aux_obsaware": "main+safe_progress+clearance", "aux_appr_obs": "main+approach+safe_progress+clearance",
       "aux_full_U": "full 6 tasks, SAC-U", "aux_full_xi100": "full, xi=100 (1 intention/episode)",
       "aux_full_scale1": "full, aux reward scale 1.0", "aux_full_seq1": "full, no Retrace (seq-len 1)"}
ORDER = list(LAB)


def load_csv(p):
    with open(p) as f:
        h = f.readline().strip().split(",")
        a = np.array([[float(x) for x in l.strip().split(",")] for l in f if l.strip()])
    return {k: a[:, i] for i, k in enumerate(h)}


def main():
    rows = {}
    for p in glob.glob(os.path.join(R, "eval", "*.json")):
        nm = os.path.basename(p)[:-5]
        m = re.match(r"^(.*)_s(\d)(?:_1M)?(?:_(sample|qmax|sched))?$", nm)
        if not m: continue
        d = json.load(open(p)); v, s, act = m.group(1), int(m.group(2)), m.group(3) or "mean"
        rows.setdefault((v, act), {})[s] = d
    keys = ["goal", "collision", "timeout", "reversals_per_ep", "near_obstacle_reversals_per_ep", "path_ratio_goal", "len_goal"]
    L = ["| 变体 | 动作 | seeds | goal | collision | timeout | 反向次数/回合 | 障碍旁反向/回合 | 路径比(成功) | 成功步数 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(vals, k):
        a = np.array(vals, float)
        if k in ("goal", "collision", "timeout"):
            return " / ".join(str(int(x)) for x in a) + (f" (均值 {a.mean():.1f})" if len(a) > 1 else "")
        return f"{a.mean():.2f}" + (f" ± {a.std():.2f}" if len(a) > 1 else "")
    for v in ORDER + sorted({k[0] for k in rows} - set(ORDER)):
        for act in ("mean", "sample", "qmax", "sched"):
            if (v, act) not in rows: continue
            ss = sorted(rows[(v, act)]); d = [rows[(v, act)][s] for s in ss]
            L.append(f"| {LAB.get(v, v)} | {act} | {','.join(map(str, ss))} | " +
                     " | ".join(fmt([x[k] for x in d], k) for k in keys) + " |")
    open(os.path.join(R, "eval_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

    # plots: learning curves (det. success), timeouts and reversals per variant (deterministic main head)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2))
    curves = {}
    for p in glob.glob(os.path.join(R, "runs", "*_s*.csv")):
        m = re.match(r"(.+)_s(\d)\.csv$", os.path.basename(p))
        if m.group(2) != "1": continue  # seeds 0/2 were stopped by the user (incomplete logs)
        curves.setdefault(m.group(1), []).append(load_csv(p))
    for nm, base in (("sacx_full", "sacx_q"), ("sac", "sac")):
        fs = sorted(glob.glob(f"../results/final/runs_1M/{base}_s[0-9].csv"))
        if fs: curves[nm] = [load_csv(p) for p in fs]
    for v in ORDER:
        if v not in curves: continue
        rs = curves[v]; n = min(len(r["env_steps"]) for r in rs)
        y = np.stack([r["eval_success"][:n] for r in rs]); x = rs[0]["env_steps"][:n]
        ax[0].plot(x, y.mean(0), label=f"{LAB[v]} (n={len(rs)} seed{'s' if len(rs)>1 else ' 1'})", lw=2.2 if v in ("sac", "sacx_full") else 1.3,
                   ls="--" if v in ("sac", "sacx_full") else "-")
    ax[0].set_title("eval success (deterministic, 512 eps, training-time eval)"); ax[0].set_xlabel("env steps")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_ylim(0.6, 1.0)
    vs = [v for v in ORDER if (v, "mean") in rows]
    for a, k, t in ((ax[1], "timeout", "timeouts on the 512 GIF layouts (seed 2026), det. main head"),
                    (ax[2], "reversals_per_ep", "direction reversals per episode (det.)")):
        mu = [np.mean([d[k] for d in rows[(v, "mean")].values()]) for v in vs]
        pts = [[d[k] for d in rows[(v, "mean")].values()] for v in vs]
        a.barh(range(len(vs)), mu, color=["#9467bd" if v == "sac" else "#e377c2" if v == "sacx_full" else "#17becf" for v in vs])
        for i, p in enumerate(pts): a.plot(p, [i] * len(p), "k.", ms=5)
        a.set_yticks(range(len(vs))); a.set_yticklabels([LAB[v] for v in vs], fontsize=8); a.invert_yaxis()
        a.set_title(t, fontsize=10); a.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(R, "sacx_aux_comparison.png"), dpi=120)


if __name__ == "__main__":
    main()
