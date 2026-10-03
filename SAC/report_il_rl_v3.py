"""
Tables + plots for IL vs RL on env v3 (main level).  python report_il_rl_v3.py
reads results/v3/il_rl/eval_<mode>[_hard].json, results/v3/il/*.json, results/v3/runs/{ppo,sac,sacx}_<mode>_s*.csv
-> results/v3/il_rl/table_il_rl.md, il_rl_curves.png, il_scaling.png
"""
import csv, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/v3"))
O = os.path.join(R, "il_rl"); IL = os.path.join(R, "il")
MODES = ["static", "random", "bounce"]
GROUPS = [("A*", r"^astar$"),
          ("BC-1k (2×256, 20k it)", r"^bc_[a-z]+_1k$"), ("BC-10k (2×256, 20k it)", r"^bc_[a-z]+_10k$"), ("BC-50k (2×256, 20k it)", r"^bc_[a-z]+_50k$"),
          ("BC-1k (2×256, 100k it)", r"^bc_[a-z]+_1k_s100k$"), ("BC-10k (2×256, 100k it)", r"^bc_[a-z]+_10k_s100k$"),
          ("BC-50k (2×256, 100k it)", r"^bc_[a-z]+_50k_s100k$"), ("BC-50k (2×256, 300k it)", r"^bc_[a-z]+_50k_s300k$"),
          ("BC-1k (2×1024, 100k it)", r"^bc_[a-z]+_1k_h1024$"), ("BC-10k (2×1024, 100k it)", r"^bc_[a-z]+_10k_h1024$"),
          ("BC-50k (2×1024, 100k it)", r"^bc_[a-z]+_50k_h1024$"),
          ("DAgger (2×256, 1k+10×1k)", r"^dagger_[a-z]+$"), ("DAgger (2×256, 10k+8×5k)", r"^dagger_[a-z]+_big$"), ("DAgger (2×1024, 1k+10×1k)", r"^dagger_[a-z]+_h1024$"),
          ("DAgger (2×1024, 10k+8×5k)", r"^dagger_[a-z]+_h1024_big$"),
          ("PPO (3M)", r"^ppo_"), ("SAC (1M)", r"^sac_"), ("SAC-X (1M)", r"^sacx_")]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def ms(v):
    v = [x for x in v if x is not None]
    if not v: return "–"
    return f"{np.mean(v):.3f}" + (f"±{np.std(v):.3f}" if len(v) > 1 else "")


def cost(mode, label, keys):
    """(env steps, expert-labelled episodes, wall-time string) per model group."""
    if label == "A*":
        return "–", "–", "planner"
    if label.startswith("BC"):
        st = jl(os.path.join(IL, f"demos_{mode}_stats.json")); j = jl(os.path.join(IL, keys[0] + ".json"))
        if not (st and j): return "–", "–", "–"
        frac = j["demo_episodes"] / st["kept_episodes"]
        return (f"{j['samples']:,} (expert)", f"{j['demo_episodes']:,} kept / {round(st['attempted'] * frac):,} rolled out",
                f"demos {st['time_s'] * frac:.0f}s + train {j['train_time_s']:.0f}s")
    if label.startswith("DAgger"):
        j = jl(os.path.join(IL, keys[0] + ".json"))
        if not j: return "–", "–", "–"
        return (f"{j['samples']:,} (expert+learner)", f"{j['expert_labelled_episodes']:,} ({j['init_demo_episodes'] // 1000}k demos + {j['iters']}×{j['eps_per_iter']} relabelled)",
                f"{j['time_s']:.0f}s")
    algo = {"PPO (3M)": "ppo", "SAC (1M)": "sac", "SAC-X (1M)": "sacx"}[label]
    st, wt = [], []
    for k in keys:
        r = rows(os.path.join(R, "runs", k + ".csv"))
        if r:
            st.append(float(r[-1]["env_steps"])); wt.append(float(r[-1]["wall_time"]))
    if not st: return "–", "–", "–"
    return f"{np.mean(st) / 1e6:.2f}M (reward)", "0", f"{np.mean(wt) / 60:.0f} min/run"


def table(lvl=""):
    out = []
    hdr = "| mode | method | n | success | collision | timeout | path / A* | env steps | expert episodes | wall time |"
    out += [hdr, "|" + "---|" * 10]
    for mode in MODES:
        ev = jl(os.path.join(O, f"eval_{mode}{lvl}.json"))
        if not ev: continue
        for label, pat in GROUPS:
            ks = [k for k in ev if re.search(pat, k)]
            if not ks: continue
            g = lambda f: ms([ev[k][f] for k in ks])
            c = cost(mode, label, ks) if not lvl else ("", "", "")
            out.append(f"| {mode}{lvl} | {label} | {len(ks)} | {g('goal')} | {g('collision')} | {g('timeout')} | "
                       f"{g('path_ratio_vs_astar')} | {c[0]} | {c[1]} | {c[2]} |")
    return "\n".join(out)


def curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    col = {"ppo": "#1f77b4", "sac": "#17becf", "sacx": "#bcbd22"}
    for ax, mode in zip(axes, MODES):
        for algo in ("ppo", "sac", "sacx"):
            for f in sorted(glob.glob(os.path.join(R, "runs", f"{algo}_{mode}_s[0-9].csv"))):
                r = rows(f)
                if not r: continue
                kx = next(k for k in ("env_steps", "step", "steps") if k in r[0])
                ky = next(k for k in ("eval_success", "succ", "success") if k in r[0])
                x = [float(q[kx]) for q in r if q.get(ky) not in (None, "")]; y = [float(q[ky]) for q in r if q.get(ky) not in (None, "")]
                ax.plot(x, y, color=col[algo], alpha=0.8, label=f"{algo.upper().replace('SACX', 'SAC-X')} {os.path.basename(f)[-6:-4]}")
        ev = jl(os.path.join(O, f"eval_{mode}.json")) or {}
        for k, c, ls, lab in ((f"bc_{mode}_50k_s300k", "#d62728", "--", "BC-50k 2×256 (300k it)"), (f"bc_{mode}_50k", "#8c564b", ":", "BC-50k 2×256 (20k it)"),
                              (f"dagger_{mode}_big", "#ff7f0e", "-.", "DAgger 2×256 (10k+8×5k)"), ("astar", "#9467bd", ":", "A*")):
            if k in ev:
                ax.axhline(ev[k]["goal"], color=c, ls=ls, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("env steps (RL)"); ax.set_ylabel("eval success (512 layouts)"); ax.set_title(f"v3 {mode} (main)")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(O, "il_rl_curves.png"), dpi=110); plt.close(fig)


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cm = {"static": "#1f77b4", "random": "#2ca02c", "bounce": "#d62728"}
    for mode in MODES:
        for suf, ls, lab in (("", "o:", "2×256, 20k it"), ("_s100k", "o-", "2×256, 100k it"), ("_h1024", "s--", "2×1024, 100k it")):
            pts = []
            for t in ("1k", "10k", "50k"):
                j = jl(os.path.join(IL, f"bc_{mode}_{t}{suf}.json"))
                if j: pts.append((j["demo_episodes"], j["eval"]["eval_success"]))
            if pts: axes[0].plot(*zip(*pts), ls, color=cm[mode], label=f"BC {mode} ({lab})")
        for suf, ls, lab in (("", "o:", "2×256, 1k+10×1k"), ("_big", "o-", "2×256, 10k+8×5k"), ("_h1024_big", "s--", "2×1024, 10k+8×5k")):
            d = jl(os.path.join(IL, f"dagger_{mode}{suf}.json"))
            if d:
                h = d["history"]
                axes[1].plot([q["expert_labelled_episodes"] for q in h], [q["eval"]["eval_success"] for q in h], ls, color=cm[mode], label=f"DAgger {mode} ({lab})")
        pts = [(jl(os.path.join(IL, f"bc_{mode}_{t}_s100k.json")) or {}) for t in ("1k", "10k", "50k")]
        pts = [(j["demo_episodes"], j["eval"]["eval_success"]) for j in pts if j]
        if pts: axes[1].plot(*zip(*pts), "x-.", color=cm[mode], alpha=0.6, label=f"BC {mode} (2×256, 100k it)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("A* demo episodes (successful)"); axes[0].set_title("BC: data scaling and network size")
    axes[1].set_xscale("log"); axes[1].set_xlabel("expert-labelled episodes"); axes[1].set_title("DAgger iterations vs BC (same label budget axis)")
    for ax in axes: ax.set_ylabel("eval success (512 layouts)"); ax.grid(alpha=0.3); ax.legend(fontsize=6); ax.set_ylim(0.5, 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(O, "il_scaling.png"), dpi=110); plt.close(fig)


def main():
    os.makedirs(O, exist_ok=True)
    md = ["# IL vs RL, env v3 main level — 512 fixed eval layouts (env seed 2026), deterministic", "", table(), ""]
    h = table("_hard")
    if h.count("\n") > 1: md += ["## Hard level (zero-shot, models trained on main level)", "", h, ""]
    md += ["## RL sample efficiency (training-time eval, CUDA layout set): first env step with det. success ≥ threshold", "",
           "| run | ≥0.90 | ≥0.95 | ≥0.98 | final |", "|---|---|---|---|---|"]
    for mode in MODES:
        for algo in ("ppo", "sac", "sacx"):
            for f in sorted(glob.glob(os.path.join(R, "runs", f"{algo}_{mode}_s[0-9].csv"))):
                r = [q for q in rows(f) if q.get("eval_success") not in (None, "")]
                if not r: continue
                fst = lambda th: next((f"{float(q['env_steps']) / 1e3:.0f}k" for q in r if float(q["eval_success"]) >= th), "never")
                md.append(f"| {os.path.basename(f)[:-4]} | {fst(0.9)} | {fst(0.95)} | {fst(0.98)} | {float(r[-1]['eval_success']):.3f} |")
    md += ["", "## Demo filtering", "", "| mode | attempted | kept | kept % | collision | timeout | samples | time |", "|---|---|---|---|---|---|---|---|"]
    for mode in MODES:
        s = jl(os.path.join(IL, f"demos_{mode}_stats.json"))
        if s: md.append(f"| {mode} | {s['attempted']} | {s['kept_episodes']} | {100 * s['kept_fraction']:.2f} | {s['collision']} | {s['timeout']} | {s['samples']:,} | {s['time_s']:.0f}s |")
    open(os.path.join(O, "table_il_rl.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    curves(); scaling(); print("\n".join(md))


if __name__ == "__main__":
    main()
