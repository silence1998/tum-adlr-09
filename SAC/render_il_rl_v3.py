"""
GIF per mode: A* vs BC vs DAgger vs SAC vs SAC-X vs PPO (deterministic) on the same fixed eval layouts (env seed 2026).
Rows = policies, columns = layouts. Layout pick: (1) a layout with a large A* detour where BC fails but DAgger succeeds
(covariate-shift example) if any, (2) the largest-detour layout where all policies are evaluated.
    python render_il_rl_v3.py --mode bounce [--layouts 3,17]
-> results/v3/il_rl/gifs/il_rl_<mode>.gif (+ gifs_meta.json)
"""
import argparse, io, json, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from env_v3 import EnvV3Config, evaluate_v3
from astar_v3 import AStarPolicy
import render_v3
from eval_il_rl_v3 import load_policy, R

render_v3.ALG.update({"BC": "#8c564b", "DAgger": "#e377c2", "SAC": "#17becf", "SAC-X": "#bcbd22", "PPO": "#1f77b4", "A*": "#9467bd"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--layouts", default="")
    ap.add_argument("--seed", type=int, default=0, help="RL seed to show")
    ap.add_argument("--bc", default="50k_s300k", help="BC tag: bc_<mode>_<tag>.pt")
    ap.add_argument("--dagger", default="_big", help="DAgger suffix: dagger_<mode><suffix>.pt")
    ap.add_argument("--max-frames", type=int, default=45)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = EnvV3Config.from_yaml(os.path.join(os.path.dirname(__file__), f"../configs/v3_{a.mode}.yaml"))
    C, I = os.path.join(R, "checkpoints"), os.path.join(R, "il")
    cand = [("BC", os.path.join(I, f"bc_{a.mode}_{a.bc}.pt")), ("DAgger", os.path.join(I, f"dagger_{a.mode}{a.dagger}.pt")),
            ("SAC", os.path.join(C, f"sac_{a.mode}_s{a.seed}.pt")), ("SAC-X", os.path.join(C, f"sacx_{a.mode}_s{a.seed}.pt")),
            ("PPO", os.path.join(C, f"ppo_{a.mode}_s{a.seed}.pt"))]
    runs = {"A*": evaluate_v3(AStarPolicy(cfg), cfg, 512, "cpu", 2026, record=True)}
    for name, p in cand:
        if os.path.exists(p):
            with torch.no_grad():
                runs[name] = evaluate_v3(load_policy(p), cfg, 512, "cpu", 2026, record=True)
    for r in runs.values(): r["goal_r"] = cfg.goal_radius
    oc = {k: np.array(v["outcome"]) for k, v in runs.items()}
    Ra = runs["A*"]; fr0 = Ra["frames"][0]
    straight = lambda i: float((fr0["agent"][i] - fr0["goal"][i]).norm()) - cfg.goal_radius
    det = {i: (Ra["path_len"][i] / max(straight(i), 1e-6) if oc["A*"][i] == "goal" else 0) for i in range(512)}
    if a.layouts:
        pick = [int(x) for x in a.layouts.split(",")]
    else:
        order = sorted(range(512), key=lambda i: -det[i])
        pick = []
        if "BC" in oc and "DAgger" in oc:
            cs = [i for i in order if oc["BC"][i] != "goal" and oc["DAgger"][i] == "goal" and det[i] > 1.05]
            pick += cs[:1]
        pick += [i for i in order if i not in pick][:2 - len(pick)]
    names = list(runs); L = len(pick)
    fig, axes = plt.subplots(len(names), L, figsize=(2.3 * L, 2.3 * len(names)), dpi=60)
    axes = np.array(axes).reshape(len(names), L)
    Tm = min(max(runs[n]["length"][i] for n in names for i in pick), a.max_frames)
    frames = []
    for t in list(range(Tm + 1)) + [Tm] * 6:
        for r, n in enumerate(names):
            for j, i in enumerate(pick):
                render_v3.draw(axes[r, j], runs[n], i, t, f"{n} L{i}", cfg.size)
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=48))
    plt.close(fig)
    out = os.path.join(R, "il_rl", "gifs", f"il_rl_{a.mode}.gif"); os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    meta = {"mode": a.mode, "layouts": pick, "rl_seed": a.seed, "bc": a.bc, "dagger": a.dagger, "bytes": os.path.getsize(out), "frames": len(frames),
            "per_layout": {int(i): {n: [runs[n]["outcome"][i], runs[n]["length"][i]] for n in names} for i in pick}}
    mp = os.path.join(os.path.dirname(out), "gifs_meta.json")
    allm = json.load(open(mp)) if os.path.exists(mp) else {}
    allm[os.path.basename(out)] = meta; json.dump(allm, open(mp, "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
