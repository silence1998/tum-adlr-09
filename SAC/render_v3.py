"""
GIFs for env v3: PPO (deterministic) vs A* on the fixed eval layouts (env seed 2026), circles of varying size,
moving obstacles drawn at their current position (with a faint trail). CPU only.
    python render_v3.py --config ../configs/v3_bounce.yaml --ppo ../results/v3/checkpoints/ppo_bounce_s0.pt \
        --out ../results/v3/gifs/bounce.gif [--layouts 3,17,42]
"""
import argparse, io, json, os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from env_v3 import EnvV3Config, evaluate_v3
from astar_v3 import AStarPolicy
from training_PPO import PPOAgent

COL = {"goal": "#2ca02c", "collision": "#d62728", "timeout": "#ff7f0e"}
ALG = {"PPO": "#1f77b4", "A*": "#9467bd"}


def draw(ax, R, i, t, title, size):
    T = min(t, R["length"][i]); fr = R["frames"]
    ax.clear(); ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.add_patch(Rectangle((0, 0), size, size, fill=False, lw=1.5))
    g = fr[0]["goal"][i].numpy()
    ax.add_patch(Circle(g, R["goal_r"], color=COL["goal"], alpha=0.35)); ax.plot(*g, "*", color=COL["goal"], ms=8)
    st = fr[T]
    for k in np.nonzero(st["active"][i].numpy())[0]:
        trail = np.array([fr[s]["obs_pos"][i, k].numpy() for s in range(max(0, T - 8), T + 1)])
        if len(trail) > 1 and np.abs(trail[-1] - trail[0]).sum() > 1e-6:
            ax.plot(trail[:, 0], trail[:, 1], "-", color="#999999", lw=1, alpha=0.6)
        ax.add_patch(Circle(st["obs_pos"][i, k].numpy(), float(st["obs_r"][i, k]), color="#555555"))
    path = np.array([fr[s]["agent"][i].numpy() for s in range(T + 1)])
    c = ALG[title.split()[0]]
    ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=1.8); ax.plot(*path[0], "s", color="k", ms=4)
    ax.plot(*path[-1], "o", color=c, ms=6, mec="k", mew=0.6)
    done = t >= R["length"][i]; oc = R["outcome"][i]
    if done and oc == "collision":
        ax.plot(*path[-1], "x", color=COL["collision"], ms=11, mew=2.5)
    ax.set_title(f"{title} | t={T}" + (f" | {oc.upper()}" if done else ""), fontsize=8,
                 color=COL[oc] if done else "black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layouts", default="")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--max-frames", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cfg = EnvV3Config.from_yaml(a.config)
    ck = torch.load(a.ppo, map_location="cpu", weights_only=False)
    ag = PPOAgent(9 + 7 * cfg.n_obs_max, 2); ag.load_state_dict(ck["agent"]); ag.eval()
    with torch.no_grad():
        Rp = evaluate_v3(lambda o, e: ag.actor_mean(o).clamp(-1, 1), cfg, a.n, "cpu", 2026, record=True)
    Ra = evaluate_v3(AStarPolicy(cfg), cfg, a.n, "cpu", 2026, record=True)
    for R in (Rp, Ra): R["goal_r"] = cfg.goal_radius
    op, oa = np.array(Rp["outcome"]), np.array(Ra["outcome"])
    if a.layouts:
        pick = [int(x) for x in a.layouts.split(",")]
    else:  # two where both succeed with a longer path (obstacles in the way), one where PPO fails if any
        both = [i for i in range(a.n) if op[i] == "goal" and oa[i] == "goal"]
        fr0 = Ra["frames"][0]
        straight = lambda i: float((fr0["agent"][i] - fr0["goal"][i]).norm()) - cfg.goal_radius
        plen = lambda R, i: sum(float((R["frames"][s + 1]["agent"][i] - R["frames"][s]["agent"][i]).norm()) for s in range(R["length"][i]))
        both.sort(key=lambda i: -plen(Ra, i) / max(straight(i), 1e-6))   # largest A* detour first
        fail = [i for i in range(a.n) if op[i] != "goal"]
        pick = both[:2] + (fail[:1] if fail else both[2:3])
    L = len(pick)
    fig, axes = plt.subplots(2, L, figsize=(2.6 * L, 5.4), dpi=72)
    axes = np.array(axes).reshape(2, L)
    Tm = min(max(max(Rp["length"][i], Ra["length"][i]) for i in pick), a.max_frames)
    frames = []
    for t in list(range(Tm + 1)) + [Tm] * 6:
        for j, i in enumerate(pick):
            draw(axes[0, j], Rp, i, t, f"PPO L{i}", cfg.size)
            draw(axes[1, j], Ra, i, t, f"A* L{i}", cfg.size)
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    meta = {"config": a.config, "ppo": a.ppo, "layout_seed": 2026, "layouts": pick, "bytes": os.path.getsize(a.out),
            "frames": len(frames), "max_frames_shown": a.max_frames,
            "ppo_outcomes_512": {k: int((op == k).sum()) for k in COL}, "astar_outcomes_512": {k: int((oa == k).sum()) for k in COL},
            "per_layout": {int(i): {"PPO": [Rp["outcome"][i], Rp["length"][i]], "A*": [Ra["outcome"][i], Ra["length"][i]]} for i in pick}}
    mp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "gifs_meta.json")
    allm = json.load(open(mp)) if os.path.exists(mp) else {}
    allm[os.path.basename(a.out)] = meta; json.dump(allm, open(mp, "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
