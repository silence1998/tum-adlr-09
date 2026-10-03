"""
Render GIFs of the final 1M-step policies (PPO sorted input, plain SAC, SAC-X) on the v2 env.
All three policies are rolled out deterministically (tanh(mu) / clipped mean) from the SAME start states
(identical env seed -> identical layouts), on the CPU (no GPU use).

    python render_gifs.py --sac checkpoints/sac_s1_1M.pt   (writes to ../results/final/gifs)
"""
import argparse, importlib.util, io, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from torch_env import TorchContinuousWorld, obs_features, feature_dim

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def make_policies(args):
    sx = load_module("sacx_mod", "training_SAC-X.py")
    pp = load_module("ppo_mod", "training_PPO.py")
    O, A = feature_dim(5), 2
    f = lambda o: obs_features(o, 10, sort_obstacles=True)
    pols = {}
    ck = torch.load(args.ppo, map_location="cpu", weights_only=False)
    assert ck["args"]["features"] == "relative_sorted"
    ag = pp.PPOAgent(O, A); ag.load_state_dict(ck["agent"]); ag.eval()
    pols["PPO"] = lambda o, ag=ag: ag.actor_mean(f(o)).clamp(-1, 1)
    for name, path in (("SAC", args.sac), ("SAC-X", args.sacx)):
        if not path or not os.path.exists(path):
            print(f"{name}: no checkpoint ({path!r}) -> skipped"); continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        T = len(ck["args"]["tasks"].split(","))
        assert (T == 1) == (name == "SAC"), f"{path}: tasks={ck['args']['tasks']}"
        act = sx.MultiHeadActor(O, A, T); act.load_state_dict(ck["actor"]); act.eval()
        pols[name] = lambda o, act=act: torch.tanh(act(f(o))[0][:, 0])
    return pols


def first_hit(p, q, obstacles, r_obs, target, r_goal, lo, hi):
    """Point where the drawn segment p->q should end (first obstacle / wall entry), for display only."""
    d = q - p; ts = [1.0]
    for c in obstacles:
        f = p - c; a = d @ d; b = 2 * f @ d; cc = f @ f - r_obs ** 2; disc = b * b - 4 * a * cc
        if a > 0 and disc >= 0:
            t = (-b - np.sqrt(disc)) / (2 * a)
            if 0 <= t <= 1: ts.append(t)
    for k in range(2):
        if d[k] > 0 and q[k] > hi: ts.append((hi - p[k]) / d[k])
        if d[k] < 0 and q[k] < lo: ts.append((lo - p[k]) / d[k])
    return p + min(ts) * d


@torch.no_grad()
def rollout(policy, n, seed):
    env = TorchContinuousWorld(n, 10, 5, 100, device="cpu", seed=seed, shaping_coef=0.0)
    obs = env.reset()
    lay = {"agent0": env.agent.clone().numpy(), "target": env.target.clone().numpy(),
           "obstacles": env.obstacles.clone().numpy(), "r_obs": env.obstacle_radius, "r_goal": env.goal_radius}
    traj = [[env.agent[i].numpy().copy()] for i in range(n)]
    outcome = ["timeout"] * n; alive = np.ones(n, bool)
    for t in range(100):
        obs, r, term, trunc, info = env.step(policy(obs), auto_reset=False)
        for i in np.nonzero(alive)[0]:
            p, q = traj[i][-1], info["agent"][i].numpy().copy()
            if info["collided"][i]:
                q = first_hit(p, q, lay["obstacles"][i], lay["r_obs"], lay["target"][i], lay["r_goal"], -0.5, 9.5)
                outcome[i] = "collision"
            elif info["reached"][i]:
                outcome[i] = "goal"
            traj[i].append(q)
        alive &= ~(term | trunc).numpy()
        if not alive.any(): break
    return lay, traj, outcome


COL = {"goal": "#2ca02c", "collision": "#d62728", "timeout": "#ff7f0e"}
ALGCOL = {"PPO": "#1f77b4", "SAC": "#9467bd", "SAC-X": "#e377c2"}


def draw(ax, lay, i, pts, k_sub, title, outcome=None, done=False):
    ax.clear(); ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.5); ax.set_aspect("equal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(labelsize=5, length=0)
    ax.grid(color="#dddddd", lw=0.5); ax.add_patch(Rectangle((-0.5, -0.5), 10, 10, fill=False, lw=1.5))
    for c in lay["obstacles"][i]:
        ax.add_patch(Circle(c, lay["r_obs"], color="#555555"))
    ax.add_patch(Circle(lay["target"][i], lay["r_goal"], color="#2ca02c", alpha=0.35))
    ax.plot(*lay["target"][i], marker="*", color="#2ca02c", ms=9)
    ax.plot(*pts[0], marker="s", color="black", ms=4)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=ALGCOL.get(title.split()[0], "C0"), lw=1.8)
    ax.plot(*pts[-1], marker="o", color=ALGCOL.get(title.split()[0], "C0"), ms=7, mec="black", mew=0.6)
    if done and outcome == "collision":
        ax.plot(*pts[-1], marker="x", color="#d62728", ms=12, mew=2.5)
    lab = f"{title} | step {k_sub}" + (f" | {outcome.upper()}" if done else "")
    ax.set_title(lab, fontsize=8, color=COL[outcome] if done else "black")


def partial(traj, s, sub):
    """trajectory up to sub-step s (sub sub-steps per env step, linear interpolation along the segments)."""
    k, frac = divmod(s, sub)
    pts = traj[:k + 1]
    if k + 1 < len(traj) and frac:
        pts = pts + [traj[k] + (traj[k + 1] - traj[k]) * frac / sub]
    return pts, min(k + (1 if frac else 0), len(traj) - 1)


def make_gif(path, panels, ncols, sub=4, hold=8, size=3.0, fps=10):
    """panels: list of (lay, i, traj, outcome, title). All panels animate in lockstep."""
    nrows = int(np.ceil(len(panels) / ncols))
    T = max(len(p[2]) - 1 for p in panels) * sub
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), dpi=80)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]: ax.axis("off")
    frames = []
    for s in list(range(T + 1)) + [T] * hold:
        for ax, (lay, i, traj, outcome, title) in zip(axes, panels):
            pts, k = partial(traj, min(s, (len(traj) - 1) * sub), sub)
            draw(ax, lay, i, pts, k, title, outcome, done=s >= (len(traj) - 1) * sub)
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return os.path.getsize(path), len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", default="checkpoints/ppo_sorted_s1_1M.pt")
    ap.add_argument("--sac", default="checkpoints/sac_s1_1M.pt", help="skipped if the file is missing")
    ap.add_argument("--sacx", default="checkpoints/sacx_q_s1_1M.pt")
    ap.add_argument("--n", type=int, default=512, help="candidate layouts (same seed for all algorithms)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="../results/final/gifs")
    ap.add_argument("--layouts", default="298,5,259,0",
                    help="indices of the 4 GIF layouts (default = the ones picked on 2026-10-01: PPO collides, SAC-X "
                         "reaches the goal, longest SAC-X detour, plus one easy layout); '' = pick automatically")
    args = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    pols = make_policies(args)
    res = {k: rollout(p, args.n, args.seed) for k, p in pols.items()}
    lay = res["PPO"][0]
    for k in res:  # same start states for all three
        assert np.allclose(res[k][0]["obstacles"], lay["obstacles"]) and np.allclose(res[k][0]["agent0"], lay["agent0"])
    oc = {k: np.array(res[k][2]) for k in res}
    stats = {k: {o: int((oc[k] == o).sum()) for o in COL} for k in res}
    print("outcomes over", args.n, "layouts:", stats)
    sacs = [k for k in res if k != "PPO"]
    ppo_col_sac_ok = np.nonzero((oc["PPO"] == "collision") & np.all([oc[k] == "goal" for k in sacs], 0))[0]
    all_ok = np.nonzero(np.all([oc[k] == "goal" for k in res], 0))[0]
    print("PPO collides, all SAC variants reach goal:", len(ppo_col_sac_ok), "| all reach goal:", len(all_ok))
    # prefer interesting layouts: visible detour (longer SAC path than straight line)
    def detour(i):
        return sum(len(res[k][1][i]) for k in sacs)
    pick_col = sorted(ppo_col_sac_ok, key=detour, reverse=True)[:3]
    pick_ok = list(all_ok[:1])
    four = [int(i) for i in args.layouts.split(",")] if args.layouts else [int(i) for i in (pick_col[:3] + pick_ok)][:4]
    meta = {"checkpoints": {"PPO": args.ppo, "SAC": args.sac if "SAC" in res else None, "SAC-X": args.sacx},
            "layout_seed": args.seed, "n_layouts": args.n, "outcomes": stats,
            "n_ppo_collides_sacs_goal": int(len(ppo_col_sac_ok)), "n_all_goal": int(len(all_ok)),
            "ppo_collides_and_goal_for": {k: int(((oc["PPO"] == "collision") & (oc[k] == "goal")).sum()) for k in sacs},
            "four_layouts": four, "files": {}}
    for k in res:
        panels = [(lay, i, res[k][1][i], res[k][2][i], f"{k} layout {j + 1}") for j, i in enumerate(four)]
        fn = os.path.join(args.out, f"{k.lower().replace('-', '')}_4layouts.gif")
        sz, nf = make_gif(fn, panels, ncols=2)
        meta["files"][os.path.basename(fn)] = {"bytes": sz, "frames": nf, "outcomes": [res[k][2][i] for i in four]}
        print(fn, sz, nf)
    for j, i in enumerate(four[:3]):
        panels = [(lay, i, res[k][1][i], res[k][2][i], f"{k}") for k in ("PPO", "SAC", "SAC-X") if k in res]
        fn = os.path.join(args.out, f"side_by_side_layout{j + 1}.gif")
        sz, nf = make_gif(fn, panels, ncols=len(panels))
        meta["files"][os.path.basename(fn)] = {"bytes": sz, "frames": nf, "layout_index": i,
                                               "outcomes": {k: res[k][2][i] for k in res},
                                               "steps": {k: len(res[k][1][i]) - 1 for k in res}}
        print(fn, sz, nf)
    with open(os.path.join(args.out, "gifs_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
