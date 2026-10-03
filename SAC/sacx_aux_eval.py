"""
Layout evaluation + oscillation diagnostics for SAC / SAC-X checkpoints (main head, deterministic tanh(mu)),
on the same 512 layouts as render_gifs.py (TorchContinuousWorld, env seed 2026), on the CPU.

    python sacx_aux_eval.py --ckpt checkpoints/sacx_q_s1_1M.pt --name sacx_full_s1 --out ../results/sacx_aux/eval
    python sacx_aux_eval.py --ckpt ... --diag          # extra per-head / Q-landscape diagnostics on the timeout layouts

Metrics (per checkpoint):
  goal / collision / timeout counts over the 512 layouts
  reversals/ep    : steps whose displacement points backwards w.r.t. the previous step (cos < 0), mean per episode
  near-obs rev/ep : the same, counted only when the agent is within 0.6 of an obstacle surface
  path ratio      : path length / (start-goal distance - goal radius), successful episodes only (1.0 = straight line)
  len (goal)      : mean steps of successful episodes
"""
import argparse, importlib.util, json, math, os
import numpy as np
import torch
from torch_env import TorchContinuousWorld, obs_features, normalize_obs, feature_dim

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sacx_mod", os.path.join(HERE, "training_SAC-X.py"))
sx = importlib.util.module_from_spec(spec); spec.loader.exec_module(sx)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    tasks = [int(t) for t in a["tasks"].split(",")]
    feat = a.get("features", "relative")
    O = normalize_obs(torch.zeros(1, 14), 10).shape[1] if feat == "absolute" else feature_dim(5)
    actor = sx.MultiHeadActor(O, 2, len(tasks)); actor.load_state_dict(ck["actor"]); actor.eval()
    critic = sx.MultiHeadCritic(O, 2, len(tasks)); critic.load_state_dict(ck["critic"]); critic.eval()
    f = (lambda o: normalize_obs(o, 10)) if feat == "absolute" else \
        (lambda o: obs_features(o, 10, sort_obstacles=(feat == "relative_sorted")))
    return actor, critic, f, tasks, ck


def clearance(pos, obstacles, r):
    return (pos.unsqueeze(1) - obstacles).norm(dim=-1).min(dim=1).values - r


@torch.no_grad()
def rollout(actor, f, n=512, seed=2026, act="mean", critic=None, ck=None, tasks=None):
    """act: mean = tanh(mu) of the main head (default, = render_gifs); sample = stochastic main head;
    qmax = argmax of the main min-Q over 72 unit directions + the mean action; sched = execute the SAC-Q schedule
    (intention re-sampled every xi steps, like during training) instead of the main head."""
    g = torch.Generator().manual_seed(seed + 1)
    dirs = torch.stack([torch.tensor([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
    if act == "sched":
        q = np.asarray(ck["scheduler_q"]); xi = ck["args"].get("xi", 20); rng = np.random.default_rng(seed)
        cur = np.zeros(n, dtype=int)
    env = TorchContinuousWorld(n, 10, 5, 100, device="cpu", seed=seed, shaping_coef=0.0)
    obs = env.reset()
    start, target, obst, r = env.agent.clone(), env.target.clone(), env.obstacles.clone(), env.obstacle_radius
    O, P, D = [], [env.agent.clone()], []
    outcome = np.array(["timeout"] * n, dtype=object); alive = torch.ones(n, dtype=torch.bool)
    alive_hist = []
    for t in range(100):
        x = f(obs)
        mu, ls = actor(x)
        if act == "mean":
            a = torch.tanh(mu[:, 0])
        elif act == "sample":
            a = torch.tanh(mu[:, 0] + ls[:, 0].exp() * torch.randn(mu[:, 0].shape, generator=g))
        elif act == "qmax":
            cand = torch.cat([dirs.unsqueeze(0).expand(x.shape[0], -1, -1), torch.tanh(mu[:, :1])], 1)  # (n,73,2)
            K = cand.shape[1]
            qv = critic.min_q(x.repeat_interleave(K, 0), cand.reshape(-1, 2))[:, 0].reshape(-1, K)
            a = cand[torch.arange(x.shape[0]), qv.argmax(1)]
        elif act == "sched":
            if t % xi == 0:
                h = min(t // xi, q.shape[0] - 1)
                for i in range(n):
                    prev = q.shape[1] - 1 if t == 0 else cur[i]
                    lg = q[h, prev] / 0.1; p = np.exp(lg - lg.max()); p /= p.sum(); p = 0.9 * p + 0.1 / len(p)
                    cur[i] = rng.choice(len(p), p=p)
            a = torch.tanh(mu[torch.arange(n), torch.as_tensor(cur)])
        O.append(obs.clone()); alive_hist.append(alive.clone())
        obs, rew, term, trunc, info = env.step(a, auto_reset=False)
        D.append(info["agent"] - info["prev_agent"]); P.append(info["agent"].clone())
        for i in torch.nonzero(alive & info["collided"]).flatten().tolist(): outcome[i] = "collision"
        for i in torch.nonzero(alive & info["reached"]).flatten().tolist(): outcome[i] = "goal"
        alive &= ~(term | trunc)
        if not alive.any(): break
    return dict(obs=torch.stack(O), pos=torch.stack(P), disp=torch.stack(D), alive=torch.stack(alive_hist),
                outcome=outcome, start=start, target=target, obst=obst, r=r, goal_r=env.goal_radius)


def metrics(ro):
    disp, alive, pos = ro["disp"], ro["alive"], ro["pos"]          # (T,n,2), (T,n), (T+1,n,2)
    T, n = alive.shape
    cos = torch.nn.functional.cosine_similarity(disp[1:], disp[:-1], dim=-1)
    valid = alive[1:] & (disp[1:].norm(dim=-1) > 1e-3) & (disp[:-1].norm(dim=-1) > 1e-3)
    rev = (cos < 0) & valid
    clear = torch.stack([clearance(pos[t], ro["obst"], ro["r"]) for t in range(1, T)])  # clearance before step t+1
    near_rev = rev & (clear < 0.6)
    steps = alive.sum(0).float()
    plen = (disp.norm(dim=-1) * alive).sum(0)
    straight = ((ro["start"] - ro["target"]).norm(dim=-1) - ro["goal_r"]).clamp(min=0.5)
    oc = ro["outcome"]; g = torch.as_tensor(oc == "goal")
    return {"goal": int((oc == "goal").sum()), "collision": int((oc == "collision").sum()),
            "timeout": int((oc == "timeout").sum()),
            "reversals_per_ep": float(rev.sum(0).float().mean()),
            "near_obstacle_reversals_per_ep": float(near_rev.sum(0).float().mean()),
            "reversals_per_ep_timeout": float(rev.sum(0).float()[torch.as_tensor(oc == "timeout")].mean()) if (oc == "timeout").any() else None,
            "path_ratio_goal": float((plen / straight)[g].mean()),
            "len_goal": float(steps[g].mean()), "rev_per_ep_goal": float(rev.sum(0).float()[g].mean())}


@torch.no_grad()
def diag(actor, critic, f, tasks, ck, ro, sac_ro=None, max_eps=19):
    """Oscillation diagnostics on the timeout episodes."""
    out = {}
    names = [sx.TASK_NAMES[t] for t in tasks]
    oc = ro["outcome"]; idx = np.nonzero(oc == "timeout")[0][:max_eps]
    out["timeout_layouts"] = idx.tolist()
    # scheduler: SAC-Q table is state-independent (period h, previous task) -> probabilities per period at prev='none'
    if "scheduler_q" in ck and len(tasks) > 1:
        q = np.asarray(ck["scheduler_q"]); temp, eps = 0.1, 0.1
        pr = []
        for h in range(q.shape[0]):
            for prev in range(q.shape[1]):
                lg = q[h, prev] / temp; p = np.exp(lg - lg.max()); p /= p.sum(); p = (1 - eps) * p + eps / len(tasks)
                pr.append(p)
        pr = np.array(pr).reshape(q.shape[0], q.shape[1], -1)
        out["scheduler_mean_prob_per_task"] = dict(zip(names, np.round(pr.mean((0, 1)), 3).tolist()))
        out["scheduler_prob_h0_prev_none"] = dict(zip(names, np.round(pr[0, -1], 3).tolist()))
    obs, alive, pos = ro["obs"], ro["alive"], ro["pos"]
    T = alive.shape[0]
    dirs = torch.stack([torch.tensor([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
    near_steps, bimodal, gaps, cos_heads, axis_main, qh = 0, 0, [], [], [], []
    for i in idx:
        for t in range(T):
            if not alive[t, i]: break
            o = obs[t, i:i + 1]
            c = clearance(pos[t, i:i + 1], ro["obst"][i:i + 1], ro["r"])[0]
            if c > 0.6: continue
            near_steps += 1
            x = f(o)
            mu, _ = actor(x); a_heads = torch.tanh(mu[0])               # (T, 2)
            am = a_heads[0]
            axis_main.append(float(am.abs().max() / am.norm().clamp(min=1e-6)))
            if len(tasks) > 1:
                cos_heads.append(torch.nn.functional.cosine_similarity(a_heads[1:], am.unsqueeze(0), dim=-1).numpy())
            qa = critic.min_q(x.expand(72, -1), dirs)[:, 0]               # main-task Q over 72 unit directions
            qh.append(critic.min_q(x, am.unsqueeze(0))[0].numpy())
            # local maxima on the circle
            lm = [(qa[k].item(), k) for k in range(72) if qa[k] >= qa[k - 1] and qa[k] >= qa[(k + 1) % 72]]
            lm.sort(reverse=True)
            if len(lm) >= 2:
                sep = abs(lm[0][1] - lm[1][1]); sep = min(sep, 72 - sep) * 5
                if sep >= 45:
                    bimodal += 1; gaps.append(lm[0][0] - lm[1][0])
    out["near_obstacle_steps_in_timeout_eps"] = near_steps
    out["frac_bimodal_main_Q_(2 maxima >=45deg apart)"] = round(bimodal / max(near_steps, 1), 3)
    out["median_gap_between_top2_maxima"] = float(np.median(gaps)) if gaps else None
    out["main_action_axis_alignment_mean(|a|max/|a|; 1=axis, 0.707=diagonal)"] = round(float(np.mean(axis_main)), 3) if axis_main else None
    if cos_heads:
        out["cos(aux head action, main head action) near obstacles"] = dict(zip(names[1:], np.round(np.mean(cos_heads, 0), 3).tolist()))
    if qh:
        out["mean Q per head (executed main action) near obstacles"] = dict(zip(names, np.round(np.mean(qh, 0), 3).tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", default=None)
    ap.add_argument("--out", default="../results/sacx_aux/eval")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--act", default="mean", choices=["mean", "sample", "qmax", "sched"])
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)
    names = args.name or [os.path.splitext(os.path.basename(c))[0] for c in args.ckpt]
    for c, nm in zip(args.ckpt, names):
        actor, critic, f, tasks, ck = load(c)
        if args.act == "sched" and len(tasks) < 2:
            continue
        ro = rollout(actor, f, act=args.act, critic=critic, ck=ck, tasks=tasks)
        if args.act != "mean":
            nm = nm + "_" + args.act
        res = {"ckpt": c, "act": args.act, "tasks": [sx.TASK_NAMES[t] for t in tasks], "scheduler": ck["args"].get("scheduler"),
               "xi": ck["args"].get("xi"), "total_steps": ck["args"].get("total_steps"), **metrics(ro)}
        if args.diag:
            res["diag"] = diag(actor, critic, f, tasks, ck, ro)
        with open(os.path.join(args.out, nm + ".json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(nm, json.dumps({k: v for k, v in res.items() if k != "diag"}))
        if args.diag:
            print("  diag:", json.dumps(res["diag"], indent=1))


if __name__ == "__main__":
    main()
