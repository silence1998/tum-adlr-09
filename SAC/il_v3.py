"""
Imitation learning on env v3 with the A* planner as expert (same 65-dim observation as the RL policies).

  python il_v3.py demos  --mode static --episodes 50000            -> results/v3/il/demos_static.pt
  python il_v3.py bc     --mode static --n-episodes 10000          -> results/v3/il/bc_static_10k.pt (+ eval json)
  python il_v3.py dagger --mode static --init-episodes 1000 --iters 10 --eps-per-iter 1000

Demonstrations: A* (static: plan once; moving: re-plan every step + one-step safety filter) on training layouts
with env seeds 100000 + batch index (eval layouts use env seed 2026 -> disjoint streams); only successful
episodes are kept. Policy = the SAC actor MLP (MultiHeadActor, 1 head, 2x256), loss = MSE between tanh(mu) and the
expert action; deterministic evaluation tanh(mu) on the 512 eval layouts (same as SAC).
DAgger (Ross et al. 2011): start from BC on `init-episodes` demos; each iteration rolls out the CURRENT learner
(beta = 0, pure learner actions) on fresh training layouts (env seeds 200000 + ...), the expert labels every
visited state (A* re-planned from that exact state, safety filter in moving modes), the labelled states are
added to the dataset, and the policy is trained further on the aggregate.
"""
import argparse, dataclasses, importlib.util, json, os, time
import numpy as np
import torch
from env_v3 import EnvV3, EnvV3Config, evaluate_v3
from astar_v3 import AStarPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../results/v3"))
IL = os.path.join(R, "il")
_spec = importlib.util.spec_from_file_location("sacx_mod", os.path.join(HERE, "training_SAC-X.py"))
sx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sx)


def cfg_of(mode):
    return EnvV3Config.from_yaml(os.path.join(HERE, f"../configs/v3_{mode}.yaml"))


@torch.no_grad()
def rollout_label(cfg, n, seed, learner=None, device="cpu", margin=0.15, act_noise=0.0):
    """Roll out n episodes on layouts of env seed `seed`. learner=None -> execute the expert (demos);
    otherwise execute the learner and label every visited state with the expert (DAgger).
    Returns obs (M,65), expert act (M,2), episode id (M,), per-episode outcome list."""
    env = EnvV3(n, cfg, device="cpu", seed=seed)
    obs = env.reset()
    moving = cfg.motion != "static"
    expert = AStarPolicy(cfg, margin=margin, replan_always=learner is not None or act_noise > 0)
    alive = torch.ones(n, dtype=torch.bool)
    O, A, E = [], [], []
    succ = torch.zeros(n, dtype=torch.bool); coll = torch.zeros(n, dtype=torch.bool)
    for t in range(cfg.max_episode_steps):
        env._alive_np = alive.numpy()
        a_exp = expert(obs, env).cpu()
        idx = torch.nonzero(alive).squeeze(1)
        O.append(obs[idx].clone()); A.append(a_exp[idx].clone()); E.append(idx.clone())
        if learner is None:
            a = a_exp if act_noise <= 0 else (a_exp + act_noise * torch.randn_like(a_exp)).clamp(-1, 1)   # DART-style noisy execution, clean labels
        else:
            mu, _ = learner(obs.to(device)); a = torch.tanh(mu[:, 0]).cpu()
        obs, r, term, trunc, info = env.step(a, auto_reset=False)
        succ |= info["reached"] & alive; coll |= info["collided"] & alive
        alive &= ~(term | trunc)
        if not alive.any():
            break
    oc = ["goal" if s else "collision" if c else "timeout" for s, c in zip(succ.tolist(), coll.tolist())]
    return torch.cat(O), torch.cat(A), torch.cat(E), oc, expert.stats


def cmd_demos(a):
    cfg = cfg_of(a.mode); os.makedirs(IL, exist_ok=True)
    t0 = time.time(); Os, As, Es, stats = [], [], [], {"attempted": 0, "goal": 0, "collision": 0, "timeout": 0,
                                                         "initial_plan_fail": 0}
    nb = (a.episodes + a.batch - 1) // a.batch
    off = 0
    for b in range(nb):
        o, act, e, oc, st = rollout_label(cfg, a.batch, 100000 + b, margin=a.margin, act_noise=a.act_noise)
        oc = np.array(oc); keep = torch.as_tensor(oc == "goal")
        m = keep[e]
        new_id = torch.cumsum(keep.long(), 0) - 1 + off
        Os.append(o[m]); As.append(act[m]); Es.append(new_id[e[m]])
        off += int(keep.sum())
        stats["attempted"] += a.batch
        for k in ("goal", "collision", "timeout"): stats[k] += int((oc == k).sum())
        stats["initial_plan_fail"] += st.get("initial_plan_fail", 0)
        print(f"[demos {a.mode}] batch {b + 1}/{nb}: kept {off} episodes, {sum(len(x) for x in Os)} samples, "
              f"{time.time() - t0:.0f}s", flush=True)
    stats["kept_episodes"] = off; stats["samples"] = int(sum(len(x) for x in Os)); stats["time_s"] = time.time() - t0
    stats["kept_fraction"] = off / stats["attempted"]; stats["margin"] = a.margin; stats["act_noise"] = a.act_noise
    torch.save({"obs": torch.cat(Os), "act": torch.cat(As), "ep": torch.cat(Es), "stats": stats,
                "cfg": dataclasses.asdict(cfg)}, os.path.join(IL, f"demos_{a.mode}{a.suffix}.pt"))
    json.dump(stats, open(os.path.join(IL, f"demos_{a.mode}{a.suffix}_stats.json"), "w"), indent=1)
    print(json.dumps(stats))


def new_actor(dev, hidden=256):
    return sx.MultiHeadActor(65, 2, 1, hidden=hidden).to(dev)


def train(actor, obs, act, steps, dev, lr=3e-4, batch=1024, val=None, log_every=1000):
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    obs, act = obs.to(dev), act.to(dev)
    best, best_sd = float("inf"), None
    for it in range(1, steps + 1):
        i = torch.randint(0, len(obs), (min(batch, len(obs)),), device=dev)
        mu, _ = actor(obs[i])
        loss = ((torch.tanh(mu[:, 0]) - act[i]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if val is not None and (it % log_every == 0 or it == steps):
            with torch.no_grad():
                vm, _ = actor(val[0].to(dev)); vl = float(((torch.tanh(vm[:, 0]) - val[1].to(dev)) ** 2).sum(-1).mean())
            if vl < best:
                best, best_sd = vl, {k: v.clone() for k, v in actor.state_dict().items()}
    if best_sd is not None:
        actor.load_state_dict(best_sd)
    return best


def evaluate(actor, cfg, dev, n=512, seed=2026, per_episode=False):
    actor.eval()
    with torch.no_grad():
        ev = evaluate_v3(lambda o, e: torch.tanh(actor(o)[0][:, 0]), cfg, n, dev, seed, per_episode=per_episode)
    actor.train()
    return ev


def load_demos(mode, n_eps):
    d = torch.load(os.path.join(IL, f"demos_{mode}.pt"), weights_only=False)
    n_eps = min(n_eps, int(d["ep"].max()) + 1)   # moving modes keep ~98-99% of 50176 attempted -> use all kept
    m = d["ep"] < n_eps
    return d["obs"][m], d["act"][m], d["ep"][m]


def split(obs, act, ep, frac=0.1):
    n = int(ep.max()) + 1; v = ep >= int(n * (1 - frac))
    return (obs[~v], act[~v]), (obs[v], act[v])


def save(actor, path, meta, hidden=256):
    torch.save({"actor": actor.state_dict(), "args": {"tasks": "0", "features": "v3", "hidden": hidden}, "il": meta}, path)


def cmd_bc(a):
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    cfg = cfg_of(a.mode)
    obs, act, ep = load_demos(a.mode + a.demos, a.n_episodes)
    (to, ta), val = split(obs, act, ep)
    actor = new_actor(dev, a.hidden); t0 = time.time()
    vl = train(actor, to, ta, a.steps, dev, val=val)
    tt = time.time() - t0
    ev = evaluate(actor, cfg, dev)
    tag = (f"bc_{a.mode}_{a.n_episodes // 1000}k" if a.n_episodes >= 1000 else f"bc_{a.mode}_{a.n_episodes}") + a.suffix
    meta = {"method": "BC", "mode": a.mode, "demo_episodes": int(ep.max()) + 1, "samples": len(obs), "steps": a.steps,
            "val_mse": vl, "train_time_s": tt, "hidden": a.hidden, "demos": "demos_" + a.mode + a.demos, "eval": ev}
    save(actor, os.path.join(IL, tag + ".pt"), meta, a.hidden)
    json.dump(meta, open(os.path.join(IL, tag + ".json"), "w"), indent=1)
    print(tag, json.dumps({k: v for k, v in meta.items() if k != "eval"}), json.dumps(ev))


def cmd_dagger(a):
    dev = torch.device(a.device); torch.manual_seed(a.seed)
    cfg = cfg_of(a.mode)
    obs, act, ep = load_demos(a.mode + a.demos, a.init_episodes)
    (to, ta), val = split(obs, act, ep)
    actor = new_actor(dev, a.hidden); t0 = time.time()
    train(actor, to, ta, a.init_steps, dev, val=val)
    D_o, D_a = [to], [ta]
    hist = []
    ev = evaluate(actor, cfg, dev)
    hist.append({"iter": 0, "expert_labelled_episodes": a.init_episodes, "samples": len(to), "eval": ev,
                 "time_s": time.time() - t0})
    print(f"[dagger {a.mode}] iter 0 succ {ev['eval_success']:.3f} coll {ev['eval_collision']:.3f}", flush=True)
    labelled_eps = a.init_episodes
    for it in range(1, a.iters + 1):
        o, act_e, e, oc, st = rollout_label(cfg, a.eps_per_iter, 200000 + it, learner=actor, device=dev, margin=a.margin)
        D_o.append(o); D_a.append(act_e); labelled_eps += a.eps_per_iter
        Do, Da = torch.cat(D_o), torch.cat(D_a)
        train(actor, Do, Da, a.iter_steps, dev, val=None)   # last weights (no selection on expert-only val data)
        ev = evaluate(actor, cfg, dev)
        roll = {k: oc.count(k) / len(oc) for k in ("goal", "collision", "timeout")}
        hist.append({"iter": it, "expert_labelled_episodes": labelled_eps, "samples": len(Do), "eval": ev,
                     "learner_rollout_outcomes": roll, "time_s": time.time() - t0})
        print(f"[dagger {a.mode}] iter {it} samples {len(Do)} | learner rollouts {roll} | eval succ "
              f"{ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} to {ev['eval_timeout']:.3f} | {time.time() - t0:.0f}s", flush=True)
    tag = f"dagger_{a.mode}{a.suffix}"
    meta = {"method": "DAgger", "mode": a.mode, "init_demo_episodes": a.init_episodes, "iters": a.iters,
            "eps_per_iter": a.eps_per_iter, "expert_labelled_episodes": labelled_eps, "samples": int(len(torch.cat(D_o))),
            "time_s": time.time() - t0, "hidden": a.hidden, "iter_steps": a.iter_steps, "margin": a.margin,
            "history": hist, "eval": hist[-1]["eval"]}
    save(actor, os.path.join(IL, tag + ".pt"), meta, a.hidden)
    json.dump(meta, open(os.path.join(IL, tag + ".json"), "w"), indent=1)


@torch.no_grad()
def cmd_diag(a):
    """Where does the learner deviate from the expert? Roll out the learner on the 512 eval layouts, relabel every visited
    state with A* (replanning from the current state), compare actions overall vs at the last state before a collision."""
    cfg = cfg_of(a.mode)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    actor = new_actor("cpu", ck["args"].get("hidden", 256)); actor.load_state_dict(ck["actor"]); actor.eval()
    o, ae, e, oc, _ = rollout_label(cfg, 512, 2026, learner=actor, device="cpu", margin=a.margin)
    al = torch.tanh(actor(o)[0][:, 0])
    err = (al - ae).norm(dim=-1)
    cos = (al * ae).sum(-1) / (al.norm(dim=-1) * ae.norm(dim=-1)).clamp_min(1e-6)
    ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    last = torch.zeros(len(e), dtype=torch.bool)
    seen = set()
    for j in range(len(e) - 1, -1, -1):
        k = int(e[j])
        if k not in seen: seen.add(k); last[j] = True
    occ = np.array(oc)[e.numpy()]
    res = {"ckpt": os.path.basename(a.ckpt), "outcomes": {k: oc.count(k) for k in ("goal", "collision", "timeout")},
           "all_steps": {"n": len(e), "l2": float(err.mean()), "angle_deg_median": float(ang.median()), "frac_angle_gt_45": float((ang > 45).float().mean())}}
    for name, m in (("last_step_collision_eps", last & torch.as_tensor(occ == "collision")), ("last_step_goal_eps", last & torch.as_tensor(occ == "goal")),
                    ("nonlast_steps", ~last)):
        if m.any():
            res[name] = {"n": int(m.sum()), "l2": float(err[m].mean()), "angle_deg_median": float(ang[m].median()),
                         "frac_angle_gt_45": float((ang[m] > 45).float().mean()), "expert_speed": float(ae[m].norm(dim=-1).mean()),
                         "learner_speed": float(al[m].norm(dim=-1).mean())}
    print(json.dumps(res, indent=1))
    out = os.path.join(IL, "diag_" + os.path.basename(a.ckpt)[:-3] + ".json"); json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demos"); d.add_argument("--mode", required=True); d.add_argument("--episodes", type=int, default=50000)
    d.add_argument("--batch", type=int, default=1024); d.add_argument("--margin", type=float, default=0.15)
    d.add_argument("--act-noise", type=float, default=0.0); d.add_argument("--suffix", default="")
    b = sp.add_parser("bc"); b.add_argument("--mode", required=True); b.add_argument("--n-episodes", type=int, default=10000)
    b.add_argument("--steps", type=int, default=20000)
    g = sp.add_parser("dagger"); g.add_argument("--mode", required=True); g.add_argument("--init-episodes", type=int, default=1000)
    g.add_argument("--init-steps", type=int, default=10000); g.add_argument("--iters", type=int, default=10)
    g.add_argument("--eps-per-iter", type=int, default=1000); g.add_argument("--iter-steps", type=int, default=4000)
    g.add_argument("--margin", type=float, default=0.15)
    for p in (b, g):
        p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--hidden", type=int, default=256); p.add_argument("--demos", default="", help="demo file suffix")
        p.add_argument("--suffix", default="", help="output tag suffix")
    q = sp.add_parser("diag"); q.add_argument("--mode", required=True); q.add_argument("--ckpt", required=True)
    q.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    {"demos": cmd_demos, "bc": cmd_bc, "dagger": cmd_dagger, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
