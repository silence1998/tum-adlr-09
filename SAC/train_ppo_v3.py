"""
PPO on env v3 (same PPO algorithm/hyper-parameters as training_PPO.py; only the env differs).
    python train_ppo_v3.py --env-config ../configs/v3_static.yaml --total-steps 2000000 --seed 0 \
        --log ../results/v3/runs/ppo_static_s0.csv --ckpt ../results/v3/checkpoints/ppo_static_s0.pt
Eval: deterministic and stochastic policy on 512 fixed layouts (env seed --eval-seed, default 2026), no shaping.
"""
import argparse, csv, dataclasses, json, os, time
import numpy as np
import torch
import torch.nn as nn
from env_v3 import EnvV3, EnvV3Config, evaluate_v3
from training_PPO import PPOAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=8)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=512)
    ap.add_argument("--eval-seed", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt", required=True)
    EnvV3Config.add_cli_args(ap)
    args = ap.parse_args()
    cfg = EnvV3Config.from_cli(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    N, S = args.num_envs, args.num_steps
    env = EnvV3(N, cfg, device=dev, seed=args.seed)
    O, A = env.obs_dim, env.act_dim
    agent = PPOAgent(O, A).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    B = N * S; mb = B // args.num_minibatches; U = max(1, args.total_steps // B)
    buf = {k: torch.zeros(S, N, *sh, device=dev) for k, sh in
           dict(obs=(O,), act=(A,), logp=(), rew=(), done=(), val=()).items()}
    for p in (args.log, args.ckpt):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    f = open(args.log, "w", newline=""); w = csv.writer(f)
    w.writerow(["env_steps", "eval_return", "eval_success", "eval_collision", "eval_timeout", "eval_length",
                "eval_success_stoch", "eval_collision_stoch", "train_success", "train_collision", "entropy", "wall_time"])
    det = lambda o, e=None: agent.actor_mean(o).clamp(-1, 1)
    sto = lambda o, e=None: agent.get_action_and_value(o)[0].clamp(-1, 1)
    t0 = time.time(); obs = env.reset()
    rs, rc = [], []; steps, nxt, ent = 0, 0, float("nan")
    for u in range(1, U + 2):
        if steps >= nxt or u > U:
            with torch.no_grad():
                ev = evaluate_v3(det, cfg, args.eval_episodes, dev, args.eval_seed)
                es = evaluate_v3(sto, cfg, args.eval_episodes, dev, args.eval_seed)
            ts = float(np.mean(rs[-1000:])) if rs else float("nan"); tc = float(np.mean(rc[-1000:])) if rc else float("nan")
            w.writerow([steps, ev["eval_return"], ev["eval_success"], ev["eval_collision"], ev["eval_timeout"],
                        ev["eval_length"], es["eval_success"], es["eval_collision"], ts, tc, ent, time.time() - t0]); f.flush()
            print(f"[{os.path.basename(args.log)}] {steps:>8d} | det succ {ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} "
                  f"to {ev['eval_timeout']:.3f} len {ev['eval_length']:.1f} | stoch succ {es['eval_success']:.3f} | "
                  f"train succ {ts:.3f} | {time.time() - t0:.0f}s", flush=True)
            nxt += args.eval_every
        if u > U:
            break
        opt.param_groups[0]["lr"] = args.lr * (1.0 - (u - 1.0) / U)
        for t in range(S):
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(obs)
            nobs, r, term, trunc, info = env.step(a)
            r = r * args.reward_scale
            if trunc.any():
                with torch.no_grad():
                    r = r + args.gamma * agent.get_value(info["final_obs"]) * trunc.float()
            buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, a, lp, v
            buf["rew"][t], buf["done"][t] = r, (term | trunc).float()
            d = term | trunc
            if d.any():
                rs += info["reached"][d].float().tolist(); rc += info["collided"][d].float().tolist()
            obs = nobs
        steps += B
        with torch.no_grad():
            nv = agent.get_value(obs); adv = torch.zeros(S, N, device=dev); last = torch.zeros(N, device=dev)
            for t in reversed(range(S)):
                nvt = nv if t == S - 1 else buf["val"][t + 1]
                nt = 1.0 - buf["done"][t]
                delta = buf["rew"][t] + args.gamma * nvt * nt - buf["val"][t]
                last = delta + args.gamma * args.gae_lambda * nt * last; adv[t] = last
            ret = adv + buf["val"]
        bo, ba, bl = buf["obs"].reshape(-1, O), buf["act"].reshape(-1, A), buf["logp"].reshape(-1)
        bad, bre, bv = adv.reshape(-1), ret.reshape(-1), buf["val"].reshape(-1)
        for _ in range(args.update_epochs):
            perm = torch.randperm(B, device=dev)
            for s0 in range(0, B, mb):
                i = perm[s0:s0 + mb]
                _, nl, en, nvv = agent.get_action_and_value(bo[i], ba[i])
                lr_ = nl - bl[i]; ratio = lr_.exp()
                ma = (bad[i] - bad[i].mean()) / (bad[i].std() + 1e-8)
                pg = torch.max(-ma * ratio, -ma * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                vu = (nvv - bre[i]) ** 2
                vc = bv[i] + (nvv - bv[i]).clamp(-args.clip_coef, args.clip_coef)
                vl = 0.5 * torch.max(vu, (vc - bre[i]) ** 2).mean()
                ent = en.mean().item()
                loss = pg - args.ent_coef * en.mean() + args.vf_coef * vl
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); opt.step()
    f.close()
    torch.save({"agent": agent.state_dict(), "args": vars(args), "env_cfg": dataclasses.asdict(cfg)}, args.ckpt)
    print("done", time.time() - t0, "reset stats", json.dumps(env.stats))


if __name__ == "__main__":
    main()
