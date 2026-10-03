"""
PPO (Schulman et al. 2017) for the GridWorld motion-planning task, with vectorised GPU environments
(torch_env.TorchGridWorld). Same task / observation / action space as SAC and SAC-X in this folder.

  * N parallel envs stepped on the GPU, rollouts of `num_steps` per env
  * GAE(lambda) advantages, bootstrapping from V(final_obs) on time-limit truncation (not treated as terminal)
  * clipped surrogate objective, (optionally clipped) value loss, entropy bonus
  * per-minibatch advantage normalisation, global grad-norm clipping, linear LR annealing
  * Gaussian policy with state-independent log-std; actions are clipped to [-1, 1] by the env
  * observations normalised to [-1, 1]; rewards = original env rewards * reward_scale (default 0.1 -> +1 / -5)

Run (from the SAC folder):
    python training_PPO.py --total-steps 1000000 --seed 0 --log ../results/runs/ppo_s0.csv
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import parameters
from torch_env import make_env, normalize_obs, obs_features, feature_dim, evaluate_policy_v


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=64, help="rollout length per env")
    ap.add_argument("--size", type=int, default=parameters.env_parameters["env_size"])
    ap.add_argument("--num-obstacles", type=int, default=parameters.env_parameters["num_obstacles"])
    ap.add_argument("--max-episode-steps", type=int, default=parameters.env_parameters.get("max_episode_steps", 100))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-lr", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=8)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--clip-vloss", type=int, default=1)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=512)
    ap.add_argument("--features", choices=["relative", "relative_sorted", "absolute"], default="relative",
                    help="network input: normalised absolute coords (+ egocentric relative vectors if 'relative')")
    ap.add_argument("--env-version", type=int, default=2, choices=[1, 2],
                    help="1 = original rounded grid env (earlier results), 2 = continuous env (default)")
    ap.add_argument("--collision-value", type=float, default=None, help="env: collision reward (v1 -50, v2 -10)")
    ap.add_argument("--time-penalty", type=float, default=None, help="env: per-step reward (v1 0, v2 -0.05)")
    ap.add_argument("--shaping", type=float, default=None, help="env: potential-based shaping coef (v1 0, v2 1)")
    ap.add_argument("--mem-frac", type=float, default=None,
                    help="cap this process' share of GPU memory (torch.cuda.set_per_process_memory_fraction)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", default="../results/runs/ppo.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    if args.mem_frac is not None and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    N, S = args.num_envs, args.num_steps
    env = make_env(args.env_version, N, args.size, args.num_obstacles, args.max_episode_steps, device=dev,
                   seed=args.seed, collision_value=args.collision_value, time_penalty=args.time_penalty,
                   shaping_coef=args.shaping, shaping_gamma=args.gamma)
    O, A = env.obs_dim, env.act_dim
    ap_feat = args.features
    if ap_feat == "absolute":
        norm = lambda o: normalize_obs(o, args.size)
    else:  # 'relative_sorted': obstacle slots ordered by distance to the agent (permutation-invariant input)
        norm = lambda o, _s=(ap_feat == "relative_sorted"): obs_features(o, args.size, sort_obstacles=_s)
    O = env.obs_dim if ap_feat == "absolute" else feature_dim(args.num_obstacles)
    agent = PPOAgent(O, A).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    batch_size = N * S
    mb_size = batch_size // args.num_minibatches
    num_updates = max(1, args.total_steps // batch_size)

    obs_b = torch.zeros(S, N, O, device=dev)
    act_b = torch.zeros(S, N, A, device=dev)
    logp_b = torch.zeros(S, N, device=dev)
    rew_b = torch.zeros(S, N, device=dev)
    term_b = torch.zeros(S, N, device=dev)   # true terminal of transition t
    done_b = torch.zeros(S, N, device=dev)   # episode ended (terminal or truncated) after transition t
    val_b = torch.zeros(S, N, device=dev)

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    f = open(args.log, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["env_steps", "eval_return", "eval_success", "eval_collision", "eval_length",
                     "train_success", "train_return", "train_episodes", "approx_kl", "entropy", "wall_time",
                     "eval_return_stoch", "eval_success_stoch"])

    def policy_det(o):
        return agent.actor_mean(norm(o)).clamp(-1, 1)

    def policy_stoch(o):
        return agent.get_action_and_value(norm(o))[0].clamp(-1, 1)

    t0 = time.time()
    obs = norm(env.reset())
    ep_ret = torch.zeros(N, device=dev)
    recent_succ, recent_ret, total_eps = [], [], 0
    env_steps, next_eval = 0, 0
    approx_kl, ent = float("nan"), float("nan")

    for update in range(1, num_updates + 2):
        if env_steps >= next_eval or update > num_updates:
            ev = evaluate_policy_v(policy_det, args.env_version, args.size, args.num_obstacles, args.max_episode_steps,
                                 args.eval_episodes, device=dev, seed=10_000 + env_steps // args.eval_every)
            evs = evaluate_policy_v(policy_stoch, args.env_version, args.size, args.num_obstacles, args.max_episode_steps,
                                  args.eval_episodes, device=dev, seed=10_000 + env_steps // args.eval_every)
            ts = float(np.mean(recent_succ[-500:])) if recent_succ else float("nan")
            tr = float(np.mean(recent_ret[-500:])) if recent_ret else float("nan")
            writer.writerow([env_steps, ev["eval_return"], ev["eval_success"], ev["eval_collision"],
                             ev["eval_length"], ts, tr, total_eps, approx_kl, ent, time.time() - t0,
                             evs["eval_return"], evs["eval_success"]]); f.flush()
            print(f"[PPO s{args.seed}] steps {env_steps:>8d} | eval ret {ev['eval_return']:7.2f} succ "
                  f"{ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} len {ev['eval_length']:.1f} | stoch succ "
                  f"{evs['eval_success']:.3f} | train "
                  f"succ {ts:.3f} ret {tr:.2f} | kl {approx_kl:.4f} ent {ent:.3f} | {time.time() - t0:.0f}s",
                  flush=True)
            next_eval += args.eval_every
        if update > num_updates:
            break
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = args.lr * (1.0 - (update - 1.0) / num_updates)

        # ---------------- rollout
        for t in range(S):
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs)
            next_obs_raw, r, term, trunc, info = env.step(action)
            r_s = r * args.reward_scale
            if trunc.any():  # bootstrap time-limit truncations with V(final_obs) (they are NOT terminal)
                with torch.no_grad():
                    v_final = agent.get_value(norm(info["final_obs"]))
                r_s = r_s + args.gamma * v_final * trunc.float()
            obs_b[t], act_b[t], logp_b[t], val_b[t] = obs, action, logp, value
            rew_b[t], term_b[t], done_b[t] = r_s, term.float(), (term | trunc).float()
            ep_ret += r
            d = term | trunc
            if d.any():
                idx = torch.nonzero(d).squeeze(1)
                recent_succ += info["reached"][idx].float().tolist()
                recent_ret += ep_ret[idx].tolist()
                total_eps += idx.numel()
                ep_ret[idx] = 0
            obs = norm(next_obs_raw)
        env_steps += batch_size

        # ---------------- GAE
        with torch.no_grad():
            next_value = agent.get_value(obs)
            adv = torch.zeros(S, N, device=dev)
            lastgaelam = torch.zeros(N, device=dev)
            for t in reversed(range(S)):
                nv = next_value if t == S - 1 else val_b[t + 1]
                nonterminal = 1.0 - done_b[t]  # truncation already bootstrapped inside the reward
                delta = rew_b[t] + args.gamma * nv * nonterminal - val_b[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
                adv[t] = lastgaelam
            ret = adv + val_b

        b_obs, b_act, b_logp = obs_b.reshape(-1, O), act_b.reshape(-1, A), logp_b.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), ret.reshape(-1), val_b.reshape(-1)

        # ---------------- optimise
        stop = False
        for epoch in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=dev)
            for start in range(0, batch_size, mb_size):
                mb = perm[start:start + mb_size]
                _, newlogp, entropy, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                if args.clip_vloss:
                    v_unclipped = (newv - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + (newv - b_val[mb]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newv - b_ret[mb]) ** 2).mean()
                ent = entropy.mean().item()
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop = True
                    break
            if stop:
                break

    f.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"agent": agent.state_dict(), "args": vars(args)}, os.path.join("checkpoints", os.path.splitext(os.path.basename(args.log))[0] + ".pt"))  # named after the log (ppo_s0.pt, ppo_sorted_s0.pt): runs no longer overwrite each other
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
