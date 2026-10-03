"""
SAC-X: Scheduled Auxiliary Control (Riedmiller et al. 2018, https://arxiv.org/abs/1802.10567)
for the GridWorld motion-planning task (environment.GridWorldEnv / its GPU twin torch_env.TorchGridWorld).

NOTE: the previous version of this file was *not* SAC-X: it was an older copy of training.py (plain single-task SAC
with a V-network), without auxiliary intentions, scheduler, per-intention heads or Retrace. The original is kept in
baseline_original/training_SAC-X_original.py (and baseline_original/run_original_sacx.py runs it headless).

Components (paper section in brackets):
  * Intentions / auxiliary rewards [Sec. 4, 5]: rewards in [0, 1], computed for every transition for every task,
    so every intention learns off-policy from all data ("learning from all experience").
        0 REACH   (main task)  1 when the agent lands on the target (sparse)
        1 APPROACH             1 when the euclidean distance to the target strictly decreased (and no collision)
        2..5 MOVE_{+x,-x,+y,-y} 1 when the agent moved in that direction without colliding
        (v2 only, obstacle-aware, added 2026-10-02 for the aux-task ablation, see results/sacx_aux/)
        6 SAFE_PROGRESS        progress towards the target in [0, 1], -1 on collision
        7 CLEARANCE            distance of the new position to the nearest obstacle surface, clipped to [0, 1]; -1 on collision
    Collisions (obstacle / wall) terminate the episode for every intention (they lose all future reward).
  * Policy with a shared trunk and one tanh-Gaussian head per intention [Sec. 4.1 / Fig. 2].
  * Q network with a shared trunk and one output per intention, Q_T(s, a) [Sec. 4.2] + target network
    (twin critics / clipped double-Q added against overestimation, see MultiHeadCritic).
  * Retrace(lambda) off-policy targets on replayed trajectory snippets [Eq. 4/5, Munos et al. 2016]:
        Q_ret(s_i,a_i) = r_i + g(1-term_i) [ V(s_{i+1}) + c_{i+1} (Q_ret(s_{i+1},a_{i+1}) - Q'(s_{i+1},a_{i+1})) ]
        c_k = lambda * min(1, pi_T(a_k|s_k) / mu(a_k|s_k)),  V(s) = E_{a~pi_T}[Q'(s,a) - alpha_T log pi_T(a|s)]
    mu is the behaviour probability stored at collection time (the intention that was being executed).
    Traces are cut at episode boundaries; time-limit truncation bootstraps from V(final_obs) (not terminal).
  * Entropy-regularised policy improvement per intention [Eq. 6/7]: min E[alpha_T log pi_T(a|s) - Q_T(s,a)],
    with per-intention temperature alpha_T (auto-tuned to a target entropy; the paper uses a fixed alpha).
  * Scheduler [Sec. 4.3]: intentions switch every `xi` steps.
        SAC-U: uniform over all intentions.
        SAC-Q: Boltzmann over a table Q_S(h, T_prev, T) of the main-task return obtained after choosing T in period
               h (Monte-Carlo update, Eq. 8/9; the full intention history is approximated by (h, T_prev)).
  * Replay buffer of trajectories (per parallel env ring buffer, sampled as length-L snippets).
  * Environments are stepped on the GPU in parallel (torch_env.TorchGridWorld), observations normalised to [-1, 1].

Run (from the SAC folder):
    python training_SAC-X.py --scheduler Q --total-steps 1000000 --seed 0 --log ../results/runs/sacx_q_s0.csv
"""

import argparse
import csv
import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parameters
from torch_env import make_env, normalize_obs, obs_features, feature_dim, evaluate_policy_v
from env_v3 import EnvV3, EnvV3Config, evaluate_v3

TASK_NAMES = ["reach(main)", "approach", "move+x", "move-x", "move+y", "move-y", "safe_progress", "clearance"]
NUM_TASKS = len(TASK_NAMES)
MAIN_TASK = 0
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _obstacle_aware_rewards(info, ok, dist_prev, dist_new):
    """Tasks 6 (safe_progress) and 7 (clearance); zeros if the env does not report obstacles."""
    if "obstacles" not in info:
        z = torch.zeros_like(ok)
        return [z, z]
    col = info["collided"].float()
    new = info["agent"].float()
    clear = ((new.unsqueeze(1) - info["obstacles"].float()).norm(dim=-1).min(dim=1).values
             - float(info["obstacle_radius"]))
    return [ok * (dist_prev - dist_new).clamp(0, 1) - col,
            ok * clear.clamp(0, 1) - col]


def compute_task_rewards(info, version=1):
    """(N, NUM_TASKS) auxiliary + main rewards in [0, 1] from one env step.
    v1 (grid moves): binary indicators. v2 (continuous): progress amounts clipped to [0, 1] (max step = 1 cell),
    so tiny jitter is not rewarded like a full move."""
    prev, new, tgt = info["prev_agent"].float(), info["agent"].float(), info["target"].float()
    ok = (~info["collided"]).float()
    d = new - prev
    dist_prev = (prev - tgt).norm(dim=1)
    dist_new = (new - tgt).norm(dim=1)
    if version in (2, 3):
        cols = [
            info["reached"].float(),
            ok * (dist_prev - dist_new).clamp(0, 1),
            ok * d[:, 0].clamp(0, 1),
            ok * (-d[:, 0]).clamp(0, 1),
            ok * d[:, 1].clamp(0, 1),
            ok * (-d[:, 1]).clamp(0, 1),
        ]
        if version == 3:  # tasks 6/7 not implemented for env v3 (zeros keep the column layout)
            return torch.stack(cols + [torch.zeros_like(ok)] * 2, dim=1)
        return torch.stack(cols + _obstacle_aware_rewards(info, ok, dist_prev, dist_new), dim=1)
    r = torch.stack([
        info["reached"].float(),
        ok * (dist_new < dist_prev).float(),
        ok * (d[:, 0] > 0).float(),
        ok * (d[:, 0] < 0).float(),
        ok * (d[:, 1] > 0).float(),
        ok * (d[:, 1] < 0).float(),
    ], dim=1)
    return r


def mlp(inp, hidden, n_layers=2):
    layers, d = [], inp
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*layers)


class MultiHeadActor(nn.Module):
    """Shared trunk + one tanh-Gaussian head per intention."""

    def __init__(self, obs_dim, act_dim, n_tasks, hidden=256):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden)
        self.heads = nn.Linear(hidden, n_tasks * 2 * act_dim)
        self.n_tasks, self.act_dim = n_tasks, act_dim

    def forward(self, obs):
        out = self.heads(self.trunk(obs)).view(-1, self.n_tasks, 2, self.act_dim)
        mu, log_std = out[:, :, 0], out[:, :, 1].clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std  # (B, T, A)

    @staticmethod
    def log_prob_from_u(mu, log_std, u):
        """log pi(tanh(u)) for pre-tanh sample u, incl. tanh Jacobian. mu/log_std/u broadcast, sum over last dim."""
        lp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # numerically stable log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))
        lp = lp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return lp

    def sample(self, obs):
        """Reparameterised sample for every head: action (B,T,A), log-prob (B,T)."""
        mu, log_std = self(obs)
        u = mu + log_std.exp() * torch.randn_like(mu)
        return torch.tanh(u), self.log_prob_from_u(mu, log_std, u)


class MultiHeadCritic(nn.Module):
    """Twin Q(s, a) networks -> one value per intention each. forward() returns (2, ..., T).

    The paper uses a single Q network. With a single critic the main-task Q (true value <= 1) drifted to > 2.5 and
    kept growing in our runs: the reparameterised policy update exploits critic errors, which matters a lot here
    because the env rounds actions to grid moves (the true Q is piecewise constant in a, the learned one is not).
    Clipped double-Q (min over two critics, Fujimoto et al. 2018 / SAC v2) removes that overestimation.
    """

    def __init__(self, obs_dim, act_dim, n_tasks, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(mlp(obs_dim + act_dim, hidden), nn.Linear(hidden, n_tasks))
        self.q2 = nn.Sequential(mlp(obs_dim + act_dim, hidden), nn.Linear(hidden, n_tasks))

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=-1)
        return torch.stack((self.q1(x), self.q2(x)))

    def min_q(self, obs, act):
        return self(obs, act).min(0).values


class TrajectoryReplay:
    """Ring buffer of shape (capacity, num_envs, ...) on the device; samples length-L snippets."""

    def __init__(self, capacity, num_envs, obs_dim, act_dim, n_tasks, device):
        self.cap, self.n = capacity, num_envs
        z = lambda *s, dt=torch.float32: torch.zeros(capacity, num_envs, *s, dtype=dt, device=device)
        self.obs, self.next_obs = z(obs_dim), z(obs_dim)
        self.u = z(act_dim)                  # pre-tanh action (exact log-probs for all heads)
        self.mu_logp = z()                   # behaviour log-prob of the executed intention
        self.rew = z(n_tasks)
        self.term = z(); self.done = z()
        self.ptr, self.full, self.device = 0, False, device

    def add(self, obs, u, mu_logp, rew, term, done, next_obs):
        i = self.ptr
        self.obs[i], self.u[i], self.mu_logp[i], self.rew[i] = obs, u, mu_logp, rew
        self.term[i], self.done[i], self.next_obs[i] = term.float(), done.float(), next_obs
        self.ptr = (self.ptr + 1) % self.cap
        self.full = self.full or self.ptr == 0

    def size(self):
        return self.cap if self.full else self.ptr

    def sample(self, batch, L):
        n_start = (self.cap if self.full else self.ptr) - L + 1  # number of valid snippet starts per env
        k = torch.randint(0, n_start, (batch,), device=self.device)
        start = (k + self.ptr) % self.cap if self.full else k  # oldest data first when full -> never crosses ptr
        t = (start.unsqueeze(1) + torch.arange(L, device=self.device)) % self.cap  # (B, L)
        e = torch.randint(0, self.n, (batch,), device=self.device).unsqueeze(1).expand(-1, L)
        return (self.obs[t, e], self.u[t, e], self.mu_logp[t, e], self.rew[t, e], self.term[t, e],
                self.done[t, e], self.next_obs[t, e])


class Scheduler:
    """SAC-U (uniform) or SAC-Q (learned Boltzmann scheduler over intentions)."""

    def __init__(self, kind, num_envs, n_tasks, n_periods, device, temperature=0.1, eps=0.1, lr=0.1, seed=0):
        self.kind, self.n, self.T, self.H = kind, num_envs, n_tasks, n_periods
        self.temperature, self.eps, self.lr = temperature, eps, lr
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_periods, n_tasks + 1, n_tasks))  # (period, prev task or 'none', task)
        self.history = [[] for _ in range(num_envs)]  # per env list of (h, prev, task, main_return_so_far)
        self.main_return = np.zeros(num_envs)
        self.current = np.zeros(num_envs, dtype=np.int64)

    def probs(self, h, prev):
        if self.kind == "U":
            return np.full(self.T, 1.0 / self.T)
        logits = self.q[min(h, self.H - 1), prev] / self.temperature
        p = np.exp(logits - logits.max()); p /= p.sum()
        return (1 - self.eps) * p + self.eps / self.T

    def choose(self, env_ids, h):
        for e in env_ids:
            prev = self.history[e][-1][2] if self.history[e] else self.T
            task = self.rng.choice(self.T, p=self.probs(h, prev))
            self.history[e].append((min(h, self.H - 1), prev, task, self.main_return[e]))
            self.current[e] = task

    def end_episode(self, env_ids):
        for e in env_ids:
            total = self.main_return[e]
            if self.kind == "Q":
                for (h, prev, task, ret_before) in self.history[e]:
                    self.q[h, prev, task] += self.lr * ((total - ret_before) - self.q[h, prev, task])
            self.history[e] = []
            self.main_return[e] = 0.0


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", choices=["U", "Q"], default="Q")
    ap.add_argument("--total-steps", type=int, default=1_000_000, help="environment steps (summed over envs)")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--size", type=int, default=parameters.env_parameters["env_size"])
    ap.add_argument("--num-obstacles", type=int, default=parameters.env_parameters["num_obstacles"])
    ap.add_argument("--max-episode-steps", type=int, default=parameters.env_parameters.get("max_episode_steps", 100))
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128, help="number of trajectory snippets per update")
    ap.add_argument("--seq-len", type=int, default=8, help="Retrace snippet length")
    ap.add_argument("--retrace-lambda", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--updates-per-step", type=int, default=4, help="gradient steps per vectorised env step")
    ap.add_argument("--buffer", type=int, default=40000, help="ring buffer length per env")
    ap.add_argument("--warmup", type=int, default=10000, help="env steps before learning starts")
    ap.add_argument("--xi", type=int, default=20, help="scheduler period (steps between intention switches)")
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=512)
    ap.add_argument("--tasks", default="0,1,2,3,4,5",
                    help="comma separated intention ids to use (0 = main task must be included); '0' = SAC+Retrace")
    ap.add_argument("--features", choices=["relative", "relative_sorted", "absolute"], default="relative",
                    help="network input: normalised absolute coords (+ egocentric relative vectors if 'relative')")
    ap.add_argument("--env-version", type=int, default=2, choices=[1, 2, 3],
                    help="1 = original rounded grid env (earlier results), 2 = continuous env (default), "
                         "3 = env_v3 (needs --env-config; eval on the 512 fixed layouts, env seed 2026)")
    ap.add_argument("--env-config", default=None, help="env v3: YAML with EnvV3Config fields (configs/v3_*.yaml)")
    ap.add_argument("--collision-value", type=float, default=None, help="env: collision reward (v1 -50, v2 -10)")
    ap.add_argument("--time-penalty", type=float, default=None, help="env: per-step reward (v1 0, v2 -0.05)")
    ap.add_argument("--shaping", type=float, default=None, help="env: potential-based shaping coef (v1 0, v2 1)")
    ap.add_argument("--main-reward", choices=["sparse", "env"], default=None,
                    help="main intention reward: sparse 1 on reaching (paper style) or env reward * 0.1 "
                         "(default: sparse for env v1, env for env v2 = same task reward PPO optimises)")
    ap.add_argument("--main-reward-scale", type=float, default=0.1,
                    help="scale of the env reward for --main-reward env (0.1 = earlier runs)")
    ap.add_argument("--alpha-init", type=float, default=0.1, help="initial entropy temperature (all intentions)")
    ap.add_argument("--aux-reward-scale", type=float, default=1.0,
                    help="scale of auxiliary rewards (keeps aux Q magnitudes comparable to the main task's in the "
                         "shared trunks)")
    ap.add_argument("--mem-frac", type=float, default=None,
                    help="cap this process' share of GPU memory (torch.cuda.set_per_process_memory_fraction)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log", default="../results/runs/sacx.csv")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="where the final checkpoint <log name>.pt is written")
    args = ap.parse_args()

    if args.main_reward is None:
        args.main_reward = "sparse" if args.env_version == 1 else "env"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    if args.mem_frac is not None and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    if args.env_version == 3:  # env v3: config from YAML, observation = ready-made 65-dim feature vector
        v3cfg = EnvV3Config.from_yaml(args.env_config)
        args.max_episode_steps = v3cfg.max_episode_steps
        env = EnvV3(args.num_envs, v3cfg, device=dev, seed=args.seed)
    else:
        env = make_env(args.env_version, args.num_envs, args.size, args.num_obstacles, args.max_episode_steps, device=dev,
                       seed=args.seed, collision_value=args.collision_value, time_penalty=args.time_penalty,
                       shaping_coef=args.shaping, shaping_gamma=args.gamma)
    task_ids = [int(x) for x in args.tasks.split(",")]
    assert args.env_version == 2 or max(task_ids) <= 5, "tasks 6/7 (obstacle-aware) need env version 2"
    assert task_ids[0] == MAIN_TASK, "the main task (0) must be listed first"
    O, A, T, N = env.obs_dim, env.act_dim, len(task_ids), args.num_envs
    ap_feat = args.features
    if ap_feat == "absolute":
        norm = lambda o: normalize_obs(o, args.size)
    else:  # 'relative_sorted': obstacle slots ordered by distance to the agent (permutation-invariant input)
        norm = lambda o, _s=(ap_feat == "relative_sorted"): obs_features(o, args.size, sort_obstacles=_s)
    O = env.obs_dim if ap_feat == "absolute" else feature_dim(args.num_obstacles)
    if args.env_version == 3:
        norm = lambda o: o
        O = env.obs_dim

    actor = MultiHeadActor(O, A, T).to(dev)
    critic = MultiHeadCritic(O, A, T).to(dev)
    critic_targ = MultiHeadCritic(O, A, T).to(dev)
    critic_targ.load_state_dict(critic.state_dict())
    for p in critic_targ.parameters():
        p.requires_grad_(False)
    log_alpha = torch.full((T,), math.log(args.alpha_init), device=dev, requires_grad=True)
    target_entropy = -float(A)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)

    replay = TrajectoryReplay(args.buffer, N, env.obs_dim, A, T, dev)  # stores raw obs
    n_periods = max(1, math.ceil(args.max_episode_steps / args.xi))
    sched = Scheduler(args.scheduler, N, T, n_periods, dev, seed=args.seed)
    print(f"intentions: {[TASK_NAMES[i] for i in task_ids]}, scheduler SAC-{args.scheduler}", flush=True)
    sched.choose(range(N), 0)

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    f = open(args.log, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["env_steps", "eval_return", "eval_success", "eval_collision", "eval_length",
                     "train_main_success", "train_episodes", "critic_loss", "alpha_main", "wall_time",
                     "eval_return_stoch", "eval_success_stoch"])

    def policy_main(o):
        mu, _ = actor(norm(o))
        return torch.tanh(mu[:, MAIN_TASK])

    def policy_main_stoch(o):
        mu, log_std = actor(norm(o))
        return torch.tanh(mu[:, MAIN_TASK] + log_std[:, MAIN_TASK].exp() * torch.randn_like(mu[:, MAIN_TASK]))

    def do_eval(step, stats):
        actor.eval()
        if args.env_version == 3:  # fixed eval layouts (env seed 2026), same as PPO / A* / IL on v3
            ev = evaluate_v3(lambda o, e: policy_main(o), v3cfg, args.eval_episodes, dev, 2026)
            evs = evaluate_v3(lambda o, e: policy_main_stoch(o), v3cfg, args.eval_episodes, dev, 2026)
        else:
            ev = evaluate_policy_v(policy_main, args.env_version, args.size, args.num_obstacles, args.max_episode_steps,
                                   args.eval_episodes, device=dev, seed=10_000 + step // max(1, args.eval_every))
            evs = evaluate_policy_v(policy_main_stoch, args.env_version, args.size, args.num_obstacles, args.max_episode_steps,
                                    args.eval_episodes, device=dev, seed=10_000 + step // max(1, args.eval_every))
        actor.train()
        row = [step, ev["eval_return"], ev["eval_success"], ev["eval_collision"], ev["eval_length"],
               stats.get("train_main_success", float("nan")), stats.get("train_episodes", 0),
               stats.get("critic_loss", float("nan")), log_alpha[MAIN_TASK].exp().item(), time.time() - t0,
               evs["eval_return"], evs["eval_success"]]
        writer.writerow(row); f.flush()
        print(f"[SAC-X-{args.scheduler} s{args.seed}] steps {step:>8d} | eval ret {ev['eval_return']:7.2f} "
              f"succ {ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} len {ev['eval_length']:.1f} | "
              f"stoch succ {evs['eval_success']:.3f} | train succ {row[5]:.3f} | qloss {row[7]:.4f} | alpha {row[8]:.4f} | {row[9]:.0f}s", flush=True)
        if diag:
            print("    mean Q per intention", np.round(diag["q"], 3), "| mean log pi", np.round(diag["logp"], 2),
                  "| alphas", np.round(log_alpha.exp().tolist(), 4), flush=True)

    def update():
        obs, u, mu_logp, rew, term, done, next_obs = replay.sample(args.batch, args.seq_len)
        B, L = obs.shape[:2]
        o, no = norm(obs), norm(next_obs)
        alpha = log_alpha.exp().detach()
        a = torch.tanh(u)
        with torch.no_grad():
            # pi_T(a|s) for all heads on the stored actions -> truncated importance weights
            mu_p, ls_p = actor(o.reshape(B * L, O))
            pi_logp = MultiHeadActor.log_prob_from_u(mu_p, ls_p, u.reshape(B * L, 1, A)).view(B, L, T)
            c = args.retrace_lambda * torch.exp(torch.clamp(pi_logp - mu_logp.unsqueeze(-1), max=0.0))
            q_targ_sa = critic_targ.min_q(o, a)  # (B, L, T)
            # soft state value of next states for every intention: V_T(s') = Q'_T(s', a_T) - alpha_T log pi_T
            a_next, logp_next = actor.sample(no.reshape(B * L, O))  # (BL, T, A), (BL, T)
            no_rep = no.reshape(B * L, 1, O).expand(-1, T, -1)
            q_next = critic_targ.min_q(no_rep, a_next)  # (BL, T, T): [state, sampled head, q-head]
            v_next = torch.diagonal(q_next, dim1=1, dim2=2) - alpha * logp_next
            v_next = v_next.view(B, L, T)
            # Retrace recursion, backwards over the snippet
            q_ret = torch.zeros(B, L, T, device=dev)
            g = args.gamma
            nxt = rew[:, -1] + g * (1 - term[:, -1]).unsqueeze(-1) * v_next[:, -1]
            q_ret[:, -1] = nxt
            for i in range(L - 2, -1, -1):
                cont = (1 - done[:, i]).unsqueeze(-1)  # trace only continues inside the same episode
                corr = cont * c[:, i + 1] * (q_ret[:, i + 1] - q_targ_sa[:, i + 1])
                q_ret[:, i] = rew[:, i] + g * (1 - term[:, i]).unsqueeze(-1) * (v_next[:, i] + corr)
        q = critic(o, a)  # (2, B, L, T)
        critic_loss = F.mse_loss(q, q_ret.unsqueeze(0).expand_as(q)) * 2
        opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt_critic.step()

        # policy improvement for every intention on all replayed states
        flat = o.reshape(B * L, O)
        a_new, logp_new = actor.sample(flat)
        q_new = critic.min_q(flat.unsqueeze(1).expand(-1, T, -1), a_new)  # (BL, T, T)
        q_new = torch.diagonal(q_new, dim1=1, dim2=2)
        actor_loss = (alpha * logp_new - q_new).mean(0).sum()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt_actor.step()

        alpha_loss = -(log_alpha * (logp_new.detach() + target_entropy)).mean(0).sum()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        soft_update(critic_targ, critic, args.tau)
        diag["q"] = q.detach().min(0).values.mean((0, 1)).tolist()
        diag["logp"] = logp_new.detach().mean(0).tolist()
        return critic_loss.item()

    diag = {}
    t0 = time.time()
    obs = env.reset()
    ep_step = torch.zeros(N, dtype=torch.long, device=dev)
    env_steps, next_eval = 0, 0
    train_main_succ, train_main_eps, total_eps = [], 0, 0
    stats = {}
    closs = float("nan")
    arangeN = torch.arange(N, device=dev)
    while env_steps <= args.total_steps:
        if env_steps >= next_eval:
            stats["train_main_success"] = float(np.mean(train_main_succ[-200:])) if train_main_succ else float("nan")
            stats["train_episodes"] = total_eps
            stats["critic_loss"] = closs
            do_eval(env_steps, stats)
            next_eval += args.eval_every
        # ---- act with the currently scheduled intention of every env
        cur = torch.as_tensor(sched.current, device=dev)
        with torch.no_grad():
            mu, log_std = actor(norm(obs))
            mu_c, ls_c = mu[arangeN, cur], log_std[arangeN, cur]
            u = mu_c + ls_c.exp() * torch.randn_like(mu_c)
            b_logp = MultiHeadActor.log_prob_from_u(mu_c, ls_c, u)
        next_obs, r_env, term, trunc, info = env.step(torch.tanh(u))
        done = term | trunc
        rew = compute_task_rewards(info, args.env_version)
        if args.aux_reward_scale != 1.0:
            rew[:, 1:] = rew[:, 1:] * args.aux_reward_scale
        if args.main_reward == "env":
            rew[:, MAIN_TASK] = args.main_reward_scale * r_env
        rew = rew[:, task_ids]
        replay.add(obs, u, b_logp, rew, term, done, info["final_obs"])
        obs = next_obs
        env_steps += N
        ep_step += 1

        # ---- scheduler bookkeeping (CPU, small)
        main_r = rew[:, MAIN_TASK].cpu().numpy()
        sched.main_return += main_r
        done_np = done.cpu().numpy()
        cur_np = sched.current.copy()
        done_ids = np.nonzero(done_np)[0]
        for e in done_ids:
            total_eps += 1
            if cur_np[e] == MAIN_TASK:
                train_main_succ.append(float(info["reached"][e].item()))
        if len(done_ids):
            sched.end_episode(done_ids)
            ep_step[torch.as_tensor(done_ids, device=dev)] = 0
            sched.choose(done_ids, 0)
        ep_np = ep_step.cpu().numpy()
        switch = np.nonzero((~done_np) & (ep_np % args.xi == 0))[0]
        for e in switch:  # re-schedule every xi steps (period index h = steps // xi)
            sched.choose([e], int(ep_np[e] // args.xi))

        # ---- learn
        if env_steps >= args.warmup and replay.size() >= args.seq_len:
            for _ in range(args.updates_per_step):
                closs = update()

    f.close()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args),
                "scheduler_q": sched.q}, os.path.join(args.ckpt_dir, os.path.splitext(os.path.basename(args.log))[0] + ".pt"))  # named after the log (sac_s0.pt, sacx_q_s0.pt): SAC and SAC-X no longer overwrite each other
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
