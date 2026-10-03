"""
Environment v3: continuous 2-D navigation with randomly sized/placed circular obstacles, optionally moving.
ONE vectorised torch implementation (GPU or CPU); every variant is selected through EnvV3Config
(dataclass <-> YAML <-> CLI). v1/v2 (torch_env.py) are untouched.

World / layout (per episode, per env)
  * arena [0, size]^2, point agent (optional agent_radius > 0 inflates every obstacle).
  * start, goal, obstacle centres sampled in continuous space (no grid).
  * n_obstacles ~ U{n_obs_min..n_obs_max}, radius ~ U[r_min, r_max]; slots >= n are padding (mask 0).
  * guaranteed spacing (vectorised rejection sampling, fixed number of tries -> reset never hangs):
      - obstacle-obstacle surface gap >= min_gap (+ 2*rw_radius in 'random' mode, so the gap >= min_gap at all times),
      - obstacle-wall gap >= wall_gap (+ rw_radius in 'random' mode),
      - obstacle surface to start >= start_clear, to goal >= goal_clear (+ rw_radius in 'random' mode),
      - |start - goal| >= start_goal_min, start/goal >= spawn_margin from the walls.
    If a slot finds no valid candidate within `sample_tries` it is dropped (fewer obstacles in that episode;
    counted in self.stats["dropped_obstacles"]). If no goal candidate satisfies start_goal_min, the farthest
    candidate is used (counted in stats["goal_fallback"]).
  * feasibility (static snapshot): obstacles are pairwise disjoint and do not touch the walls (gaps > 0), so
    the free space (square minus disjoint closed discs) is path-connected; start/goal lie in it. Hence a
    collision-free path from start to goal ALWAYS exists for a point agent (for agent_radius < min(min_gap,
    wall_gap)/2 as well). With min_gap >= the agent's needed passage width this also holds with a margin. In
    'random' mode the guarantee holds at every instant (the extra spawn spacing keeps gaps >= min_gap); in
    'bounce' mode obstacles keep pairwise/wall separation via collisions but a path can be temporarily blocked
    in time and obstacles may touch walls/each other momentarily (no path guarantee for 'bounce').

Obstacle motion (motion = static | random | bounce), obstacle displacement per env step = v (dt = 1):
  * static: no motion.
  * random: smooth random velocity (Ornstein-Uhlenbeck: v <- rw_beta*v + rw_sigma*N(0,I), |v| <= speed_max);
    the obstacle stays within rw_radius of its spawn point (reflection of the radial velocity component at that
    circle). Spawn spacing (above) => never leaves the arena, never overlaps the goal disc or another obstacle.
  * bounce: constant speed ~ U[speed_min, speed_max], random direction; specular reflection at the walls (an
    obstacle may touch a wall) and at the protected goal disc (radius goal_radius + goal_protect: obstacles never
    cover the goal); obstacle-obstacle contacts per obstacle_interaction: 'elastic' (elastic collision along the
    contact normal, mass ~ r^2, overlap resolved by pushing apart in proportion to the other mass) or 'pass'.
    Within one env step an obstacle's path is the straight segment from its old to its new (post-bounce) position.

Agent step (as v2): a in [-1,1]^2 clipped to the unit disc, displacement max_step * a.
Collision checking is exact for the piecewise-linear motion inside a step: obstacle k moves c -> c + dc during
the step (dc = its displacement after motion/bounce for this step), the agent p -> p + d; in the obstacle's frame
the agent moves along p - c + t (d - dc), t in [0,1], and we compute the first entry time into the disc of radius
r_k + agent_radius (segment/circle intersection in relative coordinates). Walls: the segment leaving
[agent_radius, size - agent_radius]^2. Goal: the segment entering the goal disc. The earliest event decides.

Reward (as v2): +goal_value on goal, collision_value on collision, time_penalty every step,
+ shaping * (gamma * Phi(s') - Phi(s)), Phi = -|agent - goal|. Time limit -> truncated (not terminal).

Observation (fixed size 9 + 7*n_obs_max, all roughly in [-1, 1]):
  [agent pos (2, scaled to [-1,1]), goal - agent (2, /size), |goal-agent| (1, /size),
   wall distances x, size-x, y, size-y (4, /size),
   per obstacle slot sorted by surface distance (padding last):
   rel. position (2, /size), radius (1, /r_max), velocity (2, /max(speed_max, 1e-6)), surface distance (1, /size),
   mask (1)]
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass, field, fields

import torch


@dataclass
class EnvV3Config:
    size: float = 10.0
    n_obs_min: int = 3             # defaults = MAIN level (configs/v3_<mode>.yaml); HARD: 4 / 0.8 / 0.5
    n_obs_max: int = 8
    r_min: float = 0.3
    r_max: float = 0.6
    agent_radius: float = 0.0
    min_gap: float = 0.8           # obstacle-obstacle surface gap
    wall_gap: float = 0.5          # obstacle surface - wall
    start_clear: float = 0.5       # obstacle surface - start
    goal_clear: float = 0.8        # obstacle surface - goal centre (at spawn)
    goal_protect: float = 0.2      # bounce mode: obstacles reflect off the disc goal_radius + goal_protect
    start_goal_min: float = 5.0
    spawn_margin: float = 0.5      # start/goal distance from the walls
    sample_tries: int = 64
    motion: str = "static"         # static | random | bounce
    speed_min: float = 0.05
    speed_max: float = 0.2
    rw_radius: float = 0.5         # random mode: max distance from spawn
    rw_beta: float = 0.8           # random mode: velocity persistence
    rw_sigma: float = 0.08         # random mode: velocity noise std per step
    obstacle_interaction: str = "elastic"  # bounce mode: elastic | pass
    max_step: float = 1.0
    goal_radius: float = 0.5
    goal_value: float = 10.0
    collision_value: float = -10.0
    time_penalty: float = -0.05
    shaping: float = 1.0
    gamma: float = 0.99
    max_episode_steps: int = 100

    # ---------- config helpers
    @classmethod
    def from_yaml(cls, path, **overrides):
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def to_yaml(self, path):
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)

    @classmethod
    def add_cli_args(cls, ap: argparse.ArgumentParser, prefix="env-"):
        for fl in fields(cls):
            ap.add_argument(f"--{prefix}{fl.name.replace('_', '-')}", type=type(fl.default), default=None,
                            help=f"EnvV3Config.{fl.name} (default {fl.default})")
        ap.add_argument("--env-config", default=None, help="YAML file with EnvV3Config fields")

    @classmethod
    def from_cli(cls, args, prefix="env_"):
        ov = {fl.name: getattr(args, prefix + fl.name) for fl in fields(cls)}
        if getattr(args, "env_config", None):
            return cls.from_yaml(args.env_config, **ov)
        return cls(**{k: v for k, v in ov.items() if v is not None})

    def validate(self):
        assert self.motion in ("static", "random", "bounce"), self.motion
        assert self.obstacle_interaction in ("elastic", "pass")
        assert 0 <= self.n_obs_min <= self.n_obs_max and 0 < self.r_min <= self.r_max
        assert self.speed_min <= self.speed_max


def segment_circle_entry(p, d, c, r):
    """First t in [0,1] at which p + t d enters the circle (c, r) (inf if never); t = 0 if p is inside.
    p, d: (..., 2), c: broadcastable (..., 2), r: broadcastable (...)."""
    f = p - c
    a = (d * d).sum(-1)
    b = 2 * (f * d).sum(-1)
    cc = (f * f).sum(-1) - r * r
    disc = b * b - 4 * a * cc
    sq = torch.sqrt(torch.clamp(disc, min=0))
    t1 = (-b - sq) / torch.clamp(2 * a, min=1e-12)
    inside = cc <= 0
    hit = ((disc >= 0) & (a > 0) & (t1 >= 0) & (t1 <= 1)) | inside
    t = torch.where(inside, torch.zeros_like(t1), t1)
    return torch.where(hit, t, torch.full_like(t, float("inf")))


class EnvV3:
    def __init__(self, num_envs, cfg: EnvV3Config | None = None, device="cuda", seed=0, dtype=torch.float32):
        self.cfg = cfg = cfg or EnvV3Config()
        cfg.validate()
        self.n, self.K = num_envs, cfg.n_obs_max
        self.device, self.dtype = torch.device(device), dtype
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(seed)
        n, K = self.n, max(self.K, 1)
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)
        self.agent, self.goal = z(n, 2), z(n, 2)
        self.obs_pos, self.obs_vel, self.spawn = z(n, K, 2), z(n, K, 2), z(n, K, 2)
        self.obs_r = z(n, K)
        self.active = torch.zeros(n, K, dtype=torch.bool, device=self.device)
        self.t = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stats = {"resets": 0, "dropped_obstacles": 0, "goal_fallback": 0}
        self.obs_dim, self.act_dim = 9 + 7 * self.K, 2

    # ---------- random helpers
    def _u(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _n(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    # ---------- layout sampling
    def _sample(self, m):
        c, T, K = self.cfg, self.cfg.sample_tries, max(self.K, 1)
        S = c.size
        rw = c.rw_radius if c.motion == "random" else 0.0
        lo, hi = c.spawn_margin, S - c.spawn_margin
        start = lo + (hi - lo) * self._u(m, 2)
        cand = lo + (hi - lo) * self._u(m, T, 2)
        dist = (cand - start[:, None]).norm(dim=-1)
        ok = dist >= c.start_goal_min
        first = torch.where(ok.any(1), ok.float().argmax(1), dist.argmax(1))
        self.stats["goal_fallback"] += int((~ok.any(1)).sum())
        goal = cand[torch.arange(m, device=self.device), first]
        pos, rad = torch.zeros(m, K, 2, device=self.device, dtype=self.dtype), torch.zeros(m, K, device=self.device, dtype=self.dtype)
        act = torch.zeros(m, K, dtype=torch.bool, device=self.device)
        n_req = torch.randint(c.n_obs_min, c.n_obs_max + 1, (m,), generator=self.gen, device=self.device)
        ar = torch.arange(m, device=self.device)
        for k in range(self.K):
            r = c.r_min + (c.r_max - c.r_min) * self._u(m, T)
            wall = r + c.wall_gap + rw + c.agent_radius
            p = wall[..., None] + (S - 2 * wall[..., None]) * self._u(m, T, 2)
            okk = (S - 2 * wall) > 0
            okk &= (p - start[:, None]).norm(dim=-1) - r >= c.start_clear + rw + c.agent_radius
            okk &= (p - goal[:, None]).norm(dim=-1) - r >= c.goal_clear + rw + c.agent_radius
            if k > 0:
                dd = (p[:, :, None] - pos[:, None, :k]).norm(dim=-1) - r[..., None] - rad[:, None, :k]  # (m,T,k)
                okk &= ((dd >= c.min_gap + 2 * rw + 2 * c.agent_radius) | ~act[:, None, :k]).all(-1)
            want = k < n_req
            got = okk.any(1) & want
            j = okk.float().argmax(1)
            pos[:, k] = torch.where(got[:, None], p[ar, j], pos[:, k])
            rad[:, k] = torch.where(got, r[ar, j], rad[:, k])
            act[:, k] = got
            self.stats["dropped_obstacles"] += int((want & ~got).sum())
        vel = torch.zeros_like(pos)
        if c.motion == "bounce":
            sp = c.speed_min + (c.speed_max - c.speed_min) * self._u(m, K)
            th = 2 * math.pi * self._u(m, K)
            vel = torch.stack((th.cos(), th.sin()), -1) * sp[..., None]
        elif c.motion == "random":
            vel = c.speed_max * 0.5 * self._u(m, K)[..., None] * torch.nn.functional.normalize(self._n(m, K, 2), dim=-1)
        vel = vel * act[..., None]
        return start, goal, pos, rad, act, vel

    def reset_idx(self, idx):
        if idx.numel() == 0:
            return
        s, g, p, r, a, v = self._sample(idx.numel())
        self.agent[idx], self.goal[idx], self.obs_pos[idx], self.obs_r[idx] = s, g, p, r
        self.active[idx], self.obs_vel[idx], self.spawn[idx] = a, v, p.clone()
        self.t[idx] = 0
        self.stats["resets"] += idx.numel()

    def reset(self):
        self.reset_idx(torch.arange(self.n, device=self.device))
        return self.get_obs()

    def set_state(self, agent, goal, obs_pos, obs_r, active=None, obs_vel=None, t=None):
        """for tests / external layouts: tensors with leading dim num_envs (obstacle dims K)."""
        K = obs_pos.shape[1]
        self.agent = agent.to(self.device, self.dtype).clone(); self.goal = goal.to(self.device, self.dtype).clone()
        self.obs_pos[:, :K] = obs_pos.to(self.device, self.dtype); self.obs_r[:, :K] = obs_r.to(self.device, self.dtype)
        self.active[:] = False
        self.active[:, :K] = True if active is None else active.to(self.device)
        self.obs_vel[:] = 0
        if obs_vel is not None:
            self.obs_vel[:, :K] = obs_vel.to(self.device, self.dtype)
        self.spawn = self.obs_pos.clone()
        self.t[:] = 0 if t is None else t

    # ---------- obstacle motion (returns displacement for this step)
    def _move_obstacles(self):
        c = self.cfg
        if c.motion == "static" or self.K == 0:
            return torch.zeros_like(self.obs_pos)
        p0, v, r, act = self.obs_pos, self.obs_vel, self.obs_r, self.active
        if c.motion == "random":
            v = c.rw_beta * v + c.rw_sigma * self._n(*v.shape)
            sp = v.norm(dim=-1, keepdim=True)
            v = v * torch.clamp(c.speed_max / sp.clamp(min=1e-9), max=1.0)
            q = p0 + v
            off = q - self.spawn
            dist = off.norm(dim=-1, keepdim=True)
            out = dist > c.rw_radius
            nrm = off / dist.clamp(min=1e-9)
            vr = (v * nrm).sum(-1, keepdim=True)
            v = torch.where(out & (vr > 0), v - 2 * vr * nrm, v)          # reflect radial component
            q = torch.where(out, self.spawn + nrm * c.rw_radius, q)        # stay on the allowed circle
        else:  # bounce: free flight -> obstacle-obstacle contacts -> walls -> protected goal disc
            q = p0 + v
            if c.obstacle_interaction == "elastic" and self.K > 1:
                dv = q[:, :, None] - q[:, None, :]                       # (n,K,K,2) i - j
                dd = dv.norm(dim=-1)
                rs = r[:, :, None] + r[:, None, :]
                pair = (dd < rs) & act[:, :, None] & act[:, None, :]
                pair &= ~torch.eye(self.K, dtype=torch.bool, device=self.device)
                nrm = dv / dd.clamp(min=1e-9)[..., None]
                relv = ((v[:, :, None] - v[:, None, :]) * nrm).sum(-1)   # approaching if < 0
                m = r ** 2
                mi, mj = m[:, :, None], m[:, None, :]
                coef = torch.where(pair & (relv < 0), 2 * mj / (mi + mj) * relv, torch.zeros_like(relv))
                v = v - (coef[..., None] * nrm).sum(2)
                push = torch.where(pair, (rs - dd) * mj / (mi + mj), torch.zeros_like(dd))
                q = q + (push[..., None] * nrm).sum(2)
            lo, hi = r[..., None], c.size - r[..., None]
            below, above = q < lo, q > hi
            v = torch.where(below & (v < 0) | above & (v > 0), -v, v)
            q = torch.where(below, torch.minimum(2 * lo - q, hi), torch.where(above, torch.maximum(2 * hi - q, lo), q))
            R = r + c.goal_radius + c.goal_protect
            off = q - self.goal[:, None]
            dist = off.norm(dim=-1)
            hit = dist < R
            nrm = off / dist.clamp(min=1e-9)[..., None]
            vn = (v * nrm).sum(-1, keepdim=True)
            v = torch.where(hit[..., None] & (vn < 0), v - 2 * vn * nrm, v)
            q = torch.where(hit[..., None], self.goal[:, None] + nrm * R[..., None], q)
            # goal close to a wall: the projection may leave the arena -> clamp, then slide along that wall
            # (keep the clamped coordinate, move the other one onto the protected circle, on the current side)
            qc = torch.minimum(torch.maximum(q, lo), hi)
            clamped = (qc != q)
            q = qc
            g = self.goal[:, None]
            for ax in (0, 1):
                ot = 1 - ax
                rem = R ** 2 - (q[..., ax] - g[..., ax]) ** 2
                inside = clamped[..., ax] & ((q - g).norm(dim=-1) < R) & (rem > 0)
                side = torch.where(q[..., ot] >= g[..., ot], 1.0, -1.0)
                new_ot = g[..., ot] + side * torch.sqrt(rem.clamp(min=0))
                q[..., ot] = torch.where(inside, new_ot, q[..., ot])
            q = torch.minimum(torch.maximum(q, lo), hi)
        q = torch.where(act[..., None], q, p0)
        v = torch.where(act[..., None], v, torch.zeros_like(v))
        dc = q - p0
        self.obs_pos, self.obs_vel = q, v
        return dc

    # ---------- step
    def step(self, action, auto_reset=True):
        c = self.cfg
        a = action.to(self.device, self.dtype).clamp(-1, 1)
        a = a / torch.clamp(a.norm(dim=1, keepdim=True), min=1.0)
        p = self.agent.clone()
        d = c.max_step * a
        q = p + d
        c0 = self.obs_pos.clone()
        dc = self._move_obstacles()
        # walls
        lo, hi = c.agent_radius, c.size - c.agent_radius
        safe_d = torch.where(d == 0, torch.ones_like(d), d)
        inf = torch.full_like(d, float("inf"))
        t_ax = torch.where(d > 0, (hi - p) / safe_d, torch.where(d < 0, (lo - p) / safe_d, inf))
        t_wall = t_ax.min(1).values
        t_wall = torch.where(t_wall <= 1, t_wall, torch.full_like(t_wall, float("inf")))
        # obstacles: relative motion within the step
        if self.K > 0:
            rel_p = p[:, None] - c0
            rel_d = d[:, None] - dc
            t_o = segment_circle_entry(rel_p, rel_d, torch.zeros_like(rel_p), self.obs_r + c.agent_radius)
            t_o = torch.where(self.active, t_o, torch.full_like(t_o, float("inf"))).min(1).values
        else:
            t_o = torch.full_like(t_wall, float("inf"))
        t_goal = segment_circle_entry(p, d, self.goal, torch.full_like(t_wall, c.goal_radius))
        t_col = torch.minimum(t_wall, t_o)
        reached = torch.isfinite(t_goal) & (t_goal <= t_col)
        collided = torch.isfinite(t_col) & ~reached
        term = reached | collided
        rew = torch.full((self.n,), c.time_penalty, device=self.device, dtype=self.dtype)
        rew = rew + reached.to(self.dtype) * c.goal_value + collided.to(self.dtype) * c.collision_value
        if c.shaping != 0:
            rew = rew + c.shaping * (c.gamma * -(q - self.goal).norm(dim=1) + (p - self.goal).norm(dim=1))
        self.agent = q
        self.t += 1
        trunc = (~term) & (self.t >= c.max_episode_steps)
        final_obs = self.get_obs()
        info = {"final_obs": final_obs, "prev_agent": p, "agent": q.clone(), "collided": collided,
                "reached": reached, "obs_disp": dc, "target": self.goal.clone()}
        done = term | trunc
        if auto_reset and done.any():
            self.reset_idx(torch.nonzero(done).squeeze(1))
            obs = self.get_obs()
        else:
            obs = final_obs
        return obs, rew, term, trunc, info

    # ---------- observation
    def get_obs(self):
        c, S = self.cfg, self.cfg.size
        p, g = self.agent, self.goal
        rel_g = g - p
        base = [2 * p / S - 1, rel_g / S, rel_g.norm(dim=1, keepdim=True) / S,
                p[:, :1] / S, (S - p[:, :1]) / S, p[:, 1:] / S, (S - p[:, 1:]) / S]
        if self.K == 0:
            return torch.cat(base, 1)
        rel = self.obs_pos - p[:, None]
        sd = rel.norm(dim=-1) - self.obs_r
        key = torch.where(self.active, sd, torch.full_like(sd, float("inf")))
        order = key.argsort(dim=1)
        gat = lambda x: torch.gather(x, 1, order) if x.dim() == 2 else torch.gather(x, 1, order[..., None].expand_as(x))
        rel, r, v, sd, m = gat(rel), gat(self.obs_r), gat(self.obs_vel), gat(sd), gat(self.active).to(self.dtype)
        vs = max(c.speed_max, 1e-6)
        slot = torch.cat([rel / S, (r / c.r_max)[..., None], v / vs, (sd / S)[..., None], m[..., None]], -1)
        slot = slot * m[..., None]
        return torch.cat(base + [slot.reshape(self.n, -1)], 1)

    def state(self):
        """snapshot for renderer / planners (CPU tensors)."""
        return {"agent": self.agent.cpu().clone(), "goal": self.goal.cpu().clone(), "obs_pos": self.obs_pos.cpu().clone(),
                "obs_r": self.obs_r.cpu().clone(), "active": self.active.cpu().clone(), "obs_vel": self.obs_vel.cpu().clone()}


def evaluate_v3(policy_fn, cfg: EnvV3Config, num_episodes=512, device="cuda", seed=2026, record=False, per_episode=False):
    """Deterministic rollout of policy_fn(obs, env) on fixed layouts; return measured WITHOUT shaping.
    Returns dict with success/collision/timeout rates and (record=True) per-step states for rendering."""
    cfg_e = dataclasses.replace(cfg, shaping=0.0)
    env = EnvV3(num_episodes, cfg_e, device=device, seed=seed)
    obs = env.reset()
    alive = torch.ones(num_episodes, dtype=torch.bool, device=env.device)
    ret = torch.zeros(num_episodes, device=env.device); length = torch.zeros(num_episodes, device=env.device)
    succ = torch.zeros_like(alive); coll = torch.zeros_like(alive)
    frames = [env.state()] if record else None
    plen = torch.zeros(num_episodes, device=env.device)
    alive_hist = []
    for _ in range(cfg.max_episode_steps):
        env._alive_np = alive.cpu().numpy()          # lets planners skip finished episodes
        obs, r, term, trunc, info = env.step(policy_fn(obs, env), auto_reset=False)
        ret += r * alive; length += alive.float()
        plen += (info["agent"] - info["prev_agent"]).norm(dim=1) * alive
        succ |= info["reached"] & alive; coll |= info["collided"] & alive
        if record:
            frames.append(env.state()); alive_hist.append(alive.cpu().clone())
        alive &= ~(term | trunc)
        if not alive.any():
            break
    out = {"eval_return": ret.mean().item(), "eval_success": succ.float().mean().item(),
           "eval_collision": coll.float().mean().item(), "eval_timeout": 1 - succ.float().mean().item() - coll.float().mean().item(),
           "eval_length": length.mean().item(), "n": num_episodes}
    if record or per_episode:
        out["outcome"] = ["goal" if s else "collision" if cc else "timeout" for s, cc in zip(succ.tolist(), coll.tolist())]
        out["length"] = length.long().tolist(); out["path_len"] = plen.tolist()
    if record:
        out["frames"], out["alive"] = frames, alive_hist
        out["outcome"] = ["goal" if s else "collision" if cc else "timeout" for s, cc in zip(succ.tolist(), coll.tolist())]
        out["length"] = length.long().tolist()
    return out
