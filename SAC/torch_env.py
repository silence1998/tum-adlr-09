"""
Vectorized, GPU-friendly re-implementation of `environment.GridWorldEnv` in pure torch.

The original env is a tiny integer grid world (agent, target, `num_obstacles` static obstacles, continuous 2D
action that is scaled by `action_step_scaling` and rounded to a grid move).  All of its logic is a handful of
elementwise comparisons, so it can be stepped for thousands of environments in parallel on the GPU.

Dynamics / rewards are identical to GridWorldEnv.step() with the default reward_parameters
(sparse rewards: +target_value on reaching the target, collision_value on hitting an obstacle or leaving the grid,
0 otherwise; both end the episode).  Differences that are *fixes* (also applied to environment.py):
  * actions are clipped to [-1, 1] before scaling (the policy is tanh-squashed anyway; PPO samples are not),
  * obstacles are sampled without overlapping each other (the original overlap check was a no-op),
  * episodes are truncated after `max_episode_steps` (reported as `truncated`, NOT as terminal).

Rounding: torch.round and np.round both use round-half-to-even, so the discretisation matches bit-for-bit.

Observation (same layout as the training scripts build it from the dict obs):
    [agent_x, agent_y, target_x, target_y, obs0_x, obs0_y, ..., obsK_x, obsK_y]   (raw grid coordinates, float)
Use `normalize_obs` to map it to [-1, 1].
"""

import torch


def normalize_obs(obs, size):
    """Map raw grid coordinates in [0, size-1] to [-1, 1] (observation normalization)."""
    return obs * (2.0 / (size - 1)) - 1.0


def sort_obstacles_by_distance(obs):
    """Reorder the obstacle slots of a raw observation by distance to the agent (nearest first).
    Makes the input permutation-invariant w.r.t. obstacle order: the critic sees the most relevant obstacle always in
    the same slot instead of having to learn collisions separately for every slot from rare events."""
    lead = obs.shape[:-1]
    pts = obs.reshape(*lead, -1, 2)
    obst = pts[..., 2:, :]
    d = (obst - pts[..., 0:1, :]).norm(dim=-1)
    idx = d.argsort(dim=-1).unsqueeze(-1).expand_as(obst)
    return torch.cat((pts[..., :2, :], torch.gather(obst, -2, idx)), dim=-2).reshape(obs.shape)


def obs_features(obs, size, sort_obstacles=False):
    """
    Network input used by the fixed SAC-X and PPO: normalised absolute coordinates plus egocentric vectors
    (target - agent, obstacle_k - agent) scaled to [-1, 1]. Same information as the raw observation, just a
    representation an MLP can use directly (the absolute-only input forces the net to learn subtraction).
    """
    if sort_obstacles:
        obs = sort_obstacles_by_distance(obs)
    lead = obs.shape[:-1]
    pts = obs.reshape(*lead, -1, 2)
    rel = (pts[..., 1:, :] - pts[..., 0:1, :]).reshape(*lead, -1) / (size - 1)
    return torch.cat((normalize_obs(obs, size), rel), dim=-1)


def feature_dim(num_obstacles):
    return 2 * (4 + 2 * num_obstacles) - 2


class TorchGridWorld:
    def __init__(self, num_envs, size=10, num_obstacles=5, max_episode_steps=100, action_step_scaling=2,
                 target_value=10.0, collision_value=-50.0, device="cuda", seed=0,
                 time_penalty=0.0, shaping_coef=0.0, shaping_gamma=0.99):
        """
        Optional env-design changes (all default OFF = original task):
          time_penalty : added every step (e.g. -0.1) so that standing still is no longer a safe optimum
          shaping_coef : potential-based shaping  k * (gamma * Phi(s') - Phi(s)),  Phi(s) = -||agent - target|| / (size-1),
                         Phi(terminal) = 0. Policy-invariant (Ng et al. 1999) but gives a dense progress signal.
          collision_value can be made less extreme (e.g. -10 instead of -50).
        """
        self.time_penalty, self.shaping_coef, self.shaping_gamma = time_penalty, shaping_coef, shaping_gamma
        self.num_envs = num_envs
        self.size = size
        self.num_obstacles = num_obstacles
        self.max_episode_steps = max_episode_steps
        self.action_step_scaling = action_step_scaling
        self.target_value = target_value
        self.collision_value = collision_value
        self.device = torch.device(device)
        self.obs_dim = 4 + 2 * num_obstacles
        self.act_dim = 2
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        self.agent = torch.zeros(num_envs, 2, device=self.device)
        self.target = torch.zeros(num_envs, 2, device=self.device)
        self.obstacles = torch.zeros(num_envs, num_obstacles, 2, device=self.device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ #
    def _sample_layouts(self, n):
        """Agent, target and obstacles on pairwise distinct cells, uniformly at random."""
        k = 2 + self.num_obstacles
        cells = torch.rand(n, self.size * self.size, generator=self.gen, device=self.device).argsort(dim=1)[:, :k]
        xy = torch.stack((cells // self.size, cells % self.size), dim=-1).float()  # (n, k, 2)
        return xy[:, 0], xy[:, 1], xy[:, 2:]

    def reset_idx(self, idx):
        if idx.numel() == 0:
            return
        a, g, o = self._sample_layouts(idx.numel())
        self.agent[idx] = a
        self.target[idx] = g
        self.obstacles[idx] = o
        self.t[idx] = 0

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_obs()

    def get_obs(self):
        return torch.cat((self.agent, self.target, self.obstacles.reshape(self.num_envs, -1)), dim=1)

    def set_state(self, agent, target, obstacles, t=None):
        """For tests: overwrite the full state."""
        self.agent[:] = agent
        self.target[:] = target
        self.obstacles[:] = obstacles
        self.t[:] = 0 if t is None else t

    # ------------------------------------------------------------------ #
    def step(self, action, auto_reset=True):
        """
        action: (num_envs, 2) tensor in [-1, 1] (clipped).
        returns obs, reward, terminated, truncated, info
          info['final_obs']: observation *before* auto-reset (use for bootstrapping on truncation)
          info['prev_agent'], info['agent'], info['target'], info['collided'], info['reached']
        """
        action = action.to(self.device, torch.float32).clamp(-1.0, 1.0)
        move = torch.round(self.action_step_scaling * action)
        prev_agent = self.agent.clone()
        new_agent = prev_agent + move

        oob = ((new_agent < 0) | (new_agent > self.size - 1)).any(dim=1)
        hit = (new_agent.unsqueeze(1) == self.obstacles).all(dim=2).any(dim=1)
        collided = oob | hit
        reached = (~collided) & (new_agent == self.target).all(dim=1)

        reward = torch.zeros(self.num_envs, device=self.device)
        reward = torch.where(collided, torch.full_like(reward, self.collision_value), reward)
        reward = torch.where(reached, torch.full_like(reward, self.target_value), reward)
        terminated = collided | reached
        if self.time_penalty != 0.0:
            reward = reward + self.time_penalty
        if self.shaping_coef != 0.0:
            phi_prev = -(prev_agent - self.target).norm(dim=1) / (self.size - 1)
            phi_new = torch.where(terminated, torch.zeros_like(phi_prev),
                                  -(new_agent - self.target).norm(dim=1) / (self.size - 1))
            reward = reward + self.shaping_coef * (self.shaping_gamma * phi_new - phi_prev)

        self.agent = new_agent
        self.t += 1
        truncated = (~terminated) & (self.t >= self.max_episode_steps)

        final_obs = self.get_obs()
        info = {"final_obs": final_obs, "prev_agent": prev_agent, "agent": new_agent.clone(),
                "target": self.target.clone(), "collided": collided, "reached": reached}
        if auto_reset:
            done_idx = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(1)
            self.reset_idx(done_idx)
            obs = self.get_obs()
        else:
            obs = final_obs
        return obs, reward, terminated, truncated, info


@torch.no_grad()
def evaluate_policy(policy_fn, size=10, num_obstacles=5, max_episode_steps=100, num_episodes=512,
                    device="cuda", seed=12345):
    """
    Run `num_episodes` episodes in parallel (one per env, no auto reset) with a deterministic policy.
    policy_fn: raw obs tensor (N, obs_dim) -> action tensor (N, 2) in [-1, 1].
    Returns dict with mean return (original env reward scale), success rate, collision rate, mean length.
    """
    env = TorchGridWorld(num_episodes, size, num_obstacles, max_episode_steps, device=device, seed=seed)
    obs = env.reset()
    alive = torch.ones(num_episodes, dtype=torch.bool, device=env.device)
    ret = torch.zeros(num_episodes, device=env.device)
    length = torch.zeros(num_episodes, device=env.device)
    success = torch.zeros(num_episodes, dtype=torch.bool, device=env.device)
    collision = torch.zeros(num_episodes, dtype=torch.bool, device=env.device)
    for _ in range(max_episode_steps):
        act = policy_fn(obs)
        obs, r, term, trunc, info = env.step(act, auto_reset=False)
        ret += r * alive
        length += alive.float()
        success |= info["reached"] & alive
        collision |= info["collided"] & alive
        alive &= ~(term | trunc)
        if not alive.any():
            break
    return {"eval_return": ret.mean().item(), "eval_success": success.float().mean().item(),
            "eval_collision": collision.float().mean().item(), "eval_length": length.mean().item()}


# ====================================================================================================== #
#  Version 2: continuous obstacle-avoidance / goal-reaching world (reference implementation)
# ====================================================================================================== #
V2_DEFAULTS = dict(max_step=1.0, obstacle_radius=0.4, goal_radius=0.5, target_value=10.0, collision_value=-10.0,
                   time_penalty=-0.05, shaping_coef=1.0)
V1_DEFAULTS = dict(target_value=10.0, collision_value=-50.0, time_penalty=0.0, shaping_coef=0.0)


def _circle_entry_time(p, d, c, r):
    """First t in [0,1] at which segment p + t d enters the circle (c, r); inf if it does not.
    p, d: (N, 2); c: (N, K, 2) -> (N, K)."""
    f = p.unsqueeze(1) - c
    dd = d.unsqueeze(1)
    a = (dd * dd).sum(-1)
    b = 2 * (f * dd).sum(-1)
    cc = (f * f).sum(-1) - r * r
    disc = b * b - 4 * a * cc
    sq = torch.sqrt(torch.clamp(disc, min=0))
    t1 = (-b - sq) / torch.clamp(2 * a, min=1e-12)
    inside = cc <= 0
    hit = ((disc >= 0) & (a > 0) & (t1 >= 0) & (t1 <= 1)) | inside
    t = torch.where(inside, torch.zeros_like(t1), t1)
    return torch.where(hit, t, torch.full_like(t1, float("inf")))


class TorchContinuousWorld(TorchGridWorld):
    """
    Env version 2 (default for new runs). Same layout sampling and observation layout as version 1, but:
      * continuous agent position (no rounding); action a in [-1,1]^2, clipped to the unit disc
        (a / max(1, ||a||)) -> displacement max_step * a  (Euclidean max step length = max_step),
      * world = [-0.5, size-0.5]^2 (cell centres are the integer coordinates, so obs normalisation is unchanged),
      * obstacles are discs of radius obstacle_radius at cell centres; collision is checked along the WHOLE movement
        segment (segment/disc intersection) and against the walls (segment leaving the box),
      * goal reached when the segment enters the disc of radius goal_radius around the goal; whichever event happens
        first along the segment (goal vs. collision) decides the outcome,
      * reward: +target_value on goal, collision_value on collision, time_penalty every step,
        + shaping_coef * (gamma * Phi(s') - Phi(s)),  Phi(s) = -||agent - goal|| (in cells)  [potential-based shaping;
        Phi is evaluated at the real next position also on terminal steps, so a collision is never "paid" by the
        shaping term]
      * time limit -> truncated (not terminal).
    """

    def __init__(self, num_envs, size=10, num_obstacles=5, max_episode_steps=100, device="cuda", seed=0,
                 max_step=None, obstacle_radius=None, goal_radius=None, target_value=None, collision_value=None,
                 time_penalty=None, shaping_coef=None, shaping_gamma=0.99, dtype=torch.float32, **_):
        cfg = dict(V2_DEFAULTS)
        for k, v in dict(max_step=max_step, obstacle_radius=obstacle_radius, goal_radius=goal_radius,
                         target_value=target_value, collision_value=collision_value, time_penalty=time_penalty,
                         shaping_coef=shaping_coef).items():
            if v is not None:
                cfg[k] = v
        super().__init__(num_envs, size, num_obstacles, max_episode_steps, device=device, seed=seed,
                         target_value=cfg["target_value"], collision_value=cfg["collision_value"],
                         time_penalty=cfg["time_penalty"], shaping_coef=cfg["shaping_coef"],
                         shaping_gamma=shaping_gamma)
        self.max_step, self.obstacle_radius, self.goal_radius = cfg["max_step"], cfg["obstacle_radius"], cfg["goal_radius"]
        self.cfg = cfg
        self.dtype = dtype
        self.agent = self.agent.to(dtype); self.target = self.target.to(dtype); self.obstacles = self.obstacles.to(dtype)

    def _sample_layouts(self, n):
        a, g, o = super()._sample_layouts(n)
        return a.to(self.dtype), g.to(self.dtype), o.to(self.dtype)

    def step(self, action, auto_reset=True):
        action = action.to(self.device, self.dtype).clamp(-1.0, 1.0)
        # Euclidean step-length limit (unit disc): with a per-axis box the diagonal moves are sqrt(2) longer, so
        # tanh-squashed policies (SAC/SAC-X) pushed both axes to +-1, saturated tanh (vanishing gradients) and
        # could only move diagonally. Clipping to the disc gives the same max speed in every direction.
        norm = action.norm(dim=1, keepdim=True)
        action = action / torch.clamp(norm, min=1.0)
        p = self.agent.clone()
        d = self.max_step * action
        q = p + d
        lo, hi = -0.5, self.size - 0.5
        # wall: first time the (convex) segment leaves the box
        inf = torch.full_like(d, float("inf"))
        safe_d = torch.where(d == 0, torch.ones_like(d), d)
        t_axis = torch.where(d > 0, (hi - p) / safe_d, torch.where(d < 0, (lo - p) / safe_d, inf))
        t_wall = t_axis.min(dim=1).values
        t_wall = torch.where(t_wall <= 1, t_wall, torch.full_like(t_wall, float("inf")))
        t_obs = _circle_entry_time(p, d, self.obstacles, self.obstacle_radius).min(dim=1).values
        t_goal = _circle_entry_time(p, d, self.target.unsqueeze(1), self.goal_radius).squeeze(1)
        t_col = torch.minimum(t_wall, t_obs)
        reached = torch.isfinite(t_goal) & (t_goal <= t_col)
        collided = torch.isfinite(t_col) & ~reached
        terminated = reached | collided

        reward = torch.full((self.num_envs,), self.time_penalty, device=self.device, dtype=self.dtype)
        reward = reward + torch.where(reached, torch.full_like(reward, self.target_value), torch.zeros_like(reward))
        reward = reward + torch.where(collided, torch.full_like(reward, self.collision_value), torch.zeros_like(reward))
        if self.shaping_coef != 0.0:
            phi_p = -(p - self.target).norm(dim=1)
            phi_q = -(q - self.target).norm(dim=1)
            reward = reward + self.shaping_coef * (self.shaping_gamma * phi_q - phi_p)

        self.agent = q
        self.t += 1
        truncated = (~terminated) & (self.t >= self.max_episode_steps)
        final_obs = self.get_obs()
        info = {"final_obs": final_obs, "prev_agent": p, "agent": q.clone(), "target": self.target.clone(),
                "collided": collided, "reached": reached,
                "obstacles": self.obstacles.clone(), "obstacle_radius": self.obstacle_radius}  # for obstacle-aware aux rewards
        if auto_reset:
            self.reset_idx(torch.nonzero(terminated | truncated, as_tuple=False).squeeze(1))
            obs = self.get_obs()
        else:
            obs = final_obs
        return obs.float(), reward.float(), terminated, truncated, info


def make_env(version, num_envs, size=10, num_obstacles=5, max_episode_steps=100, device="cuda", seed=0, **kw):
    """version 1 = original rounded grid env (reproduces earlier results), version 2 = continuous env (default)."""
    kw = {k: v for k, v in kw.items() if v is not None}
    if version == 1:
        cfg = dict(V1_DEFAULTS); cfg.update({k: v for k, v in kw.items() if k in V1_DEFAULTS or k == "shaping_gamma"})
        return TorchGridWorld(num_envs, size, num_obstacles, max_episode_steps, device=device, seed=seed, **cfg)
    return TorchContinuousWorld(num_envs, size, num_obstacles, max_episode_steps, device=device, seed=seed, **kw)


@torch.no_grad()
def evaluate_policy_v(policy_fn, version=1, size=10, num_obstacles=5, max_episode_steps=100, num_episodes=512,
                      device="cuda", seed=12345):
    """
    Evaluation for either env version. Return is measured WITHOUT shaping:
      v1: original reward (+10 goal / -50 collision), v2: +10 goal / -10 collision / -0.05 per step.
    """
    if version == 1:
        return evaluate_policy(policy_fn, size, num_obstacles, max_episode_steps, num_episodes, device, seed)
    env = TorchContinuousWorld(num_episodes, size, num_obstacles, max_episode_steps, device=device, seed=seed,
                               shaping_coef=0.0)
    obs = env.reset()
    alive = torch.ones(num_episodes, dtype=torch.bool, device=env.device)
    ret = torch.zeros(num_episodes, device=env.device)
    length = torch.zeros(num_episodes, device=env.device)
    success = torch.zeros_like(alive); collision = torch.zeros_like(alive)
    for _ in range(max_episode_steps):
        obs, r, term, trunc, info = env.step(policy_fn(obs), auto_reset=False)
        ret += r * alive; length += alive.float()
        success |= info["reached"] & alive; collision |= info["collided"] & alive
        alive &= ~(term | trunc)
        if not alive.any():
            break
    return {"eval_return": ret.mean().item(), "eval_success": success.float().mean().item(),
            "eval_collision": collision.float().mean().item(), "eval_length": length.mean().item()}
