"""
CPU / NumPy version of env version 2 (continuous obstacle avoidance + goal reaching), written independently of
torch_env.TorchContinuousWorld (which is the reference used for training) and used to cross-check it
(test_torch_env.py). Single environment, gym-style API (obs as flat array, like the training scripts use).

World: [-0.5, size-0.5]^2, agent / goal / obstacle centres start on distinct integer cells.
Action a in [-1,1]^2, clipped to the unit disc -> displacement max_step * a (continuous, no rounding).
Collision: movement segment intersects an obstacle disc (radius obstacle_radius) or leaves the box.
Goal: segment enters the goal disc (radius goal_radius) before any collision.
Reward: +target_value (goal), collision_value (collision), time_penalty every step,
        + shaping_coef * (gamma * Phi(s') - Phi(s)), Phi(s) = -||agent - goal||.
Time limit max_episode_steps -> truncated.
"""
import math
import numpy as np


class ContinuousGridWorldEnv:
    def __init__(self, size=10, num_obstacles=5, max_episode_steps=100, max_step=1.0, obstacle_radius=0.4,
                 goal_radius=0.5, target_value=10.0, collision_value=-10.0, time_penalty=-0.05, shaping_coef=1.0,
                 gamma=0.99):
        self.size, self.num_obstacles, self.max_episode_steps = size, num_obstacles, max_episode_steps
        self.max_step, self.obstacle_radius, self.goal_radius = max_step, obstacle_radius, goal_radius
        self.target_value, self.collision_value = target_value, collision_value
        self.time_penalty, self.shaping_coef, self.gamma = time_penalty, shaping_coef, gamma
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cells = self.rng.choice(self.size * self.size, size=2 + self.num_obstacles, replace=False)
        xy = np.stack((cells // self.size, cells % self.size), axis=1).astype(np.float64)
        self.agent, self.target, self.obstacles = xy[0].copy(), xy[1].copy(), xy[2:].copy()
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate((self.agent, self.target, self.obstacles.reshape(-1)))

    @staticmethod
    def _entry_time(p, d, c, r):
        """first t in [0,1] where p + t d is inside circle (c, r), else None"""
        fx, fy = p[0] - c[0], p[1] - c[1]
        a = d[0] ** 2 + d[1] ** 2
        b = 2 * (fx * d[0] + fy * d[1])
        cc = fx ** 2 + fy ** 2 - r ** 2
        if cc <= 0:
            return 0.0
        if a == 0:
            return None
        disc = b * b - 4 * a * cc
        if disc < 0:
            return None
        t1 = (-b - math.sqrt(disc)) / (2 * a)
        return t1 if 0 <= t1 <= 1 else None

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        a = a / max(1.0, float(np.hypot(a[0], a[1])))  # Euclidean step-length limit (unit disc)
        p = self.agent.copy()
        d = self.max_step * a
        q = p + d
        lo, hi = -0.5, self.size - 0.5
        times = []
        for k in range(2):
            if d[k] > 0:
                times.append((hi - p[k]) / d[k])
            elif d[k] < 0:
                times.append((lo - p[k]) / d[k])
        t_col = min([t for t in times if t <= 1], default=math.inf)
        for c in self.obstacles:
            t = self._entry_time(p, d, c, self.obstacle_radius)
            if t is not None:
                t_col = min(t_col, t)
        t_goal = self._entry_time(p, d, self.target, self.goal_radius)
        reached = t_goal is not None and t_goal <= t_col
        collided = (not reached) and t_col <= 1
        reward = self.time_penalty
        if reached:
            reward += self.target_value
        if collided:
            reward += self.collision_value
        if self.shaping_coef:
            reward += self.shaping_coef * (self.gamma * -np.linalg.norm(q - self.target)
                                           + np.linalg.norm(p - self.target))
        self.agent = q
        self.t += 1
        terminated = reached or collided
        truncated = (not terminated) and self.t >= self.max_episode_steps
        return self._obs(), reward, terminated, truncated, {"reached": reached, "collided": collided}
