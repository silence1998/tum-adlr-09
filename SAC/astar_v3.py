"""
A* path-planning baseline for env v3 (privileged: uses the true obstacle state).

* Occupancy grid with cell size `res` (default 0.1) over [0,size]^2; a cell is blocked if its centre lies within
  r_k + margin (+ speed_inflate * |v_k| for moving obstacles) of an obstacle, or within `margin` of a wall.
* A*: 8-connected, step costs res / res*sqrt(2), octile-distance heuristic (admissible and consistent -> optimal
  on the grid), no corner cutting, heapq open list, g-costs stored in an array. (The old SAC/A_star/algorithm.py [moved 2026-10-03 to SAC/legacy/v1_original/A_star/]
  wrote child.g / child.f instead of child.gCost / child.fCost, so g stayed 0 and f was never updated: it was a
  greedy search, not A*; it also worked on integer cells only. Not reused.)
  The goal set = all free cells whose centre lies inside the goal disc shrunk by res.
* Path following through the real env (same action limits, collisions/timeouts counted by the env): the agent
  heads for the farthest path point visible in a straight line (segment/circle test against the same inflated
  obstacles) and moves min(max_step, distance) towards it.
* static mode: plan once at t=0 (re-plan only if the current path became invisible, which cannot happen for
  static obstacles except through discretisation). moving modes: re-plan every step from the current obstacle
  positions, inflated by speed_inflate*|v| (default 1.0 = one step of obstacle motion).
  Fallback if no path: retry without the speed inflation, then with margin 0; if still none: stand still
  (action 0) and count a planning failure.
  Moving modes additionally use a one-step safety filter (safety=True): the chosen action is checked against
  constant-velocity predictions of all obstacles (exact relative-motion test, margin/2); if it would collide, the
  predicted-safe candidate action (16 directions x 3 step lengths + wait) closest to the target is used instead.
  The prediction is exact for 'bounce' between bounces and approximate for 'random' (velocity noise).
"""
import heapq, math
import numpy as np
import torch

SQ2 = math.sqrt(2.0)
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def occupancy(size, res, pos, rad, margin):
    """pos (K,2), rad (K,) numpy (inflated radius = rad + margin, rad may already include speed inflation)."""
    n = int(round(size / res))
    c = (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(c, c, indexing="ij")
    occ = (X < margin) | (X > size - margin) | (Y < margin) | (Y > size - margin)
    for (px, py), r in zip(pos, rad):
        occ |= (X - px) ** 2 + (Y - py) ** 2 < (r + margin) ** 2
    return occ


def astar(occ, start, goal, goal_r, res):
    """occ (n,n) bool; start/goal (2,) in world coords. Returns list of world points or None."""
    n = occ.shape[0]
    si, sj = min(max(int(start[0] / res), 0), n - 1), min(max(int(start[1] / res), 0), n - 1)
    gi, gj = goal[0] / res - 0.5, goal[1] / res - 0.5
    gr2 = (max(goal_r - res, res) / res) ** 2
    occ = occ.copy(); occ[si, sj] = False                   # the agent stands here
    g = np.full(n * n, np.inf); parent = np.full(n * n, -1, dtype=np.int64)
    def h(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return (dx + dy) + (SQ2 - 2) * min(dx, dy)
    s = si * n + sj; g[s] = 0.0
    openl = [(h(si, sj), 0.0, s)]
    closed = np.zeros(n * n, dtype=bool)
    occf = occ.ravel()
    while openl:
        f, gc, u = heapq.heappop(openl)
        if closed[u]:
            continue
        closed[u] = True
        i, j = divmod(u, n)
        if (i - gi) ** 2 + (j - gj) ** 2 <= gr2:
            path = []
            while u != -1:
                path.append(((u // n + 0.5) * res, (u % n + 0.5) * res)); u = parent[u]
            path.reverse()
            path.append((float(goal[0]), float(goal[1])))
            path[0] = (float(start[0]), float(start[1]))
            return path
        for di, dj, cst in NB:
            a, b = i + di, j + dj
            if a < 0 or b < 0 or a >= n or b >= n:
                continue
            v = a * n + b
            if occf[v] or closed[v]:
                continue
            if di and dj and (occf[a * n + j] or occf[i * n + b]):
                continue
            ng = gc + cst
            if ng < g[v]:
                g[v] = ng; parent[v] = u
                heapq.heappush(openl, (ng + h(a, b), ng, v))
    return None


def visible(p, q, pos, rad):
    """segment p->q free of the discs (pos, rad)?"""
    d = q - p
    a = d @ d
    if a < 1e-12:
        return True
    f = p[None] - pos
    b = 2 * f @ d
    cc = (f * f).sum(1) - rad ** 2
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0))
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    hit = (disc >= 0) & (t2 >= 0) & (t1 <= 1)
    return not hit.any()


class AStarPolicy:
    def __init__(self, cfg, res=0.1, margin=0.15, speed_inflate=1.0, safety=True, replan_always=False):
        self.cfg, self.res, self.margin, self.si, self.safety = cfg, res, margin, speed_inflate, safety
        self.replan_always = replan_always   # DAgger relabelling: plan from the current state every step (also static)
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        dirs = np.stack([np.cos(th), np.sin(th)], 1)
        self.cands = np.concatenate([dirs, 0.5 * dirs, 0.25 * dirs, np.zeros((1, 2))])
        self.paths = None
        self.stats = {"plans": 0, "plan_fail_steps": 0, "initial_plan_fail": 0, "fallback_plans": 0}
        self.init_fail = None

    def _plan(self, a, goal, pos, rad, vel, moving):
        for k, (infl, marg) in enumerate(((self.si if moving else 0.0, self.margin), (0.0, self.margin), (0.0, 0.0))):
            if k > 0 and not moving and marg == self.margin:
                continue
            r_eff = rad + infl * np.linalg.norm(vel, axis=1)
            occ = occupancy(self.cfg.size, self.res, pos, r_eff, marg)
            self.stats["plans"] += 1
            path = astar(occ, a, goal, self.cfg.goal_radius, self.res)
            if path is not None:
                if k > 0:
                    self.stats["fallback_plans"] += 1
                return np.array(path), r_eff + marg
        return None, None

    def _hits(self, p, a, pos, rad, vel):
        """would action a collide within the next step if every obstacle keeps its velocity? (relative motion,
        obstacles inflated by margin/2; walls with margin/2)"""
        d = a * self.cfg.max_step
        q = p + d
        m = 0.5 * self.margin
        if (q < m).any() or (q > self.cfg.size - m).any():
            return True
        if len(pos) == 0:
            return False
        f = p[None] - pos; dd = d[None] - vel
        A_ = (dd * dd).sum(1); B = 2 * (f * dd).sum(1); C = (f * f).sum(1) - (rad + m) ** 2
        disc = B * B - 4 * A_ * C
        sq = np.sqrt(np.maximum(disc, 0))
        t1 = (-B - sq) / np.maximum(2 * A_, 1e-12); t2 = (-B + sq) / np.maximum(2 * A_, 1e-12)
        hit = (C <= 0) | ((disc >= 0) & (A_ > 1e-12) & (t2 >= 0) & (t1 <= 1))
        return bool(hit.any())

    def _safe(self, p, a, tgt, pos, rad, vel):
        """one-step safety filter for moving obstacles: keep the planned action if it is predicted collision-free,
        otherwise the predicted-safe candidate (16 directions x 3 lengths + wait) that gets closest to the target."""
        if not self._hits(p, a, pos, rad, vel):
            return a
        self.stats["safety_overrides"] = self.stats.get("safety_overrides", 0) + 1
        best, bd = a, np.inf
        for c in self.cands:
            if not self._hits(p, c, pos, rad, vel):
                dist = np.linalg.norm(p + c * self.cfg.max_step - tgt)
                if dist < bd:
                    best, bd = c, dist
        return best.astype(np.float32)

    def __call__(self, obs, env):
        st = env.state()
        n = env.n
        moving = self.cfg.motion != "static"
        first = self.paths is None
        if first:
            self.paths = [None] * n; self.rinf = [None] * n; self.init_fail = np.zeros(n, dtype=bool)
        act = np.zeros((n, 2), dtype=np.float32)
        A, G = st["agent"].numpy().astype(np.float64), st["goal"].numpy().astype(np.float64)
        P, R, V, M = st["obs_pos"].numpy(), st["obs_r"].numpy(), st["obs_vel"].numpy(), st["active"].numpy()
        alive = getattr(env, "_alive_np", np.ones(n, dtype=bool))
        for e in range(n):
            if not alive[e]:
                continue
            pos, rad, vel = P[e][M[e]].astype(np.float64), R[e][M[e]].astype(np.float64), V[e][M[e]].astype(np.float64)
            if first or moving or self.replan_always or self.paths[e] is None:
                path, rinf = self._plan(A[e], G[e], pos, rad, vel, moving)
                if path is None:
                    self.stats["plan_fail_steps"] += 1
                    if first:
                        self.init_fail[e] = True
                    self.paths[e] = None
                    continue
                self.paths[e], self.rinf[e] = path, rinf
            path, rinf = self.paths[e], self.rinf[e]
            # farthest visible path point (for static paths: only points ahead of the closest one)
            k0 = int(np.argmin(((path - A[e]) ** 2).sum(1)))
            tgt = path[min(k0 + 1, len(path) - 1)]
            for k in range(len(path) - 1, k0, -1):
                if visible(A[e], path[k], pos, rinf if not moving else rad + self.margin * 0.5):
                    tgt = path[k]; break
            d = tgt - A[e]; L = np.linalg.norm(d)
            if L > 1e-9:
                act[e] = d / L * min(1.0, L / self.cfg.max_step)
            if moving and self.safety:
                act[e] = self._safe(A[e], act[e], tgt, pos, rad, vel)
        if first:
            self.stats["initial_plan_fail"] = int(self.init_fail.sum())
        return torch.as_tensor(act, device=env.device)
