"""
Equivalence tests: GPU env (torch_env.TorchGridWorld) vs. CPU envs
  (a) fixed environment.GridWorldEnv           (incl. clipping of out-of-range actions and time-limit truncation)
  (b) original baseline_original/environment_original.GridWorldEnv (in-range actions, no time limit)
  (c) reset distribution sanity checks.
Run from the SAC folder:  python test_torch_env.py
"""
import os
import sys
import types
import numpy as np
import torch

# gym / pygame stand-ins for the original env (not installed in the conda env)
try:
    import gym  # noqa
except ImportError:
    gym = types.ModuleType("gym")

    class _Env:
        def reset(self, seed=None, options=None):
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)
    gym.Env = _Env
    sp = types.ModuleType("gym.spaces"); sp.Box = sp.Dict = sp.Discrete = (lambda *a, **k: None)
    gym.spaces = sp
    sys.modules["gym"], sys.modules["gym.spaces"] = gym, sp
    sys.modules["pygame"] = types.ModuleType("pygame")

from environment import GridWorldEnv
from torch_env import TorchGridWorld

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_original"))
from environment_original import GridWorldEnv as OriginalGridWorldEnv  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE, K = 10, 5
SPECIAL = np.array([0.25, -0.25, 0.75, -0.75, 0.0, 1.0, -1.0, 0.5, -0.5, 0.2499, 0.7501])


def random_actions(rng, n, out_of_range):
    a = rng.uniform(-1.5, 1.5, size=(n, 2)) if out_of_range else rng.uniform(-1, 1, size=(n, 2))
    mask = rng.random((n, 2)) < 0.3  # many rounding-boundary values (round-half-to-even must match)
    a[mask] = rng.choice(SPECIAL, size=mask.sum())
    return a


def compare(env_cls, n_envs, max_steps, out_of_range, seed, **kw):
    rng = np.random.default_rng(seed)
    cpu = [env_cls(render_mode=None, size=SIZE, num_obstacles=K, **kw) for _ in range(n_envs)]
    for i, e in enumerate(cpu):
        e.reset(seed=seed * 100000 + i)
    genv = TorchGridWorld(n_envs, SIZE, K, max_episode_steps=max_steps, device=DEV)
    genv.set_state(torch.tensor(np.array([e._agent_location for e in cpu]), dtype=torch.float),
                   torch.tensor(np.array([e._target_location for e in cpu]), dtype=torch.float),
                   torch.tensor(np.array([[e._obstacle_locations[str(j)] for j in range(K)] for e in cpu]),
                                dtype=torch.float))
    alive = np.ones(n_envs, bool)
    n_cmp = mismatches = 0
    counts = {"reached": 0, "collided": 0, "truncated": 0}
    for step in range(max_steps if max_steps < 10 ** 6 else 200):
        acts = random_actions(rng, n_envs, out_of_range)
        _, g_r, g_term, g_trunc, info = genv.step(torch.tensor(acts, dtype=torch.float32), auto_reset=False)
        g_r, g_term, g_trunc = g_r.cpu().numpy(), g_term.cpu().numpy(), g_trunc.cpu().numpy()
        g_agent = info["agent"].cpu().numpy()
        for i in np.nonzero(alive)[0]:
            _, r, term, trunc, _ = cpu[i].step(acts[i])
            n_cmp += 1
            ok = (r == g_r[i]) and (bool(term) == bool(g_term[i])) and (bool(trunc) == bool(g_trunc[i])) and \
                np.array_equal(cpu[i]._agent_location.astype(np.float32), g_agent[i])
            if not ok:
                mismatches += 1
                if mismatches <= 5:
                    print("MISMATCH env", i, "step", step, "act", acts[i], "cpu", r, term, trunc,
                          cpu[i]._agent_location, "gpu", g_r[i], g_term[i], g_trunc[i], g_agent[i])
            if term or trunc:
                alive[i] = False
                counts["reached" if r > 0 else ("collided" if term else "truncated")] += 1
        if not alive.any():
            break
    return n_cmp, mismatches, counts


def reset_checks(n=20000):
    genv = TorchGridWorld(n, SIZE, K, device=DEV, seed=1)
    o = genv.reset().cpu().numpy().reshape(n, K + 2, 2)
    distinct = all(len({tuple(p) for p in row}) == K + 2 for row in o[:2000])
    in_bounds = (o >= 0).all() and (o <= SIZE - 1).all()
    g_cheb = np.abs(o[:, 0] - o[:, 1]).max(1).mean()
    cpu = GridWorldEnv(size=SIZE, num_obstacles=K)
    d = []
    for s in range(5000):
        cpu.reset(seed=s)
        d.append(np.abs(cpu._agent_location - cpu._target_location).max())
    return distinct, in_bounds, g_cheb, float(np.mean(d)), o[:, 0].mean(0)


if __name__ == "__main__":
    all_ok = True
    for name, cls, oor, ms, kw in [("fixed env (clip + time limit)", GridWorldEnv, True, 100, {"max_episode_steps": 100}),
                                   ("fixed env, short time limit", GridWorldEnv, False, 7, {"max_episode_steps": 7}),
                                   ("ORIGINAL env (in-range actions)", OriginalGridWorldEnv, False, 10 ** 9, {})]:
        tot = mis = 0
        cnt = {"reached": 0, "collided": 0, "truncated": 0}
        for seed in range(1, 6):
            n, m, c = compare(cls, 400, ms, oor, seed, **kw)
            tot += n; mis += m
            for k in cnt: cnt[k] += c[k]
        print(f"{name:35s}: {tot} transitions compared, {mis} mismatches, episode ends {cnt}")
        all_ok &= mis == 0
    distinct, inb, gch, cch, mean_agent = reset_checks()
    print(f"reset: all cells distinct={distinct}, in bounds={inb}, mean Chebyshev agent-target dist "
          f"GPU={gch:.3f} CPU={cch:.3f}, mean agent pos={mean_agent}")
    all_ok &= distinct and inb and abs(gch - cch) < 0.15
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")


# ---------------------------------------------------------------------------------------------- #
#  Env version 2 (continuous): torch reference vs independent NumPy implementation
# ---------------------------------------------------------------------------------------------- #
def compare_v2(n_envs=500, seeds=range(1, 6), dtype=torch.float64, steps=100):
    from environment_v2 import ContinuousGridWorldEnv
    from torch_env import TorchContinuousWorld
    tot = mis = 0
    cnt = {"reached": 0, "collided": 0, "truncated": 0}
    max_rerr = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cpu = [ContinuousGridWorldEnv(size=SIZE, num_obstacles=K) for _ in range(n_envs)]
        for i, e in enumerate(cpu):
            e.reset(seed=seed * 100000 + i)
        g = TorchContinuousWorld(n_envs, SIZE, K, 100, device=DEV, dtype=dtype)
        g.set_state(torch.tensor(np.array([e.agent for e in cpu])), torch.tensor(np.array([e.target for e in cpu])),
                    torch.tensor(np.array([e.obstacles for e in cpu])))
        alive = np.ones(n_envs, bool)
        for step in range(steps):
            # biased towards the goal half of the time so that goal events are frequent as well
            to_goal = np.array([e.target - e.agent for e in cpu])
            acts = rng.uniform(-1.3, 1.3, size=(n_envs, 2))
            mask = rng.random(n_envs) < 0.5
            acts[mask] = np.clip(to_goal[mask] + rng.normal(0, 0.3, size=(mask.sum(), 2)), -1, 1)
            _, gr, gt, gtr, info = g.step(torch.tensor(acts), auto_reset=False)
            gr, gt, gtr = gr.cpu().numpy(), gt.cpu().numpy(), gtr.cpu().numpy()
            ga = info["agent"].cpu().numpy()
            for i in np.nonzero(alive)[0]:
                _, r, term, trunc, inf = cpu[i].step(acts[i])
                tot += 1
                max_rerr = max(max_rerr, abs(r - gr[i]))
                ok = abs(r - gr[i]) < 1e-4 and term == bool(gt[i]) and trunc == bool(gtr[i]) and \
                    np.allclose(cpu[i].agent, ga[i], atol=1e-6)
                if not ok:
                    mis += 1
                    if mis <= 5:
                        print("V2 MISMATCH", i, step, acts[i], r, term, trunc, gr[i], gt[i], gtr[i])
                if term or trunc:
                    alive[i] = False
                    cnt["reached" if inf["reached"] else ("collided" if term else "truncated")] += 1
            if not alive.any():
                break
    return tot, mis, cnt, max_rerr


if __name__ == "__main__":
    for dt in (torch.float64, torch.float32):
        tot, mis, cnt, mre = compare_v2(dtype=dt)
        print(f"v2 continuous env ({str(dt)}): {tot} transitions compared, {mis} mismatches, "
              f"max |reward diff| {mre:.2e}, episode ends {cnt}")
