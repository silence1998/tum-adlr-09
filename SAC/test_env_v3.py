"""Unit tests for env_v3 (run: python test_env_v3.py  or  pytest test_env_v3.py). CPU by default."""
import dataclasses, math, time, sys
import torch
from env_v3 import EnvV3, EnvV3Config, segment_circle_entry

DEV = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "cuda") else "cpu"
T = lambda x: torch.tensor(x, dtype=torch.float32)


def _gaps(env):
    p, r, a = env.obs_pos, env.obs_r, env.active
    dd = (p[:, :, None] - p[:, None]).norm(dim=-1) - r[:, :, None] - r[:, None]
    pair = a[:, :, None] & a[:, None] & ~torch.eye(env.K, dtype=torch.bool, device=env.device)
    return torch.where(pair, dd, torch.full_like(dd, float("inf"))).flatten(1).min(1).values


def test_spacing():
    for motion in ("static", "random", "bounce"):
        c = EnvV3Config(motion=motion)
        env = EnvV3(4096, c, device=DEV, seed=1); env.reset()
        rw = c.rw_radius if motion == "random" else 0.0
        a = env.active
        assert (_gaps(env) >= c.min_gap + 2 * rw - 1e-4).all()
        S = c.size; p, r = env.obs_pos, env.obs_r
        wall = torch.stack([p[..., 0] - r, S - p[..., 0] - r, p[..., 1] - r, S - p[..., 1] - r], -1).min(-1).values
        assert (wall[a] >= c.wall_gap + rw - 1e-4).all()
        ds = (p - env.agent[:, None]).norm(dim=-1) - r
        dg = (p - env.goal[:, None]).norm(dim=-1) - r
        assert (ds[a] >= c.start_clear + rw - 1e-4).all() and (dg[a] >= c.goal_clear + rw - 1e-4).all()
        sg = (env.agent - env.goal).norm(dim=1)
        assert (sg >= c.start_goal_min - 1e-4).all() or env.stats["goal_fallback"] > 0
        n = a.sum(1)
        assert (n <= c.n_obs_max).all()
        print(f"  spacing ok [{motion}]: obstacles/episode mean {n.float().mean():.2f} (min {n.min()}, max {n.max()}), "
              f"dropped {env.stats['dropped_obstacles']}, goal fallback {env.stats['goal_fallback']}")


def test_no_hang():
    c = EnvV3Config(n_obs_min=30, n_obs_max=30, r_min=1.0, r_max=1.5, start_goal_min=13.0, motion="random")
    env = EnvV3(2048, c, device=DEV, seed=0)
    t0 = time.time(); env.reset(); dt = time.time() - t0
    assert dt < 30, dt
    assert env.stats["dropped_obstacles"] > 0 and env.stats["goal_fallback"] > 0   # impossible config -> fallbacks
    assert (_gaps(env) >= c.min_gap + 2 * c.rw_radius - 1e-4).all()           # guarantees still hold for placed ones
    print(f"  no hang ok: impossible config reset in {dt:.2f}s, dropped {env.stats['dropped_obstacles']}, "
          f"goal fallback {env.stats['goal_fallback']}")


def _env1(motion="static", **kw):
    c = EnvV3Config(motion=motion, n_obs_min=1, n_obs_max=1, shaping=0.0, **kw)
    return EnvV3(1, c, device=DEV, seed=0)


def test_collision_static():
    env = _env1()
    # obstacle on the segment -> collision
    env.set_state(T([[2., 5.]]), T([[9., 9.]]), T([[[2.5, 5.0]]]), T([[0.3]]))
    _, r, term, _, info = env.step(T([[1., 0.]]), auto_reset=False)
    assert info["collided"].item() and term.item() and abs(r.item() - (-10.05)) < 1e-4
    # tangential miss (segment passes 0.31 from the centre)
    env.set_state(T([[2., 5.]]), T([[9., 9.]]), T([[[2.5, 5.31]]]), T([[0.3]]))
    _, _, term, _, info = env.step(T([[1., 0.]]), auto_reset=False)
    assert not info["collided"].item() and not term.item()
    # end point beyond the obstacle (would be missed by an end-point-only check)
    env.set_state(T([[2., 5.]]), T([[9., 9.]]), T([[[2.5, 5.0]]]), T([[0.1]]))
    _, _, _, _, info = env.step(T([[1., 0.]]), auto_reset=False)
    assert info["collided"].item()
    # goal reached before an obstacle further along the segment
    env.set_state(T([[2., 5.]]), T([[2.6, 5.]]), T([[[2.95, 5.0]]]), T([[0.1]]))
    _, r, _, _, info = env.step(T([[1., 0.]]), auto_reset=False)
    assert info["reached"].item() and not info["collided"].item()
    # wall
    env.set_state(T([[0.5, 5.]]), T([[9., 9.]]), T([[[5., 5.0]]]), T([[0.1]]))
    _, _, _, _, info = env.step(T([[-1., 0.]]), auto_reset=False)
    assert info["collided"].item()
    print("  static collision ok")


def test_collision_moving():
    # agent standing still, obstacle sweeping over it during the step -> collision (end points both clear)
    env = _env1("bounce", obstacle_interaction="pass", speed_max=1.0)
    env.set_state(T([[5., 5.]]), T([[9., 9.]]), T([[[4.4, 5.0]]]), T([[0.2]]), obs_vel=T([[[1.2, 0.]]]))
    _, _, _, _, info = env.step(T([[0., 0.]]), auto_reset=False)
    assert info["collided"].item(), "swept obstacle must hit a static agent"
    # agent and obstacle cross paths at different times -> no collision although the static paths intersect
    env.set_state(T([[4., 5.]]), T([[9., 9.]]), T([[[4.2, 6.0]]]), T([[0.2]]), obs_vel=T([[[0., -1.0]]]))
    _, _, _, _, info = env.step(T([[1., 0.]]), auto_reset=False)   # paths cross at (4.2,5): agent there at t=0.2, obstacle at t=1
    assert not info["collided"].item()
    # both meet at the same place at the same time -> collision
    env.set_state(T([[4., 5.]]), T([[9., 9.]]), T([[[4.5, 5.5]]]), T([[0.2]]), obs_vel=T([[[0., -1.0]]]))
    _, _, _, _, info = env.step(T([[1., 0.]]), auto_reset=False)  # at t=0.5 both at (4.5, 5.0)
    assert info["collided"].item()
    print("  moving collision ok (relative-motion check)")


def test_bounce():
    env = _env1("bounce", obstacle_interaction="elastic", speed_max=0.5)
    env.set_state(T([[1., 1.]]), T([[9., 1.]]), T([[[9.6, 5.0]]]), T([[0.3]]), obs_vel=T([[[0.3, 0.1]]]))
    env.step(T([[0., 0.]]), auto_reset=False)
    assert env.obs_vel[0, 0, 0] < 0 and env.obs_pos[0, 0, 0] <= 10 - 0.3 + 1e-5, "wall reflection"
    # head-on, equal radii -> velocities swap, momentum conserved
    c = EnvV3Config(motion="bounce", n_obs_min=2, n_obs_max=2, speed_max=0.5, shaping=0.0)
    env = EnvV3(1, c, device=DEV, seed=0)
    env.set_state(T([[1., 1.]]), T([[9., 1.]]), T([[[4.6, 5.], [5.4, 5.]]]), T([[0.3, 0.3]]),
                  obs_vel=T([[[0.2, 0.], [-0.2, 0.]]]))
    env.step(T([[0., 0.]]), auto_reset=False)
    v = env.obs_vel[0]
    assert torch.allclose(v, T([[-0.2, 0.], [0.2, 0.]]).to(v.device), atol=1e-5), v
    assert (env.obs_pos[0, 1] - env.obs_pos[0, 0]).norm() >= 0.6 - 1e-5, "overlap resolved"
    # unequal masses: momentum (m ~ r^2) conserved
    env.set_state(T([[1., 1.]]), T([[9., 1.]]), T([[[4.5, 5.], [5.4, 5.]]]), T([[0.4, 0.2]]),
                  obs_vel=T([[[0.2, 0.], [-0.2, 0.]]]))
    m = T([0.16, 0.04]).to(env.device)
    p0 = (m[:, None] * env.obs_vel[0]).sum(0); env.step(T([[0., 0.]]), auto_reset=False)
    p1 = (m[:, None] * env.obs_vel[0]).sum(0)
    assert torch.allclose(p0, p1, atol=1e-6)
    # goal protection: obstacles never enter the goal disc (long rollout)
    c = EnvV3Config(motion="bounce", speed_max=0.3)
    env = EnvV3(256, c, device=DEV, seed=3); env.reset()
    for _ in range(300):
        env.step(torch.zeros(256, 2), auto_reset=False)
        dg = (env.obs_pos - env.goal[:, None]).norm(dim=-1) - env.obs_r
        assert (dg[env.active] >= c.goal_radius - 1e-4).all()
        assert (env.obs_pos >= env.obs_r[..., None] - 1e-4).all() and (env.obs_pos <= c.size - env.obs_r[..., None] + 1e-4).all()
    print("  bounce ok (wall reflection, elastic swap, momentum, goal protected, inside arena)")


def test_random_walk_limits():
    c = EnvV3Config(motion="random")
    env = EnvV3(512, c, device=DEV, seed=5); env.reset()
    for _ in range(300):
        env.step(torch.zeros(512, 2), auto_reset=False)
        a = env.active
        assert ((env.obs_pos - env.spawn).norm(dim=-1)[a] <= c.rw_radius + 1e-4).all()
        assert (env.obs_vel.norm(dim=-1) <= c.speed_max + 1e-5).all()
        assert (_gaps(env) >= c.min_gap - 1e-4).all()
        dg = (env.obs_pos - env.goal[:, None]).norm(dim=-1) - env.obs_r
        assert (dg[a] >= c.goal_clear - 1e-4).all()
    print("  random-walk limits ok (radius, speed, gaps, goal)")


def test_determinism():
    for motion in ("static", "random", "bounce"):
        c = EnvV3Config(motion=motion)
        outs = []
        for seed in (7, 7, 8):
            env = EnvV3(64, c, device=DEV, seed=seed); o = env.reset(); g = torch.Generator().manual_seed(0)
            acc = [o]
            for _ in range(150):
                o, *_ = env.step(torch.rand(64, 2, generator=g) * 2 - 1); acc.append(o)
            outs.append(torch.stack(acc))
        assert torch.equal(outs[0], outs[1]) and not torch.equal(outs[0], outs[2])
    print("  determinism ok")


def test_obs():
    c = EnvV3Config(motion="bounce", n_obs_min=2, n_obs_max=8)
    env = EnvV3(256, c, device=DEV, seed=2); o = env.reset()
    assert o.shape == (256, env.obs_dim) and env.obs_dim == 9 + 7 * 8
    slots = o[:, 9:].reshape(256, 8, 7)
    m = slots[..., 6]
    assert torch.equal(m.sum(1), env.active.sum(1).float())
    assert (m[:, :-1] >= m[:, 1:]).all(), "padding last"
    sd = torch.where(m > 0, slots[..., 5], torch.full_like(m, 9.))
    assert (sd[:, :-1] <= sd[:, 1:] + 1e-6).all(), "sorted by surface distance"
    assert (slots[m == 0] == 0).all()
    print("  observation ok (dim", env.obs_dim, ", mask/padding/sorting)")


def test_segment_circle():
    t = segment_circle_entry(T([0., 0.]), T([2., 0.]), T([1., 0.]), T(0.5))
    assert abs(t.item() - 0.25) < 1e-6
    print("  segment/circle ok")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            print(k); f()
    print("ALL TESTS PASSED")
