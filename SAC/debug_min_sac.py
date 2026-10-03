"""
Minimal standard SAC (transitions, twin critics, auto-alpha) on the GPU env, used to diagnose the Q overestimation.
Prints, at every eval: deterministic eval, and an overestimation check = mean Q(s0, a0) vs. the Monte-Carlo soft
return of the same stochastic policy from the same start states (training reward incl. shaping, rscale, entropy).
"""
import argparse, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch_env import make_env, obs_features, evaluate_policy_v

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--utd", type=int, default=8)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--version", type=int, default=2)
ap.add_argument("--shaping", type=float, default=None)
ap.add_argument("--time-penalty", type=float, default=None)
ap.add_argument("--rscale", type=float, default=0.1)
ap.add_argument("--alpha0", type=float, default=1.0)
ap.add_argument("--target-ent", type=float, default=-2.0)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--layernorm", type=int, default=0)
ap.add_argument("--eval-every", type=int, default=10000)
ap.add_argument("--mem-frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--obs-radius", type=float, default=None)
ap.add_argument("--sort", type=int, default=0)
args = ap.parse_args()
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
torch.manual_seed(args.seed)
G = args.gamma
mk = lambda n, seed: make_env(args.version, n, device=dev, seed=seed, shaping_coef=args.shaping,
                              time_penalty=args.time_penalty, shaping_gamma=G,
                              obstacle_radius=args.obs_radius)
env = mk(args.num_envs, args.seed)
O, A, N = 26, 2, args.num_envs
f = lambda o: obs_features(o, 10, sort_obstacles=bool(args.sort))
def net(i, o, ln=False):
    L = [nn.Linear(i, 256)] + ([nn.LayerNorm(256)] if ln else []) + [nn.ReLU(), nn.Linear(256, 256)] + \
        ([nn.LayerNorm(256)] if ln else []) + [nn.ReLU(), nn.Linear(256, o)]
    return nn.Sequential(*L).to(dev)
ln = bool(args.layernorm)
actor = net(O, 2 * A); q1, q2 = net(O + A, 1, ln), net(O + A, 1, ln); t1, t2 = net(O + A, 1, ln), net(O + A, 1, ln)
t1.load_state_dict(q1.state_dict()); t2.load_state_dict(q2.state_dict())
log_alpha = torch.full((1,), math.log(args.alpha0), device=dev, requires_grad=True)
oa = torch.optim.Adam(actor.parameters(), args.lr); oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), args.lr)
oal = torch.optim.Adam([log_alpha], args.lr)
def pi(o):
    mu, ls = actor(o).chunk(2, -1); ls = ls.clamp(-5, 2); std = ls.exp()
    u = mu + std * torch.randn_like(mu); a = torch.tanh(u)
    lp = (-0.5 * ((u - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1) - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
    return a, lp, torch.tanh(mu)
qmin = lambda o, a: torch.min(q1(torch.cat((o, a), -1)), q2(torch.cat((o, a), -1))).squeeze(-1)

@torch.no_grad()
def mc_check(n=1024):
    e = mk(n, 777); o = e.reset(); al = log_alpha.exp()
    a, lp, _ = pi(f(o)); q0 = qmin(f(o), a)
    ret = torch.zeros(n, device=dev); disc = torch.ones(n, device=dev); alive = torch.ones(n, dtype=torch.bool, device=dev)
    for t in range(100):
        o2, r, term, trunc, info = e.step(a, auto_reset=False)
        ret += disc * alive * args.rscale * r
        alive &= ~term
        disc *= G
        if t < 99:
            a, lp, _ = pi(f(o2)); ret -= disc * alive * al * lp  # soft bonus of the next action
            o = o2
    a, lp, _ = pi(f(o2)); ret += disc * alive * (qmin(f(o2), a))  # bootstrap after 100 steps (time limit)
    return q0.mean().item(), ret.mean().item()

C = 1_000_000
B_o = torch.zeros(C, 14, device=dev); B_a = torch.zeros(C, A, device=dev); B_r = torch.zeros(C, device=dev)
B_n = torch.zeros(C, 14, device=dev); B_d = torch.zeros(C, device=dev); ptr = 0
obs = env.reset(); t0 = time.time(); steps = 0; nxt = 0
while steps <= args.steps:
    if steps >= nxt:
        ev = evaluate_policy_v(lambda o: pi(f(o))[2], args.version, num_episodes=512, device=dev)
        evs = evaluate_policy_v(lambda o: pi(f(o))[0], args.version, num_episodes=512, device=dev)
        qm, mc = mc_check()
        print(f"{steps:6d} det succ {ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} len {ev['eval_length']:.1f} | "
              f"stoch succ {evs['eval_success']:.3f} | Q(s0,a0) {qm:7.3f} vs MC {mc:7.3f} | alpha {log_alpha.exp().item():.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        nxt += args.eval_every
    with torch.no_grad():
        a = pi(f(obs))[0] if steps > 5000 else torch.rand(N, A, device=dev) * 2 - 1
    nobs, r, term, trunc, info = env.step(a)
    idx = torch.arange(ptr, ptr + N, device=dev) % C
    B_o[idx], B_a[idx], B_r[idx], B_n[idx], B_d[idx] = obs, a, args.rscale * r, info["final_obs"], term.float()
    ptr += N; obs = nobs; steps += N
    if steps < 5000: continue
    for _ in range(args.utd):
        j = torch.randint(0, min(ptr, C), (args.batch,), device=dev)
        o, aa, rr, no, d = f(B_o[j]), B_a[j], B_r[j], f(B_n[j]), B_d[j]
        al = log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp, _ = pi(no); x = torch.cat((no, na), -1)
            y = rr + G * (1 - d) * (torch.min(t1(x), t2(x)).squeeze(-1) - al * nlp)
        x = torch.cat((o, aa), -1)
        lq = F.mse_loss(q1(x).squeeze(-1), y) + F.mse_loss(q2(x).squeeze(-1), y)
        oq.zero_grad(); lq.backward(); oq.step()
        na, lp, _ = pi(o)
        la = (al * lp - qmin(o, na)).mean()
        oa.zero_grad(); la.backward(); oa.step()
        lal = -(log_alpha * (lp.detach() + args.target_ent)).mean(); oal.zero_grad(); lal.backward(); oal.step()
        with torch.no_grad():
            for t, q in ((t1, q1), (t2, q2)):
                for tp, p in zip(t.parameters(), q.parameters()): tp.mul_(0.995).add_(0.005 * p)
@torch.no_grad()
def probe():
    """hand-made states: agent 0.8 cells from the goal / 0.8 cells from an obstacle, action towards it."""
    e = mk(2048, 99); o = e.reset().clone()
    g = o[:, 2:4]; ob = o[:, 4:6]
    # (a) one step before the goal, path clear of the other obstacles in most layouts
    dirv = torch.nn.functional.normalize(torch.randn(2048, 2, device=dev), dim=-1)
    oa_ = o.clone(); oa_[:, 0:2] = g - 0.8 * dirv
    qa = qmin(f(oa_), dirv)
    # (b) right in front of obstacle 0, moving into it
    obo = o.clone(); obo[:, 0:2] = ob - 0.8 * dirv
    qb = qmin(f(obo), dirv)
    qb_away = qmin(f(obo), -dirv)
    # TD residual on replay
    j = torch.randint(0, min(ptr, C), (4096,), device=dev)
    oo, aa, rr, no, d = f(B_o[j]), B_a[j], B_r[j], f(B_n[j]), B_d[j]
    na, nlp, _ = pi(no); y = rr + G * (1 - d) * (qmin(no, na) - log_alpha.exp() * nlp)
    q = qmin(oo, aa)
    print(f"probe: Q(1 step before goal, towards goal) {qa.mean():.2f} (ideal ~{10*args.rscale:.1f}) | "
          f"Q(into obstacle) {qb.mean():.2f} (ideal ~{-10*args.rscale:.1f}) | Q(away from obstacle) {qb_away.mean():.2f}")
    print(f"replay: mean Q {q.mean():.2f}, mean target {y.mean():.2f}, mean |TD| {(q-y).abs().mean():.3f}; "
          f"terminal frac {d.mean():.3f}, mean reward {rr.mean():.3f}, Q on terminal {q[d>0].mean():.2f}, "
          f"reward on terminal {rr[d>0].mean():.2f}")
probe()
print("done", f"{time.time()-t0:.0f}s")
