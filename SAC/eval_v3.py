"""Evaluate A* and/or PPO checkpoints on the fixed v3 eval layouts (512 episodes, env seed 2026) per config.
    python eval_v3.py --config ../configs/v3_static.yaml --astar --ppo ../results/v3/checkpoints/ppo_static_s0.pt \
        --out ../results/v3/eval/static.json"""
import argparse, json, os, time
import torch
from env_v3 import EnvV3Config, evaluate_v3
from astar_v3 import AStarPolicy
from training_PPO import PPOAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--astar", action="store_true")
    ap.add_argument("--ppo", nargs="*", default=[])
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--speed-inflate", type=float, default=1.0)
    ap.add_argument("--no-safety", action="store_true", help="A*: disable the one-step safety filter (moving modes)")
    ap.add_argument("--tag", default="astar", help="key in the output json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = EnvV3Config.from_yaml(a.config)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res["config"] = a.config
    if a.astar:
        pol = AStarPolicy(cfg, a.res, a.margin, a.speed_inflate, safety=not a.no_safety)
        t0 = time.time()
        ev = evaluate_v3(pol, cfg, a.episodes, a.device, a.seed)
        ev.update(pol.stats); ev["initial_plan_fail_rate"] = pol.stats["initial_plan_fail"] / a.episodes
        ev["time_s"] = time.time() - t0
        ev["params"] = dict(res=a.res, margin=a.margin, speed_inflate=a.speed_inflate, safety=not a.no_safety)
        res[a.tag] = ev
        print("A*", json.dumps(ev))
    for p in a.ppo:
        ck = torch.load(p, map_location=a.device, weights_only=False)
        ag = PPOAgent(9 + 7 * cfg.n_obs_max, 2); ag.load_state_dict(ck["agent"]); ag.eval()
        with torch.no_grad():
            ev = evaluate_v3(lambda o, e: ag.actor_mean(o).clamp(-1, 1), cfg, a.episodes, a.device, a.seed)
            es = evaluate_v3(lambda o, e: ag.get_action_and_value(o)[0].clamp(-1, 1), cfg, a.episodes, a.device, a.seed)
        ev["stochastic"] = {k: es[k] for k in ("eval_success", "eval_collision", "eval_timeout")}
        res[os.path.basename(p)[:-3]] = ev
        print(os.path.basename(p), json.dumps(ev))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
