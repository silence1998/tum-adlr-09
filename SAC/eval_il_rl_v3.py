"""
Evaluate A*, IL (BC, DAgger) and RL (PPO, SAC, SAC-X) policies on the 512 fixed eval layouts (env seed 2026)
of each env v3 mode, deterministic policies; also the HARD level (zero-shot, models trained on the main level).
    python eval_il_rl_v3.py [--modes static random bounce] [--hard]
-> results/v3/il_rl/eval_<mode>[_hard].json, results/v3/il_rl/table_il_rl.md
Path ratio = path length / A* path length on the same layout, over episodes where both reached the goal.
"""
import argparse, glob, importlib.util, json, os, re
import numpy as np
import torch
from env_v3 import EnvV3Config, evaluate_v3
from astar_v3 import AStarPolicy
from training_PPO import PPOAgent

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../results/v3"))
OUT = os.path.join(R, "il_rl")
_spec = importlib.util.spec_from_file_location("sacx_mod", os.path.join(HERE, "training_SAC-X.py"))
sx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sx)


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "agent" in ck:  # PPO
        ag = PPOAgent(65, 2); ag.load_state_dict(ck["agent"]); ag.eval()
        return lambda o, e: ag.actor_mean(o).clamp(-1, 1)
    T = len(str(ck["args"]["tasks"]).split(","))
    act = sx.MultiHeadActor(65, 2, T, hidden=ck["args"].get("hidden", 256)); act.load_state_dict(ck["actor"]); act.eval()
    return lambda o, e: torch.tanh(act(o)[0][:, 0])


def models(mode, only=None):
    C = os.path.join(R, "checkpoints"); I = os.path.join(R, "il")
    m = {}
    for algo in ("ppo", "sac", "sacx"):
        for p in sorted(glob.glob(os.path.join(C, f"{algo}_{mode}_s[0-9].pt"))):
            m[os.path.basename(p)[:-3]] = p
    for p in sorted(glob.glob(os.path.join(I, f"bc_{mode}_*.pt"))) + sorted(glob.glob(os.path.join(I, f"dagger_{mode}*.pt"))):
        if only and not re.search(only, os.path.basename(p)):
            continue
        m[os.path.basename(p)[:-3]] = p
    return m


def summarize(ev, ref=None):
    oc = np.array(ev["outcome"]); pl = np.array(ev["path_len"]); ln = np.array(ev["length"])
    s = {k: float((oc == k).mean()) for k in ("goal", "collision", "timeout")}
    s["steps_goal"] = float(ln[oc == "goal"].mean()) if (oc == "goal").any() else None
    if ref is not None:
        both = (oc == "goal") & (np.array(ref["outcome"]) == "goal")
        s["path_ratio_vs_astar"] = float((pl[both] / np.maximum(np.array(ref["path_len"])[both], 1e-6)).mean()) if both.any() else None
        s["n_both_goal"] = int(both.sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["static", "random", "bounce"])
    ap.add_argument("--hard", action="store_true", help="also evaluate on configs/v3_<mode>_hard.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hard-only", action="store_true")
    ap.add_argument("--only-il", default=r"(50k_s300k|50k_h1024|_big)", help="regex of IL models also evaluated on the hard level")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    for mode in a.modes:
        for lvl in (([] if a.hard_only else [""]) + (["_hard"] if a.hard or a.hard_only else [])):
            cfg = EnvV3Config.from_yaml(os.path.join(HERE, f"../configs/v3_{mode}{lvl}.yaml"))
            ref = evaluate_v3(AStarPolicy(cfg), cfg, 512, "cpu", 2026, per_episode=True)
            res = {"astar": summarize(ref, ref)}
            for name, p in models(mode, a.only_il if lvl else None).items():
                if lvl and a.only_il and not re.search(a.only_il, name) and not name.startswith(("ppo", "sac")):
                    continue
                with torch.no_grad():
                    ev = evaluate_v3(load_policy(p), cfg, 512, a.device, 2026, per_episode=True)
                res[name] = summarize(ev, ref); res[name]["ckpt"] = os.path.relpath(p, R)
                print(mode + lvl, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
            json.dump(res, open(os.path.join(OUT, f"eval_{mode}{lvl}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
