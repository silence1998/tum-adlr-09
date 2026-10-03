# extra oscillation diagnostics on timeout episodes (where/how the main head oscillates)
import os, sys, math, json, numpy as np, torch
# moved to SAC/legacy/ on 2026-10-03: make SAC/ importable and the working dir, so the relative paths below
# (checkpoints/..., ../results/...) resolve exactly as before
_SAC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(1, _SAC); os.chdir(_SAC)
sys.argv += [] ; import sacx_aux_eval as E
out = {}
for name, path in [("sacx_full_s1", "checkpoints/sacx_q_s1_1M.pt"), ("sac_s1", "checkpoints/sac_s1_1M.pt")]:
    actor, critic, f, tasks, ck = E.load(path)
    ro = E.rollout(actor, f)
    oc = ro["outcome"]; idx = np.nonzero(oc == "timeout")[0]
    obs, pos, disp = ro["obs"], ro["pos"], ro["disp"]
    T = obs.shape[0]
    rec = {"n_timeout": len(idx), "eps": []}
    dirs = torch.stack([torch.tensor([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2*math.pi, 72, endpoint=False)])
    allc, alld, amag, axis, bim, stepn, gaps = [], [], [], [], 0, 0, []
    for i in idx:
        c = E.clearance(pos[:, i], ro["obst"][i:i+1].expand(pos.shape[0], -1, -1), ro["r"])
        dg = (pos[:, i] - ro["target"][i]).norm(dim=-1)
        # second half of the episode = the stuck phase
        sl = slice(50, 100)
        x = f(obs[sl, i]); mu, _ = actor(x); am = torch.tanh(mu[:, 0])
        amag += am.norm(dim=-1).tolist(); axis += (am.abs().max(-1).values / am.norm(dim=-1).clamp(min=1e-6)).tolist()
        allc += c[sl].tolist(); alld += dg[sl].tolist()
        for t in range(50, 100, 5):
            qa = critic.min_q(f(obs[t, i:i+1]).expand(72, -1), dirs)[:, 0]
            lm = sorted([(qa[k].item(), k) for k in range(72) if qa[k] >= qa[k-1] and qa[k] >= qa[(k+1) % 72]], reverse=True)
            stepn += 1
            if len(lm) >= 2:
                sep = abs(lm[0][1]-lm[1][1]); sep = min(sep, 72-sep)*5
                if sep >= 45: bim += 1; gaps.append(lm[0][0]-lm[1][0])
        d = disp[50:100, i]
        rec["eps"].append({"layout": int(i), "clear_mean_stuck": round(float(c[sl].mean()), 2),
                           "dist_goal_stuck": round(float(dg[sl].mean()), 2),
                           "net_disp_last50": round(float((pos[99, i]-pos[50, i]).norm()), 3),
                           "mean_step_len_last50": round(float(d.norm(dim=-1).mean()), 3)})
    rec["stuck_phase_mean_clearance"] = float(np.mean(allc)); rec["frac_stuck_steps_clearance<0.6"] = float(np.mean(np.array(allc) < 0.6))
    rec["frac_stuck_steps_clearance<1.0"] = float(np.mean(np.array(allc) < 1.0))
    rec["stuck_phase_mean_dist_goal"] = float(np.mean(alld))
    rec["main_action_norm_mean"] = float(np.mean(amag)); rec["axis_align_mean"] = float(np.mean(axis))
    rec["frac_bimodal_mainQ_stuck"] = bim / max(stepn, 1); rec["median_gap"] = float(np.median(gaps)) if gaps else None
    out[name] = rec
json.dump(out, open("../results/sacx_aux/eval/diag_stuck_phase.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: b for a, b in v.items() if a != "eps"})
    for e in v["eps"]: print("  ", e)
