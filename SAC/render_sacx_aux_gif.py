"""
Side-by-side GIFs for the SAC-X aux-task study: old SAC-X (1M) | best aux variant | plain SAC, deterministic
main head, same 512 layouts (env seed 2026) as render_gifs.py, on the layouts where old SAC-X times out.
Reuses the drawing code of render_gifs.py (results/final is not touched).

    python render_sacx_aux_gif.py --new <ckpt> --new-label "SAC-X main+obs" --layouts 49,50,225 \
        --out ../results/sacx_aux/gifs
"""
import argparse, json, os
import numpy as np, torch
import render_gifs as rg
from torch_env import obs_features, feature_dim


def policy(path):
    sx = rg.load_module("sacx_mod", "training_SAC-X.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    T = len(ck["args"]["tasks"].split(","))
    act = sx.MultiHeadActor(feature_dim(5), 2, T); act.load_state_dict(ck["actor"]); act.eval()
    f = lambda o: obs_features(o, 10, sort_obstacles=True)
    return lambda o: torch.tanh(act(f(o))[0][:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="checkpoints/sacx_q_s1_1M.pt")
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-label", default="SAC-X new")
    ap.add_argument("--sac", default="checkpoints/sac_s1_1M.pt")
    ap.add_argument("--layouts", default="", help="'' = old SAC-X timeouts where the new variant reaches the goal")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="../results/sacx_aux/gifs")
    args = ap.parse_args()
    torch.set_num_threads(4); os.makedirs(args.out, exist_ok=True)
    labels = ["SAC-X (old)", args.new_label, "SAC"]
    res = {lab: rg.rollout(policy(p), 512, 2026) for lab, p in zip(labels, (args.old, args.new, args.sac))}
    lay = res[labels[0]][0]
    oc = {k: np.array(v[2]) for k, v in res.items()}
    stats = {k: {o: int((oc[k] == o).sum()) for o in rg.COL} for k in res}
    print(stats)
    if args.layouts:
        pick = [int(i) for i in args.layouts.split(",")]
    else:
        cand = np.nonzero((oc[labels[0]] == "timeout") & (oc[labels[1]] == "goal"))[0]
        def rev(i):  # direction reversals of old SAC-X in the first 60 steps (dithering, not freezing)
            d = np.diff(np.array(res[labels[0]][1][i][:61]), axis=0)
            ok = (np.linalg.norm(d[1:], axis=1) > 1e-3) & (np.linalg.norm(d[:-1], axis=1) > 1e-3)
            return int(((d[1:] * d[:-1]).sum(1)[ok] < 0).sum())
        cand = sorted(cand, key=lambda i: (oc[labels[2]][i] != "goal", -rev(i)))
        print("candidates (layout, old reversals, SAC outcome):", [(int(i), rev(i), oc[labels[2]][i]) for i in cand])
        pick = [int(i) for i in cand[:args.k]]
    rg.ALGCOL.update({"SAC-X": "#e377c2", args.new_label.split()[0]: "#17becf"})
    meta = {"checkpoints": dict(zip(labels, (args.old, args.new, args.sac))), "layout_seed": 2026, "n_layouts": 512,
            "outcomes": stats, "layouts": pick, "files": {}}
    for j, i in enumerate(pick):
        panels = [(lay, i, res[k][1][i], res[k][2][i], k) for k in labels]
        # cap very long (timeout) episodes for file size: show first 60 steps
        panels = [(a, b, c[:61], d, e) for a, b, c, d, e in panels]
        fn = os.path.join(args.out, f"sacx_old_vs_new_vs_sac_layout{i}.gif")
        sz, nf = rg.make_gif(fn, panels, ncols=3, sub=2)
        meta["files"][os.path.basename(fn)] = {"bytes": sz, "frames": nf, "layout_index": i,
                                               "outcomes": {k: res[k][2][i] for k in labels},
                                               "steps": {k: len(res[k][1][i]) - 1 for k in labels}}
        print(fn, sz, nf, meta["files"][os.path.basename(fn)])
    # one 2x2 grid per method (first 4 layouts)
    for k, slug in zip(labels, ("sacx_old", "variant", "sac")):
        panels = [(lay, i, res[k][1][i][:61], res[k][2][i], f"{k} L{i}") for i in pick[:4]]
        fn = os.path.join(args.out, f"grid_{slug}.gif")
        sz, nf = rg.make_gif(fn, panels, ncols=2, sub=2, size=2.6)
        meta["files"][os.path.basename(fn)] = {"bytes": sz, "frames": nf, "layouts": pick[:4],
                                               "outcomes": [res[k][2][i] for i in pick[:4]]}
        print(fn, sz, nf)
    meta["note"] = "deterministic main head (tanh(mu)); episodes longer than 60 steps are shown for the first 60 steps only"
    json.dump(meta, open(os.path.join(args.out, "gifs_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
