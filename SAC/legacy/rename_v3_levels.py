# ARCHIVED 2026-10-03 (moved to SAC/legacy/): already executed on 2026-10-02 - DO NOT RUN AGAIN
# (a second run would rename the *_hard files again). Kept only as a record of the renaming rule.
"""One-off (2026-10-02): rename results/v3 artefacts after the level decision:
   '<mode>' (original density) -> '<mode>_hard', '<mode>_easy' -> '<mode>', '<mode>_step05' -> '<mode>_hard_step05'.
Moves only (no deletion); refuses to overwrite. Old yaml files are replaced by the new set (copied in separately)."""
import os, re, json, sys
R = "../results/v3"
modes = ["static", "random", "bounce"]
plan = []
for d in ("runs", "checkpoints", "logs", "eval", "gifs"):
    for fn in sorted(os.listdir(os.path.join(R, d))):
        new = fn
        for m in modes:
            # order matters: step05 and easy first
            if re.search(rf"(^|_){m}_step05", fn): new = fn.replace(f"{m}_step05", f"{m}_hard_step05"); break
            if re.search(rf"(^|_){m}_easy", fn): new = fn.replace(f"{m}_easy", f"{m}"); break
            if re.search(rf"(^|_){m}(_s\d|\.json|_ppo_vs)", fn): new = re.sub(rf"(^|_){m}(_s\d|\.json|_ppo_vs)", rf"\g<1>{m}_hard\2", fn, count=1); break
        if new != fn: plan.append((os.path.join(R, d, fn), os.path.join(R, d, new)))
srcs = {a for a, _ in plan}
for a, b in plan:
    assert not (os.path.exists(b) and b not in srcs), f"would overwrite {b}"
# two-phase to avoid chains (static -> static_hard while static_easy -> static)
for a, b in plan: os.rename(a, a + ".tmpmv")
for a, b in plan: os.rename(a + ".tmpmv", b); print(os.path.basename(a), "->", os.path.basename(b))
mp = os.path.join(R, "gifs", "gifs_meta.json")
if os.path.exists(mp):
    meta = json.load(open(mp)); newm = {}
    for k, v in meta.items():
        nk = k
        for m in modes:
            if k.startswith(f"{m}_easy_"): nk = k.replace(f"{m}_easy_", f"{m}_"); break
            if k.startswith(f"{m}_ppo"): nk = k.replace(f"{m}_ppo", f"{m}_hard_ppo"); break
        v["note"] = "renamed 2026-10-02: config paths inside refer to the old file names (v3_<mode>_easy.yaml = now v3_<mode>.yaml, v3_<mode>.yaml = now v3_<mode>_hard.yaml)"
        newm[nk] = v
    json.dump(newm, open(mp, "w"), indent=1)
