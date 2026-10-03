"""Detached orchestrator for the SAC-X aux study (survives agent interruptions).
1) wait for the seed-1 screening checkpoints, 2) eval them (det + stochastic) on the 512 layouts,
3) pick the top-3 variants by (timeout + collision) of the deterministic main head, launch seeds 0 and 2,
4) wait, eval, run sacx_aux_report.py. Progress is appended to results/sacx_aux/logs/STATUS.md.
    python sacx_aux_queue.py   (run from SAC/, detached)"""
import ctypes, json, os, subprocess, sys, time, datetime

R = os.path.abspath("../results/sacx_aux")
PY = sys.executable
COMMON = ["training_SAC-X.py", "--num-envs", "8", "--updates-per-step", "1", "--batch", "32", "--features",
          "relative_sorted", "--main-reward-scale", "1.0", "--alpha-init", "0.05", "--warmup", "5000",
          "--total-steps", "1000000", "--ckpt-dir", os.path.join(R, "checkpoints")]
VARS = {"aux_main": ["--tasks", "0", "--aux-reward-scale", "0.1"],
        "aux_approach": ["--tasks", "0,1", "--aux-reward-scale", "0.1"],
        "aux_axis": ["--tasks", "0,2,3,4,5", "--aux-reward-scale", "0.1"],
        "aux_obsaware": ["--tasks", "0,6,7", "--aux-reward-scale", "0.1"],
        "aux_full_U": ["--tasks", "0,1,2,3,4,5", "--scheduler", "U", "--aux-reward-scale", "0.1"],
        "aux_full_xi100": ["--tasks", "0,1,2,3,4,5", "--xi", "100", "--aux-reward-scale", "0.1"],
        "aux_full_scale1": ["--tasks", "0,1,2,3,4,5", "--aux-reward-scale", "1.0"],
        "aux_full_seq1": ["--tasks", "0,1,2,3,4,5", "--seq-len", "1", "--batch", "256", "--aux-reward-scale", "0.1"]}
SCREEN = list(VARS)


def log(msg):
    line = f"- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} [queue] {msg}"
    with open(os.path.join(R, "logs", "STATUS.md"), "a", encoding="utf-8") as f: f.write(line + "\n")
    print(line, flush=True)


def alive(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h: return False
    code = ctypes.c_ulong(); ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def ck(name): return os.path.join(R, "checkpoints", name + ".pt")


def wait(names, pids):
    while True:
        done = [n for n in names if os.path.exists(ck(n))]
        dead = [n for n in names if n not in done and n in pids and not alive(pids[n])]
        if len(done) + len(dead) == len(names):
            if dead: log(f"runs died without checkpoint: {dead}")
            return done
        time.sleep(60)


def evaluate(names):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for act in ("mean", "sample"):
        subprocess.run([PY, "sacx_aux_eval.py", "--ckpt"] + [ck(n) for n in names] + ["--name"] + names +
                       ["--act", act, "--out", os.path.join(R, "eval")], env=env,
                       stdout=open(os.path.join(R, "logs", f"eval_{act}_{names[0]}.out"), "a"), stderr=subprocess.STDOUT)


def launch(v, s):
    name = f"{v}_s{s}"
    args = [PY] + COMMON + VARS[v] + ["--seed", str(s), "--log", os.path.join(R, "runs", name + ".csv")]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.Popen(args, env=env, stdout=open(os.path.join(R, "logs", name + ".out"), "w"),
                         stderr=open(os.path.join(R, "logs", name + ".err"), "w"),
                         creationflags=0x00000008 | 0x00000200 | 0x08000000)  # DETACHED | NEW_GROUP | NO_WINDOW
    with open(os.path.join(R, "logs", "pids.log"), "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} queue start {name} pid {p.pid}\n")
    return name, p.pid


def main():
    pids = {}
    pl = os.path.join(R, "logs", "pids.log")  # written by run_sacx_aux.ps1; may be absent (old one removed 2026-10-03)
    for l in (open(pl) if os.path.exists(pl) else []):
        parts = l.split()
        if "pid" in parts: pids[parts[parts.index("pid") - 1]] = int(parts[-1])
    log(f"queue started (pid {os.getpid()}), waiting for the 8 seed-1 screening runs")
    s1 = wait([f"{v}_s1" for v in SCREEN], pids)
    log(f"screening finished: {len(s1)}/8 checkpoints; evaluating")
    evaluate(s1)
    score = {}
    for n in s1:
        d = json.load(open(os.path.join(R, "eval", n + ".json")))
        score[n[:-3]] = (d["timeout"] + d["collision"], d["reversals_per_ep"])
    rank = sorted(score, key=lambda v: score[v])
    log("screening ranking (timeout+collision, reversals/ep, det. main head): " +
        ", ".join(f"{v}={score[v][0]}/{score[v][1]:.2f}" for v in rank))
    top = rank[:3]
    new = [launch(v, s) for v in top for s in (0, 2)]
    log("launched seeds 0/2 for " + ", ".join(top) + ": " + ", ".join(f"{n} pid {p}" for n, p in new))
    names = [n for n, _ in new]
    done = wait(names, dict(new))
    log(f"seed 0/2 runs finished: {len(done)}/{len(names)}; evaluating")
    evaluate(done)
    subprocess.run([PY, "sacx_aux_report.py"], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                   stdout=open(os.path.join(R, "logs", "report.out"), "w"), stderr=subprocess.STDOUT)
    log("ALL DONE: eval JSONs in eval/, table eval_table.md, plot sacx_aux_comparison.png")


if __name__ == "__main__":
    main()
