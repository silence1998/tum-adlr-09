"""Detached job queue for RL on env v3 (SAC, SAC-X main+approach), max N concurrent processes.
    python run_v3_rl_queue.py --max-par 4 --seeds 0 1      (from SAC/)
Appends to results/v3/logs/STATUS.md and results/v3/logs/pids.log."""
import argparse, datetime, os, subprocess, sys, time

R = os.path.abspath("../results/v3")
SACX = ["--tasks", "0,1", "--scheduler", "Q", "--num-envs", "8", "--updates-per-step", "1", "--batch", "32",
        "--features", "relative_sorted", "--main-reward-scale", "1.0", "--alpha-init", "0.05", "--warmup", "5000",
        "--aux-reward-scale", "0.1"]


def log(msg):
    line = f"- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} [rl-queue] {msg}"
    open(os.path.join(R, "logs", "STATUS.md"), "a", encoding="utf-8").write(line + "\n"); print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-par", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--modes", nargs="+", default=["static", "random", "bounce"])
    ap.add_argument("--algos", nargs="+", default=["sac", "sacx"])
    a = ap.parse_args()
    jobs = [(al, m, s) for s in a.seeds for m in a.modes for al in a.algos]
    for d in ("runs", "logs", "checkpoints"):
        os.makedirs(os.path.join(R, d), exist_ok=True)
    log(f"queue pid {os.getpid()} started: {len(jobs)} jobs, max {a.max_par} concurrent")
    running = {}
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    while jobs or running:
        for name, p in list(running.items()):
            if p.poll() is not None:
                log(f"finished {name} (exit {p.returncode})"); del running[name]
        while jobs and len(running) < a.max_par:
            al, m, s = jobs.pop(0)
            name = f"{al}_{m}_s{s}"
            if os.path.exists(os.path.join(R, "checkpoints", name + ".pt")):
                log(f"skip {name} (checkpoint exists)"); continue
            script = "training_SAC.py" if al == "sac" else "training_SAC-X.py"
            args = [sys.executable, script] + (SACX if al == "sacx" else []) + [
                "--env-version", "3", "--env-config", f"../configs/v3_{m}.yaml", "--total-steps", str(a.steps),
                "--seed", str(s), "--eval-every", "25000", "--log", os.path.join(R, "runs", name + ".csv"),
                "--ckpt-dir", os.path.join(R, "checkpoints")]
            p = subprocess.Popen(args, env=env, stdout=open(os.path.join(R, "logs", name + ".out"), "w"),
                                 stderr=open(os.path.join(R, "logs", name + ".err"), "w"),
                                 creationflags=0x00000008 | 0x00000200 | 0x08000000)
            running[name] = p
            open(os.path.join(R, "logs", "pids.log"), "a").write(
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} rl-queue start {name} pid {p.pid}\n")
            log(f"started {name} pid {p.pid}")
        time.sleep(30)
    log("queue finished")


if __name__ == "__main__":
    main()
