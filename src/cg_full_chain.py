# -*- coding: utf-8 -*-
"""cg_full_chain.py - single-point launcher for the full-cohort CADENCE-S chain.
25 cg jobs (5 folds x 5 seeds, done-flag resume) -> certification collector.
No bash loops (that was the double-launch root cause); ONE python process owns
the queue, cg_nuke-safe."""
import os, sys, time, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
PY = sys.executable
JOBS = [(f, s) for f in range(5) for s in (42, 7, 2024, 2025, 12345)]

def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(os.path.join(RUNS, "cg_full_chain.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def main():
    # single instance guard
    import psutil
    me = os.getpid()
    # argv-precise guard (AAR fix): only python.exe whose argv[-1] ENDS with
    # cg_full_chain.py counts. Substring search over full cmdline false-positives
    # on the zcode bash wrapper that embeds the command string in its own argv.
    twins = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.pid == me:
            continue
        cl = p.info["cmdline"] or []
        if (cl and cl[0].replace("\\", "/").endswith("python.exe")
                and cl[-1].replace("\\", "/").endswith("cg_full_chain.py")):
            twins.append(p.pid)
    if twins:
        log(f"another chain alive {twins} - exit")
        return 1
    env = {**os.environ, "V3_FOLDS": "folds_v3_full.json", "V3_FESUF": "_full",
           "V3_KMFOLD": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}
    pending = [(f, s) for f, s in JOBS
               if not os.path.exists(os.path.join(RUNS, f"cg_full_f{f}_s{s}.flag"))]
    log(f"start: {len(pending)}/{len(JOBS)} jobs pending")
    for i, (f, s) in enumerate(pending):
        out = os.path.join(RUNS, f"cg_full_f{f}_s{s}.log")
        t0 = time.time()
        rc = subprocess.call([PY, "-u", "cg_train.py", "--fold", str(f), "--seeds", str(s),
                              "--epochs", "300", "--gbs", "512"],
                             stdout=open(out, "a"), stderr=subprocess.STDOUT,
                             cwd=HERE, env=env)
        ok = os.path.exists(os.path.join(RUNS, f"cg_full_f{f}_s{s}.flag"))
        log(f"[{i+1}/{len(pending)}] f{f} s{s} rc={rc} flag={ok} {round((time.time()-t0)/60,1)}min")
        if not ok:
            log(f"job failed without flag - stopping chain (orderly)")
            return 2
    log("all 25 done -> certification")
    rc = subprocess.call([PY, "-u", "cadence_full_collect.py"],
                         stdout=open(os.path.join(RUNS, "cadence_full.log"), "a"),
                         stderr=subprocess.STDOUT, cwd=HERE, env=env)
    log(f"collector rc={rc}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
