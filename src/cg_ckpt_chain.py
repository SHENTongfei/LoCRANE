# -*- coding: utf-8 -*-
"""cg_ckpt_chain.py - Phase-2 launcher: H2/H3 full-cohort power runs.
Same 25-job grid as Phase 1 but flag-prefix cg_k and CG_SAVECKPT=1, so the
final per-job model weights land in runs_v3/ckpt_cg/ for representation
extraction (H2 full-cohort) while the collector-grade M-predictions are
unchanged. Strictly serialized after Phase 1 (cg_full_chain + collector).
argv-precise single-instance guard, done-flag resume, stop-on-job-failure."""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
# pin the CUDA interpreter explicitly (anaconda3 base = CPU-only torch,
# which is the root cause of the 12:39 failed launch). Overridable.
PY = os.environ.get("CRANE_PY", r"C:/Users/TS/.conda/envs/plm_master/python.exe")
JOBS = [(f, s) for f in range(5) for s in (42, 7, 2024, 2025, 12345)]

def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(os.path.join(RUNS, "cg_ckpt_chain.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def main():
    import psutil
    me = os.getpid()
    twins = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.pid == me:
            continue
        cl = p.info["cmdline"] or []
        if (cl and cl[0].replace("\\", "/").endswith("python.exe")
                and cl[-1].replace("\\", "/").endswith("cg_ckpt_chain.py")):
            twins.append(p.pid)
    if twins:
        log(f"another ckpt-chain alive {twins} - exit")
        return 1
    env = {**os.environ, "V3_FOLDS": "folds_v3_full.json", "V3_FESUF": "_full",
           "V3_KMFOLD": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "CG_FLAGPRE": "cg_k", "CG_SAVECKPT": "1"}
    pending = [(f, s) for f, s in JOBS
               if not os.path.exists(os.path.join(RUNS, f"cg_k_f{f}_s{s}.flag"))]
    log(f"start: {len(pending)}/{len(JOBS)} jobs pending (prefix cg_k, ckpt on)")
    for i, (f, s) in enumerate(pending):
        out = os.path.join(RUNS, f"cg_k_f{f}_s{s}.log")
        t0 = time.time()
        rc = subprocess.call([PY, "-u", "cg_train.py", "--fold", str(f), "--seeds", str(s),
                              "--epochs", "300", "--gbs", "512"],
                             stdout=open(out, "a"), stderr=subprocess.STDOUT,
                             cwd=HERE, env=env)
        ok = os.path.exists(os.path.join(RUNS, f"cg_k_f{f}_s{s}.flag"))
        log(f"[{i+1}/{len(pending)}] f{f} s{s} rc={rc} flag={ok} {round((time.time()-t0)/60,1)}min")
        if not ok:
            log(f"job failed without flag - stopping chain (orderly)")
            return 2
    log("all 25 ckpt jobs done -> H2 full-cohort extraction")
    rc = subprocess.call([PY, "-u", "h2_full_repr.py"],
                         stdout=open(os.path.join(RUNS, "h2_full.log"), "a"),
                         stderr=subprocess.STDOUT, cwd=HERE, env=env)
    log(f"h2_full_repr rc={rc}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
