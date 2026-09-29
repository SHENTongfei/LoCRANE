# -*- coding: utf-8 -*-
"""cg_legC.py - Leg C strict parity: single-point launcher.
Step 1 (GPU ~17min): eager train_v3 stage0 fold0 seed42 full-cohort 300ep
  (V3_FOLDS=folds_v3_full.json, V3_FESUF=_full, V3_KMFOLD=0, V3_SAVECKPT=1)
Step 2 (CPU/GPU-light): legC_compare.py vs existing cg_full_M_f0_s42.npz
Ordered: runs only if no other python chain alive (evidence-first, AAR rule).
argv-precise guard + single-instance. Writes runs_v3/legC_chain.log.
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
PY = os.environ.get("CRANE_PY", r"C:/Users/TS/.conda/envs/plm_master/python.exe")

def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(os.path.join(RUNS, "legC_chain.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def main():
    import psutil
    me = os.getpid()
    our_scripts = ("cg_full_chain.py", "cg_ckpt_chain.py", "cg_legC.py", "sequencer_phase2.py", "cg_train.py")
    alive = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.pid == me:
            continue
        cl = p.info["cmdline"] or []
        if len(cl) >= 2 and cl[0].replace("\\", "/").endswith("python.exe"):
            last = cl[-1].replace("\\", "/")
            if last.endswith(our_scripts) or "cg_train.py" in " ".join(cl):
                alive.append((p.pid, last))
    if alive:
        log(f"another trainer/chain alive {alive} - orderly exit")
        return 1
    ckpt = os.path.join(RUNS, "ckpt", "stage0_cgpar_f0_s42_best.pt")
    if os.path.exists(ckpt):
        log("eager ckpt exists - skip training (resume)")
    else:
        env = {**os.environ, "V3_FOLDS": "folds_v3_full.json", "V3_FESUF": "_full",
               "V3_KMFOLD": "0", "V3_EPOCHS": "300", "V3_BCAP": "512", "V3_SAVECKPT": "1",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "KMP_DUPLICATE_LIB_OK": "TRUE"}
        rc = subprocess.call([PY, "-u", "train_v3.py", "--arm", "stage0", "--folds", "0",
                              "--seeds", "42", "--tag", "cgpar"],
                             stdout=open(os.path.join(RUNS, "legC_eager.log"), "a"),
                             stderr=subprocess.STDOUT, cwd=HERE, env=env)
        log(f"eager train rc={rc}")
        if rc != 0 or not os.path.exists(ckpt):
            log("eager run failed - orderly stop (no compare)")
            return 2
    rc = subprocess.call([PY, "-u", "legC_compare.py"],
                         stdout=open(os.path.join(RUNS, "legC_compare.log"), "a"),
                         stderr=subprocess.STDOUT, cwd=HERE,
                         env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
    log(f"legC_compare rc={rc}")
    verdict = json.load(open(os.path.join(RUNS, "legC_results.json"))) if os.path.exists(os.path.join(RUNS, "legC_results.json")) else {}
    log(f"VERDICT: PASS={verdict.get('PASS')} pair_spearman={verdict.get('pair_spearman')} dMAE={verdict.get('MAE_diff')}")
    return rc

import json

if __name__ == "__main__":
    sys.exit(main())
