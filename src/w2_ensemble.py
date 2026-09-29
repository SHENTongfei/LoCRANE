# -*- coding: utf-8 -*-
"""w2_ensemble.py - 25-ckpt (5 folds x 5 seeds, B3 final-epoch) ensemble over the
two external cohorts' v3 feature caches -> per-sample base age predictions +
cls representation (mean over all 25 models). Writes runs_v3/w2_ens_<cohort>.npz
(pred, cls, age, band, sex, group, ids). GPU-light (~1-2 min, 130 samples).
argv-precise single-instance guard (orderly).
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CRANE_DATA_DIR", "C:/Users/TS/Desktop/crane")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
import pandas as pd
import torch
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 2024, 2025, 12345]
PY_EXE = os.path.basename(sys.executable)

def log(m): print(f"[w2ens] {m}", flush=True)

def build_model(n_immune, n_genes, assignment):
    return CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                    n_immune=n_immune, use_moe=False, use_gate=False,
                    use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)

def main():
    import psutil
    twins = [p.pid for p in psutil.process_iter(["pid", "cmdline"])
             if p.pid != os.getpid() and (p.info["cmdline"] or [None])[0]
             and p.info["cmdline"][-1].replace("\\", "/").endswith("w2_ensemble.py")
             and p.info["cmdline"][0].replace("\\", "/").endswith("python.exe")]
    if twins:
        log(f"another w2_ensemble alive {twins} - orderly exit")
        return 1
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    for cohort in ["cohort2", "gtex"]:
        z = np.load(os.path.join(RUNS, f"w2_ext_fe_{cohort}.npz"))
        Xq, Xs = torch.from_numpy(z["Xq"]).float().to(DEV), torch.from_numpy(z["Xs"]).float().to(DEV)
        SV = torch.from_numpy(z["sex"]).to(DEV)
        IV = torch.from_numpy(z["imm"].astype(np.float32)).to(DEV)
        RV = torch.from_numpy(z["rda"]).float().to(DEV)
        BV = torch.from_numpy(np.eye(3, dtype=np.float32)[z["band"]]).to(DEV)
        mu, sd = 72.0, 20.0  # folded in below from fold0
        p0 = C.load_fold(0)[2]
        mu, sd = float(p0["age"].mean()), max(1e-6, float(p0["age"].std()))
        age_preds, cls_acc = [], []
        for f in range(5):
            km = np.load(os.path.join(DATA3, f"km256_f{os.environ.get('V3_KMFOLD', '0')}.npz"))
            assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
            for s in SEEDS:
                ck = os.path.join(RUNS, "ckpt_cg", f"cg_k_f{f}_s{s}.pt")
                d = torch.load(ck, map_location=DEV, weights_only=False)
                model = build_model(z["imm"].shape[1], len(cfg_genes), assign)
                model.load_state_dict(d["model"]); model.eval()
                with torch.no_grad():
                    out = model(Xq, Xs, SV, IV, RV, BV, plm=None)
                age_preds.append(out["age"].cpu().numpy() * sd + mu)
                cls_acc.append(out["cls"].cpu().numpy())
                del model
        pred = np.mean(age_preds, 0)
        cls = np.mean(cls_acc, axis=0)
        np.savez(os.path.join(RUNS, f"w2_ens_{cohort}.npz"),
                 pred=pred, cls=cls.astype(np.float32),
                 age=z["age"], band=z["band"], sex=z["sex"], group=z["group"], ids=z["ids"])
        log(f"{cohort}: n={len(pred)} pred range [{pred.min():.1f},{pred.max():.1f}] "
            f"vs age [{z['age'].min():.0f},{z['age'].max():.0f}] cls {cls.shape}")
    log("DONE")

if __name__ == "__main__":
    main()
