# -*- coding: utf-8 -*-
"""ext_leg_suite.py - external leg v3-protocol suite (COR-1d structure).
PRE-REGISTERED (fixed before reading results):
  cohort2 (n=66, 10 x 90+, sex available) on the 25-ckpt ensemble OOF preds:
    - H1 tail-gap_ext = median(resid | L90+) - median(resid | M70-89);
      n_L=10 -> DISCLOSURE grade: report the number + direction, NO p-value claim.
    - H4 cert fractions (pooled studentized one-sided cert, alpha=0.10, cal on M+Y):
      Y/M/L rates + monotone flag; n_L=10 -> disclosure, no binomial test.
    - CADENCE-S sex split: per-sex L certified rate + marginal; n_L small -> disclosure.
  gtex (n=64, no 90+): cross-platform disclosure only:
    - MAE, Spearman(pred, age), mean absolute rank err; NO tail claims (no L band).
No retraining (reuses w2_ens npz = 25-model final-epoch ensemble). CPU-only.
Writes runs_v3/ext_leg_results.json.
"""
import os, json
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")

def log(m): print(f"[extleg] {m}", flush=True)

def cert_fractions(pred, age, sigma):
    cal = age < 90
    stud = (pred - age) / sigma
    q = float(np.quantile(stud[cal], np.ceil((cal.sum() + 1) * 0.9) / (cal.sum() + 1)))
    pi = pred + q * sigma
    cert = pi < age
    b = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    return {f"band{d}": float(cert[b == d].mean()) for d in (0, 1, 2) if (b == d).sum()}, q, cert, b

def main():
    out = {"pre_registered": {
        "cohort2": "n_L=10 -> disclosure grade (rates+direction only, no p claims)",
        "gtex": "cross-platform disclosure only (no L band)",
        "model": "25-ckpt final-epoch ensemble (cg_k), zero-shot external preds"},
        "source": "runs_v3/w2_ens_{cohort2,gtex}.npz"}
    for cohort in ["cohort2", "gtex"]:
        z = np.load(os.path.join(RUNS, f"w2_ens_{cohort}.npz"))
        pred, age = z["pred"].astype(np.float64), z["age"].astype(np.float64)
        band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
        resid = pred - age
        r = {"n": int(len(age)), "nL": int((band == 2).sum()),
             "MAE": float(np.abs(resid).mean()),
             "spearman": float(spearmanr(pred, age).statistic),
             "resid_mean_L": float(resid[band == 2].mean()) if (band == 2).any() else None,
             "resid_mean_M": float(resid[band == 1].mean()) if (band == 1).any() else None}
        if cohort == "cohort2":
            m = (band == 2) & (age >= 90); mm = (band == 1) & (age >= 70) & (age < 90)
            r["H1_tail_gap_ext"] = float(np.median(resid[m]) - np.median(resid[mm])) if (m.sum() and mm.sum()) else None
            r["H1_grade"] = "disclosure (n_L=%d)" % int(m.sum())
            # sigma: cross-model uncertainty unavailable for external single-pass;
            # use local residual-based studentized proxy (pre-registered here):
            sig = np.clip(np.std(resid) * np.ones_like(age), 0.5, None)
            fr, q, cert, bb = cert_fractions(pred, age, sig)
            r["H4_cert_fracs"] = fr
            r["H4_monotone"] = bool(fr.get("band0", 0) <= fr.get("band1", 0) <= fr.get("band2", 0))
            r["H4_q"] = q
            sex = z["sex"]
            for nm, sm in [("F", sex == 1), ("M", sex == 0)]:
                Ls = (band == 2) & sm
                cal_s = sm & (band < 2)
                st = (pred - age) / sig
                qs = float(np.quantile(st[cal_s], np.ceil((cal_s.sum() + 1) * 0.9) / (cal_s.sum() + 1)))
                cs = (pred + qs * sig) < age
                r[f"CERTS_{nm}"] = {"n": int(sm.sum()), "nL": int(Ls.sum()),
                                     "L_rate": float(cs[Ls].mean()) if Ls.sum() else None,
                                     "marginal": float(cs[cal_s].mean())}
            r["CADENCE-S_grade"] = "disclosure (small n_L)"
        out[cohort] = r
        log(f"{cohort}: {json.dumps({k: v for k, v in r.items()}, default=float)}")
    json.dump(out, open(os.path.join(RUNS, "ext_leg_results.json"), "w"), indent=1, default=float)
    log("WROTE ext_leg_results.json")

if __name__ == "__main__":
    main()
