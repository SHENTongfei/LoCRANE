# -*- coding: utf-8 -*-
"""w2_probe.py - W2 cohort fast-adaptation (meta-lite, pre-registered in
TAIL_PROTOCOL_v4 W2): ridge probe cls->age with k anchors on each external
cohort, zero retraining. PRE-REGISTERED DECISION RULES (fixed before reading
any result):
  - k arms: {5, 20}; k=100 INFEASIBLE (cohort2 n=66, gtex n=64 < 100) - documented.
  - 10 resampling seeds per arm (rng seed 20260930+i).
  - nested eval: probe fit on k random anchors, scored on the remaining n-k.
  - metrics: MAE_probe(k), delta(k)=MAE_base-MAE_probe(k) (>0 = adaptation wins),
    decile-AUC of age ranking (probe vs base).
  - MONO: mean delta(20) >= mean delta(5) within a cohort.
  - WIN = MONO and mean delta(20) > 0.05 yr AND mean delta(5) > 0.
  - KILL = otherwise -> honest verdict "zero-shot 25-model ensemble saturated;
    k-shot adaptation adds no value" (reported as kill-line hit, not hidden).
Writes runs_v3/w2_probe_results.json (verdicts per cohort + all raw curves).
CPU-only, seconds.
"""
import os, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
RNG_SEED = 20260930

def log(m): print(f"[w2probe] {m}", flush=True)

def age_auc(pred, age):
    med = np.median(age)
    y = (age > med).astype(int)
    try:
        return float(roc_auc_score(y, pred))
    except ValueError:
        return None

def main():
    out = {"pre_registered": {
        "k_arms": [5, 20], "k100": "INFEASIBLE (n=66/64 < 100, documented)",
        "n_resample_seeds": 10,
        "WIN": "MONO and mean_delta(20)>0.05yr and mean_delta(5)>0",
        "KILL": "else -> zero-shot saturated, adaptation adds nothing (honest kill-line)"}}
    for cohort in ["cohort2", "gtex"]:
        z = np.load(os.path.join(RUNS, f"w2_ens_{cohort}.npz"))
        cls, age = z["cls"], z["age"].astype(np.float64)
        base = z["pred"].astype(np.float64)
        n = len(age)
        mae_base = float(np.abs(base - age).mean())
        auc_base = age_auc(base, age)
        res = {"n": n, "MAE_base": mae_base, "AUC_base": auc_base,
               "cohort_note": "Israel chip whole blood (independent population)" if cohort == "cohort2"
                               else "GTEx tissue reference (cross-platform/tissue disclosure)"}
        for k in [5, 20]:
            ds, deltas, aucs = [], [], []
            for i in range(10):
                rng = np.random.default_rng(RNG_SEED + i)
                anchor = rng.choice(n, k, replace=False)
                rest = np.setdiff1d(np.arange(n), anchor)
                probe = Ridge(alpha=1.0).fit(cls[anchor], age[anchor]).predict(cls[rest])
                ds.append(probe)
                deltas.append(float(np.abs(base[rest] - age[rest]).mean() - np.abs(probe - age[rest]).mean()))
                aucs.append(age_auc(probe, age[rest]))
            res[f"delta_mean_k{k}"] = float(np.mean(deltas))
            res[f"delta_pos_frac_k{k}"] = float(np.mean(np.array(deltas) > 0))
            res[f"auc_probe_k{k}"] = float(np.nanmean(aucs)) if aucs else None
            res[f"mae_probe_k{k}"] = mae_base - res[f"delta_mean_k{k}"]
        mono = res["delta_mean_k20"] >= res["delta_mean_k5"]
        win = mono and res["delta_mean_k20"] > 0.05 and res["delta_mean_k5"] > 0
        res["MONO"] = bool(mono)
        res["VERDICT"] = "WIN" if win else "KILL"
        out[cohort] = res
        log(f"{cohort}: base MAE={mae_base:.2f} AUC={auc_base} | "
            f"d5={res['delta_mean_k5']:+.3f} d20={res['delta_mean_k20']:+.3f} "
            f"MONO={mono} -> {res['VERDICT']}")
    out["protocol_ref"] = "TAIL_PROTOCOL_v4 W2 cohort-fast-adaptation meta-lite"
    json.dump(out, open(os.path.join(RUNS, "w2_probe_results.json"), "w"), indent=1, default=float)
    log("WROTE w2_probe_results.json")

if __name__ == "__main__":
    main()
