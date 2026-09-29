# -*- coding: utf-8 -*-
"""h3_l5_full.py - H3 L6 escalation: full-cohort (1715, L=624) noise-ratio
battery from Phase-1 chain artifacts (NO retrain): per-sample 5-seed sigma
rebuilt from the 25 cg_full_M npz (same formula as cadence_full_collect.py).

Two estimators, both decile-matched on |resid| within the L/Y pool:
  nr_seed  = mean_d std_5seed(L,decile d) / std_5seed(Y,decile d)
  nr_width = same on CQR proxy width w = 2*z(0.875)*sigma
Pre-registered H3 gate: nr <= 0.85 and permutation p < 0.01 (L vs Y).
Writes runs_v3/h3_l5_full_results.json.
"""
import os, json
import numpy as np
import pandas as pd
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
RNG = np.random.default_rng(20260930)
Z_0875 = 1.53885
N_PERM = 2000

def log(m): print(f"[h3l5f] {m}", flush=True)

def load_full():
    # npz ids are U16-truncated (38 external ids >16 chars) - rebuild ids from
    # folds_v3_full.json (authoritative), row order = sorted(str(i) for i in fold)
    import glob
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    fold_true = {f: sorted(str(i) for i in folds[str(f)]) for f in range(5)}
    ids = [t for f in range(5) for t in fold_true[f]]
    for f in range(5):
        for s in [42, 7, 2024, 2025, 12345]:
            z = np.load(os.path.join(RUNS, f"cg_full_M_f{f}_s{s}.npz"), allow_pickle=True)
            assert [str(x) for x in z["ids"]] == [t[:16] for t in fold_true[f]], \
                f"row misalignment f{f} s{s}"
    acc = defaultdict(list)
    for f in range(5):
        true_ids = fold_true[f]
        for s in [42, 7, 2024, 2025, 12345]:
            z = np.load(os.path.join(RUNS, f"cg_full_M_f{f}_s{s}.npz"), allow_pickle=True)
            for i, sid in enumerate(true_ids):
                acc[sid].append(float(z["M"][i]))
    Mmat = np.array([[acc[i][j] for j in range(5)] for i in ids])
    assert all(len(v) == 5 for v in acc.values()), "some sample lacks 5-seed preds"
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    age = lab.loc[ids, "age"].to_numpy(np.float64)
    return Mmat, age, ids

def nr(mat, age, band2, est="seed", dec=None):
    L = band2 == 2; Y = band2 == 0
    resid = np.abs(mat.mean(1) - age)
    dec = dec if dec is not None else dec_of(resid)
    out = []
    for d in range(10):
        mL = L & (dec == d); mY = Y & (dec == d)
        if mL.sum() >= 3 and mY.sum() >= 3:
            if est == "seed":
                out.append(mat[mL].std(1).mean() / mat[mY].std(1).mean())
            else:
                w = 2 * Z_0875 * mat.std(1)
                out.append(w[mL].mean() / w[mY].mean())
    return float(np.mean(out)) if out else None

def dec_of(r):
    order = np.argsort(r, kind="stable")
    out = np.empty(len(r))
    out[order] = np.arange(len(r))
    return np.clip((out / len(r) * 10).astype(int), 0, 9)

def main():
    mat, age, ids = load_full()
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    dec = dec_of(np.abs(mat.mean(1) - age))
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    sex = lab.loc[[str(i) for i in ids], "sex"].to_numpy()
    isF = np.array([str(s).upper().startswith("F") for s in sex])
    out = {"n": int(len(ids)), "nL": int((band == 2).sum()),
           "nM": int((band == 1).sum()), "nY": int((band == 0).sum())}
    for est in ["seed", "width"]:
        obs = nr(mat, age, band, est, dec)
        pool = (band == 2) | (band == 0)
        b2 = band.copy()
        nulls = []
        for _ in range(N_PERM):
            b2[pool] = RNG.permutation(b2[pool])
            r = nr(mat, age, b2, est, dec)
            if r is not None:
                nulls.append(r)
        nulls = np.array(nulls)
        out[f"nr_{est}"] = obs
        out[f"null_{est}_mean"] = float(nulls.mean())
        out[f"p_{est}_less"] = float((nulls <= obs).mean())
        # per-sex split (disclosure, no extra gate)
        for nm, sm in [("F", isF), ("M", ~isF)]:
            bs = band.copy()
            bs[~sm] = 1
            out[f"nr_{est}_{nm}"] = nr(mat, age, bs, est, dec)
        log(f"{est}: nr={obs:.4f} null_mean={nulls.mean():.4f} p_less={float((nulls<=obs).mean()):.4f} "
            f"(gate: <=0.85 & p<0.01) | F {out[f'nr_{est}_F']:.3f} / M {out[f'nr_{est}_M']:.3f}")
    with open(os.path.join(RUNS, "h3_l5_full_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log("WROTE h3_l5_full_results.json")

if __name__ == "__main__":
    main()
