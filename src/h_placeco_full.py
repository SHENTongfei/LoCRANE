# -*- coding: utf-8 -*-
"""h_placeco_full.py - full-cohort eval-level pipeline placebo (pre-registered
false-positive control, eval-level analogue of the pipeline-level x1000).
Permutes CHRONO age labels against the frozen 5-seed OOF ensemble x1000:
  - H1 tail-gap (L/M median residual diff)
  - H2 pred-space pace ratio
  - H4 monotone Y<M<L certificate fraction
p = fraction of nulls at least as extreme as observed.
Writes runs_v3/h_placeco_full_results.json.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
RNG = np.random.default_rng(20260931)
N_PERM = 1000

def load_full():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    fold_true = {f: sorted(str(i) for i in folds[str(f)]) for f in range(5)}
    ids = [t for f in range(5) for t in fold_true[f]]
    acc = {}
    for f in range(5):
        for s in [42, 7, 2024, 2025, 12345]:
            z = np.load(os.path.join(RUNS, f"cg_full_M_f{f}_s{s}.npz"), allow_pickle=True)
            assert [str(x) for x in z["ids"]] == [t[:16] for t in fold_true[f]]
            for i, tid in enumerate(fold_true[f]):
                acc.setdefault(tid, []).append(float(z["M"][i]))
    M = np.array([np.mean(acc[i]) for i in ids])
    sig = np.array([np.std(acc[i]) + 1e-6 for i in ids])
    age = lab.loc[ids, "age"].to_numpy(np.float64)
    return M, sig, age

def tail_gap(chrono, M, band):
    r = M - chrono
    return float(np.median(r[band == 2]) - np.median(r[band == 1]))

def pace_ratio(chrono, M, band):
    def vel(b, lo, hi):
        m = (band == b) & (chrono >= lo) & (chrono < hi)
        c, mm = chrono[m], M[m]
        if len(c) < 4:
            return None
        o = np.argsort(c)
        dc, dm = np.abs(np.diff(c[o])), np.abs(np.diff(mm[o]))
        g = dc > 0.5
        return float(dm[g].sum() / dc[g].sum()) if dc[g].sum() > 0 else None
    vL, vM = vel(2, 90, 105), vel(1, 70, 90)
    if vL is None or vM in (None, 0):
        return None
    return vL / vM

def cert_fracs(M, sig, chrono, band):
    cal = band < 2
    stud = (M - chrono) / sig
    q = float(np.quantile(stud[cal], np.ceil((cal.sum() + 1) * 0.9) / (cal.sum() + 1)))
    cert = (M + q * sig) < chrono
    return [float(cert[band == d].mean()) for d in (0, 1, 2)]

def main():
    M, sig, age = load_full()
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    obs_h1 = tail_gap(age, M, band)
    obs_h2 = pace_ratio(age, M, band)
    obs_h4 = cert_fracs(M, sig, age, band)
    h1_null, h2_null, h4_mono = [], [], 0
    for _ in range(N_PERM):
        ap = age[RNG.permutation(len(age))]
        # pool-protocol-consistent: band labels stay TRUE, chrono values permuted
        h1_null.append(tail_gap(ap, M, band))
        v = pace_ratio(ap, M, band)
        if v is not None:
            h2_null.append(v)
        fr = cert_fracs(M, sig, ap, band)
        h4_mono += int(fr[0] <= fr[1] <= fr[2])
    out = {"n": int(len(age)), "n_perm": N_PERM,
           "level": "eval-level placebo: chrono labels permuted against frozen 5-seed OOF ensemble",
           "H1": {"obs_tail_gap": obs_h1,
                  "p_perm_less": float((np.array(h1_null) <= obs_h1).mean())},
           "H2_predpace": {"obs": obs_h2, "p_perm_greater": float((np.array(h2_null) >= obs_h2).mean())},
           "H4": {"obs_monotone": bool(obs_h4[0] <= obs_h4[1] <= obs_h4[2]),
                  "null_monotone_rate": h4_mono / N_PERM,
                  "obs_fracs": obs_h4}}
    json.dump(out, open(os.path.join(RUNS, "h_placeco_full_results.json"), "w"), indent=1, default=float)
    print("[placeco-full]", json.dumps(out, default=float), flush=True)

if __name__ == "__main__":
    main()
