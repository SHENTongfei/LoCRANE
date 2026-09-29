# -*- coding: utf-8 -*-
"""headlines_full.py - full-cohort (1715) headline numbers, CPU-only.
Upgrades the pool-era (324 OOF) headlines to full-cohort scale using the
Phase-1 25 npz (5-seed ensemble per sample), all protocol-consistent:
  - B3 OOF: overall MAE/Spearman + per-band MAE
  - H1 tail-gap: median(M-age|L) - median(M-age|M) + eval-level placebo
    (chrono-perm x2000, same method as h_placebo.py; pre-registered
    PIPELINE-level x1000 still documented pending)
  - H2 pace-ratio (prediction space): L-band vel / M-band(70-89) vel,
    p_perm_greater (companion to the repr-space H2 awaiting Phase-2 ckpts)
  - H4 significant-decelerator fraction per band (pooled studentized
    one-sided certificate, alpha=0.10) -> monotonicity Y<M<L
  - H3 direction disclosure: L vs Y noise-ratio (5-seed sigma, decile)
ids rebuilt from folds_v3_full.json (npz ids are U16-truncated - AAR #6).
Writes runs_v3/headlines_full.json.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
RNG = np.random.default_rng(20260930)
N_PERM = 2000

def load_full():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    fold_true = {f: sorted(str(i) for i in folds[str(f)]) for f in range(5)}
    ids = [t for f in range(5) for t in fold_true[f]]
    assert len(ids) == 1715
    acc = {}
    for f in range(5):
        for s in [42, 7, 2024, 2025, 12345]:
            z = np.load(os.path.join(RUNS, f"cg_full_M_f{f}_s{s}.npz"), allow_pickle=True)
            assert [str(x) for x in z["ids"]] == [t[:16] for t in fold_true[f]], f"misalign f{f} s{s}"
            for i, tid in enumerate(fold_true[f]):
                acc.setdefault(tid, []).append(float(z["M"][i]))
    M = np.array([np.mean(acc[i]) for i in ids])
    sig = np.array([np.std(acc[i]) + 1e-6 for i in ids])
    age = lab.loc[ids, "age"].to_numpy(np.float64)
    sex = lab.loc[ids, "sex"].to_numpy()
    return M, sig, age, sex, ids

def tail_gap(chrono, M, band):
    r = M - chrono
    return float(np.median(r[band == 2]) - np.median(r[band == 1]))

def pace_ratio(chrono, M, band):
    def vel(b):
        m = (band == b) & (b != 0)
        if b == 1:
            m = (band == 1) & (chrono >= 70)
        c, mm = chrono[m], M[m]
        o = np.argsort(c)
        dc, dm = np.abs(np.diff(c[o])), np.abs(np.diff(mm[o]))
        g = dc > 0.5
        return float(dm[g].sum() / dc[g].sum()) if dc[g].sum() > 0 else np.nan
    vM = vel(1)
    return float(vel(2) / vM) if vM and vM > 1e-9 else None

def main():
    M, sig, age, sex, ids = load_full()
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    out = {"n": int(len(ids)), "nL": int((band == 2).sum()),
           "nM": int((band == 1).sum()), "nY": int((band == 0).sum()),
           "model": "B3-scale cg_full 5-fold x 5-seed OOF ensemble (final-epoch states)"}
    # B3 OOF metrics
    out["oof"] = {"MAE": float(np.abs(M - age).mean()),
                  "spearman": float(spearmanr(M, age).statistic),
                  "MAE_by_band": {f"band{b}": float(np.abs(M[band == b] - age[band == b]).mean())
                                    for b in [0, 1, 2]},
                  "resid_mean_by_band": {f"band{b}": float((M[band == b] - age[band == b]).mean())
                                          for b in [0, 1, 2]}}
    # H1 tail-gap + eval-level placebo
    obs = tail_gap(age, M, band)
    nulls = [tail_gap(age[RNG.permutation(len(age))], M, band) for _ in range(N_PERM)]
    out["H1"] = {"obs_tail_gap": obs, "p_perm_less": float((np.array(nulls) <= obs).mean()),
                 "null_mean": float(np.mean(nulls)), "n_perm": N_PERM}
    # H2 pace-ratio (prediction space)
    obs2 = pace_ratio(age, M, band)
    nulls2 = [pace_ratio(age[RNG.permutation(len(age))], M, band) for _ in range(N_PERM)]
    nulls2 = np.array([x for x in nulls2 if x is not None and np.isfinite(x)])
    out["H2_predpace"] = {"obs_pace_ratio": obs2,
                          "p_perm_greater": float((nulls2 >= obs2).mean())}
    # H4 significant-decelerator fractions (pooled studentized certificate)
    cal = band < 2
    stud = (M - age) / sig
    q = float(np.quantile(stud[cal], np.ceil((cal.sum() + 1) * 0.9) / (cal.sum() + 1)))
    cert = (M + q * sig) < age
    fracs = {f"band{b}": float(cert[band == b].mean()) for b in [0, 1, 2]}
    out["H4"] = {"alpha": 0.1, "q": q, "band_fracs": fracs,
                 "monotone_Y_M_L": bool(fracs["band0"] <= fracs["band1"] <= fracs["band2"]),
                 "sig_decelerator_frac_L": fracs["band2"],
                 "L_enrich_pooled": float((cert[band == 2].sum() / (band == 2).sum()) /
                                          (cert[cal].mean()) if cert[cal].mean() else None)}
    # H3 disclosure (direction): decile-matched L/Y 5-seed sigma ratio
    resid = np.abs(M - age)
    dec = np.clip((np.argsort(np.argsort(resid)) / len(resid) * 10).astype(int), 0, 9)
    ratios = []
    for d in range(10):
        mL = (band == 2) & (dec == d); mY = (band == 0) & (dec == d)
        if mL.sum() >= 3 and mY.sum() >= 3:
            ratios.append(sig[mL].mean() / sig[mY].mean())
    out["H3_disclosure"] = {"nr_L_over_Y_decile": float(np.mean(ratios)),
                            "interpretation": ">1 = tail uncertainty INFLATION (clock miscalibration at tail), NOT noise homeostasis"}
    json.dump(out, open(os.path.join(RUNS, "headlines_full.json"), "w"), indent=1, default=float)
    for k in ["oof", "H1", "H2_predpace", "H4", "H3_disclosure"]:
        print(f"[headlines-full] {k}: {json.dumps(out[k], default=float)}", flush=True)
    print("[headlines-full] WROTE headlines_full.json", flush=True)

if __name__ == "__main__":
    main()
