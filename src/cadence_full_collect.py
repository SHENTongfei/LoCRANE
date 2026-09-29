# -*- coding: utf-8 -*-
"""cadence_full_collect.py (v3) - full-cohort CADENCE-S certification.

IDs: the 25 npz files store ids as dtype U16 which TRUNCATES the 38
>16-char external ids (e.g. B35-2_FKDL220012593-1a.sorted.bam -> B35-2_FKDL220012).
Do NOT trust npz ids. Rebuild the authoritative per-fold val-id list from
folds_v3_full.json in the SAME sorted(str()) order cg_train used
(lab.loc[sorted(str(i) for i in folds[f])]), and assert the truncated npz
ids equal the 16-char prefix of that list (proves row alignment). Each
sample belongs to exactly one val fold -> 5 seed-preds each.
Collapse to per-sample 5-seed ensemble (mean M, cross-seed std sigma,
authoritative label age), then per-sex studentized one-sided split-conformal
certificates + L-band enrichment + F-vs-M two-proportion z.
Writes runs_v3/cadence_sex_full.json.
"""
import os, json, glob
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")

lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))

# authoritative per-fold val ids, exact row order cg_train used
fold_true_ids = {f: sorted(str(i) for i in folds[str(f)]) for f in range(5)}
all_true = []
for f in range(5):
    all_true += fold_true_ids[f]
assert len(all_true) == 1715 and len(set(all_true)) == 1715

acc = defaultdict(list)
age_by_id = {}
for f in range(5):
    true_ids = fold_true_ids[f]
    for seed in [42, 7, 2024, 2025, 12345]:
        npz = os.path.join(RUNS, f"cg_full_M_f{f}_s{seed}.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        got_ids = [str(x) for x in z["ids"]]
        # GATE: truncated npz ids must equal 16-char prefix of true ids (row alignment)
        if got_ids != [t[:16] for t in true_ids]:
            raise AssertionError(f"row misalignment fold{f} s{seed}: npz ids diverge from folds-file order")
        M = z["M"].astype(np.float64); age = z["age"].astype(np.float64)
        assert len(got_ids) == len(true_ids)
        for k, tid in enumerate(true_ids):
            acc[tid].append(float(M[k]))
            age_by_id[tid] = float(age[k])

ids = all_true
missing = [i for i in ids if len(acc[i]) != 5]
assert not missing, f"{len(missing)} samples without 5 seed-preds, e.g. {missing[:3]}"
M = np.array([np.mean(acc[i]) for i in ids])        # 5-seed ensemble
age = np.array([age_by_id[i] for i in ids])
sigma = np.array([np.std(acc[i]) + 1e-6 for i in ids])  # cross-seed uncertainty
# authoritative labels
sex = lab.loc[ids, "sex"].to_numpy()
age_lab = lab.loc[ids, "age"].to_numpy(np.float64)
assert np.allclose(age, age_lab), "npz val-age disagrees with labels"

isF = np.array([str(s).upper().startswith("F") for s in sex])
band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))

out = {"n": int(len(ids)), "nL": int((band == 2).sum()), "nM": int((band == 1).sum()),
       "nY": int((band == 0).sum()), "alpha": 0.1, "ensemble": "5-seed-mean",
       "sigma": "cross-seed-std", "id_source": "folds_v3_full.json (npz U16 truncated - avoided)"}

def sex_cert(mask, key):
    cal = mask & (band < 2)
    stud = (M - age) / sigma
    n = int(cal.sum())
    q = float(np.quantile(stud[cal], np.ceil((n + 1) * 0.9) / (n + 1)))
    pi = M + q * sigma
    cert = pi < age
    L = mask & (band == 2)
    kL = int(cert[L].sum()); nL = int(L.sum())
    marg = float(cert[cal].mean())
    frac = float(cert[L].mean()) if nL else None
    bt = binomtest(kL, nL, marg, alternative="greater") if nL else None
    out[key] = {"n": int(mask.sum()), "nL": nL, "q": q,
                "coverage": {f"band{b}": float((age[mask & (band == b)] <= pi[mask & (band == b)]).mean())
                              for b in [0, 1, 2] if (mask & (band == b)).sum() >= 5},
                "certified_frac": {f"band{b}": float(cert[mask & (band == b)].mean())
                                    for b in [0, 1, 2] if (mask & (band == b)).sum() >= 5},
                "L_enrichment": {"rate": frac, "marginal": marg,
                                  "fold": (frac / marg) if (marg and frac is not None) else None,
                                  "binom_p_greater": float(bt.pvalue) if bt else None}}

sex_cert(isF, "F")
sex_cert(~isF, "M")
dF, dM = out["F"], out["M"]
if dF["nL"] and dM["nL"]:
    pF = dF["L_enrichment"]["rate"]; pM = dM["L_enrichment"]["rate"]
    nF, nM = dF["nL"], dM["nL"]
    ppool = (pF * nF + pM * nM) / (nF + nM)
    se = np.sqrt(max(ppool * (1 - ppool) * (1 / nF + 1 / nM), 1e-12))
    z = (pF - pM) / se
    out["F_vs_M"] = {"pF": pF, "pM": pM, "nF": nF, "nM": nM,
                     "z": float(z), "p_two": float(2 * (1 - norm.cdf(abs(z))))}
json.dump(out, open(os.path.join(RUNS, "cadence_sex_full.json"), "w"), indent=1, default=float)
print("[cadence-full-v3] L-enrichment F:", json.dumps(dF["L_enrichment"]), flush=True)
print("[cadence-full-v3] L-enrichment M:", json.dumps(dM["L_enrichment"]), flush=True)
print("[cadence-full-v3] F_vs_M:", json.dumps(out.get("F_vs_M")), flush=True)
print("[cadence-full-v3] WROTE cadence_sex_full.json")
