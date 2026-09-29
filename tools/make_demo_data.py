# -*- coding: utf-8 -*-
"""make_demo_data.py - regenerate data/demo_bundle.npz from the FULL local data.

The shipped demo_bundle.npz is a deterministic, stratified 120-sample slice of the
frozen fold-0 feature representation of the full 1715-sample cohort. This script
rebuilds it byte-comparably (same seed) from:

  <repo>/data_v3/{labels_v3.csv, folds_v3_full.json, km256_f0.npz}
  <CRANE_DATA_DIR>/expression_matrix.tsv.gz-adjacent feature cache:
      fe_cache_f0_full.npz  (produced by src/train_v3.py on first full run,
      keyed to fold 0 of folds_v3_full.json)

Usage (after the full-data Stage A/B feature cache exists):
    python tools/make_demo_data.py --fe_cache <path to fe_cache_f0_full.npz>

Nothing here fabricates values: every row is copied verbatim from the frozen
representation; the script only selects rows and records provenance.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_TR, SEED_VA = 42, 43


def strat(pool_ids, band_of, nY, nM, nL, seed):
    rng = np.random.RandomState(seed)
    picks = []
    for n, b in [(nY, 0), (nM, 1), (nL, 2)]:
        cand = [i for i in pool_ids if band_of[i] == b]
        picks += list(rng.choice(sorted(cand), size=min(n, len(cand)), replace=False))
    return sorted(picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe_cache", required=True, help="path to fe_cache_f0_full.npz")
    a = ap.parse_args()

    lab = pd.read_csv(os.path.join(HERE, "data_v3", "labels_v3.csv"))
    lab["ID"] = lab["ID"].astype(str)
    lab = lab.set_index("ID")
    folds = json.load(open(os.path.join(HERE, "data_v3", "folds_v3_full.json")))
    va_ids = sorted(folds["0"])
    va_set = set(va_ids)
    inner_ids = [i for i in lab.index if i not in va_set]
    assert len(inner_ids) == 1370 and len(va_ids) == 345, (len(inner_ids), len(va_ids))

    z = np.load(a.fe_cache)
    km = np.load(os.path.join(HERE, "data_v3", "km256_f0.npz"))
    band_code = {"Y": 0, "M": 1, "L": 2}
    bcode = lab["band"].map(band_code)

    tr_ids = strat(inner_ids, bcode, 10, 41, 45, SEED_TR)      # 96, band-proportional
    va_ids_sub = strat(va_ids, bcode, 3, 10, 11, SEED_VA)      # 24, band-proportional
    tidx = [inner_ids.index(i) for i in tr_ids]
    vidx = [va_ids.index(i) for i in va_ids_sub]

    Xq = np.concatenate([z["Xq_tr"][tidx], z["Xq_va"][vidx]], 0).astype(np.float32)
    Xs = np.concatenate([z["Xs_tr"][tidx], z["Xs_va"][vidx]], 0).astype(np.float32)
    imm = np.concatenate([z["imm_tr"][tidx], z["imm_va"][vidx]], 0).astype(np.float32)
    ids = tr_ids + va_ids_sub
    sub = lab.loc[ids]
    age = sub["age"].to_numpy(np.float32)
    sex01 = (sub["sex"].astype(str).str.upper().str[0] == "F").astype(np.int64).to_numpy()
    band = bcode.loc[ids].to_numpy(np.int64)
    is_val = np.array([0] * 96 + [1] * 24, dtype=np.int64)

    np.savez_compressed(os.path.join(HERE, "data", "demo_bundle.npz"),
                        Xq=Xq, Xs=Xs, imm=imm, age=age, sex01=sex01, band=band,
                        ids=np.array(ids), is_val=is_val,
                        km_g=km["g"].astype(np.int64), km_m=km["m"].astype(np.int64),
                        n_train=np.int64(96))
    sub.reset_index()[["ID", "age", "sex", "region", "band"]].to_csv(
        os.path.join(HERE, "data", "demo_labels.csv"), index=False)
    print("demo bundle rebuilt:", Xq.shape, {b: int((band == v).sum())
                                             for b, v in {"Y": 0, "M": 1, "L": 2}.items()})


if __name__ == "__main__":
    main()
