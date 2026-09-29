# -*- coding: utf-8 -*-
"""w2_external_fe.py - build v3 feature cache for the two external cohorts
(cohort2 GSE123696 Israel chip, 66 samples; GTEx balanced, 64 samples) using
the ZERO-SHOT external protocol (v2/GTEx-aligned): scalers + ridge fit on
fold0-train reference, then transform external gene matrices. Outputs
runs_v3/w2_ext_fe_<cohort>.npz with Xq/Xs/imm/rda/age/sex/band/group/ids.
CPU-only. B3 stage0 model uses n_plm=0, so no PLM needed.
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
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import Ridge
import common as C

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
EXT = r"C:/Users/TS/Desktop/crane/external_test_data"

def log(m): print(f"[w2fe] {m}", flush=True)

def build(name, csv, meta_cols):
    ext = pd.read_csv(os.path.join(EXT, csv), index_col=0)
    meta = [c for c in meta_cols if c in ext.columns]
    expr = ext.drop(columns=meta)
    age = ext["age_mid"].to_numpy(np.float32) if "age_mid" in ext.columns else None
    sex = ext["sex"].to_numpy() if "sex" in ext.columns else None
    grp = ext["group"].to_numpy() if "group" in ext.columns else None

    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    qsc = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    zsc = StandardScaler()
    X0a = C.align_features(p_all, cfg_genes)
    qsc.fit(X0a); zsc.fit(X0a)

    X = C.align_features(expr, cfg_genes)
    Xq, Xs = qsc.transform(X), zsc.transform(X)

    lm22 = pd.read_csv(C.LM22_PATH, sep="\t", index_col=0)
    qc = json.load(open(os.path.join(DATA3, "lm22_qc.json")))
    lm22q = lm22.loc[[g for g in lm22.index if g in p_all.columns], qc["keep_cells"]]
    imm = C.lm22_deconvolution(expr, lm22q)
    imm = imm[qc["keep_cells"]].to_numpy(np.float32)

    p0 = C.load_fold(0)[2]
    age_tr = p0["age"].to_numpy(np.float32)
    mu, sd = float(age_tr.mean()), max(1e-6, float(age_tr.std()))
    Xq_tr = qsc.transform(C.align_features(p_all, cfg_genes))
    rda = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq) - mu) / sd

    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    sex_enc = C.encode_sex(sex) if sex is not None else np.zeros(len(age), dtype=np.int64)
    ids = [str(i) for i in expr.index]
    out = os.path.join(RUNS, f"w2_ext_fe_{name}.npz")
    np.savez(out, Xq=Xq, Xs=Xs, imm=imm, rda=rda.astype(np.float32),
             age=age, sex=sex_enc, band=band,
             group=(grp.astype("U16") if grp is not None else np.full(len(ids), "?", "U16")),
             ids=np.array(ids, dtype="U40"))
    log(f"{name}: {expr.shape[0]} samples x {len(cfg_genes)} genes | Xq std={Xq.std(0).mean():.2f} "
        f"imm {imm.shape} rda {rda.shape} band counts {dict(zip(*np.unique(band, return_counts=True)))}")
    return out

if __name__ == "__main__":
    build("cohort2", "cohort2/cohort2_validation_blood.csv", ["age_mid", "group", "sex"])
    build("gtex", "gtex_validation_balanced.csv", ["age_mid", "group", "sex"])
    log("DONE")
