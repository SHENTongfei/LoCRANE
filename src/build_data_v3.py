# -*- coding: utf-8 -*-
"""build_data_v3.py - TAIL-CERT data chain v3 (B1). Read-only wrt legacy files;
writes everything under data_v3/. No model training here.

Outputs (model_mining/data_v3/):
  labels_v3.csv          ID,age,sex,region,band,w_band,w_agebucket  (all 1715)
  folds_v3.json          outer folds: val IDs per fold (reuse balanced_folds vals as OOF)
  lm22_qc.json           constant-cell list (SD<1e-4 on fold0 train) to drop
  plm_map.csv            HVG symbol -> ENSG -> ENSP (GENCODE v26 GTF) + coverage
  plm_esm2_650M_hvg.npz  sliced embeddings [n_hvg_matched, 1280] (+ ids)
  protocol_lock_v4.json  pre-registered protocol (labels/losses/metrics/gates)
"""
import os, sys, json, gzip, hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # model_mining/
OUT = os.path.join(ROOT, "data_v3")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("CRANE_DATA_DIR", "C:/Users/TS/Desktop/crane")
sys.path.insert(0, os.path.dirname(ROOT))
import common as C

BASE = os.environ["CRANE_DATA_DIR"]
report = {}

# ---------------------------------------------------------------- 1. labels v3
ph = pd.read_csv(os.path.join(BASE, "phenotype.csv"), index_col=0)
age = ph["age"].astype(float)
band = np.where(age.values >= 90, "L", np.where(age.values > 50, "M", "Y"))
lab = pd.DataFrame({"ID": ph.index.astype(str), "age": age.values,
                    "sex": ph["sex"].astype(str).str.upper().values,
                    "region": ph["region"].values, "band": band})
# imbalance plan: inverse-frequency band weights (mean 1) + 10y age-bucket weights (mean 1)
cnt = lab["band"].value_counts()
lab["w_band"] = lab["band"].map(1.0 / cnt).values
lab["w_band"] /= lab["w_band"].mean()
bk = (lab["age"] // 10).astype(int)
bkc = bk.value_counts()
lab["w_agebucket"] = bk.map(1.0 / bkc).values
lab["w_agebucket"] /= lab["w_agebucket"].mean()
lab.to_csv(os.path.join(OUT, "labels_v3.csv"), index=False)
report["labels_v3"] = {"n": int(len(lab)),
                       "band_counts": cnt.to_dict(),
                       "w_band": {k: round(float(v), 4) for k, v in
                                  lab.groupby("band")["w_band"].first().items()},
                       "sha256": hashlib.sha256(open(os.path.join(BASE, "phenotype.csv"), "rb")
                                                .read()).hexdigest()[:16]}

# ---------------------------------------------------------------- 2. outer folds (OOF reuse)
folds = {}
for f in range(5):
    p_va = C.load_fold(f)[3]
    folds[str(f)] = sorted(map(str, p_va.index))
json.dump(folds, open(os.path.join(OUT, "folds_v3.json"), "w"))
union = sorted({i for v in folds.values() for i in v})
report["folds_v3"] = {"outer_val_union_n": len(union),
                      "per_fold": {k: len(v) for k, v in folds.items()}}

# ---------------------------------------------------------------- 3. LM22 QC
x0 = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
hvg = list(x0.columns)
lm22 = pd.read_csv(C.LM22_PATH, sep="\t", index_col=0)
fr = C.lm22_deconvolution(x0, lm22)
const = [c for c, v in fr.std(axis=0).items() if v < 1e-4]
json.dump({"constant_cells_sd_lt_1e-4": const, "keep_cells": [c for c in fr.columns if c not in const]},
          open(os.path.join(OUT, "lm22_qc.json"), "w"), indent=1)
report["lm22_qc"] = {"dropped": len(const), "kept": 22 - len(const)}

# ---------------------------------------------------------------- 4. PLM symbol->ENSP map + slice
# GENCODE v26 GTF: gene_name / gene_id(ENSG) / protein_id(ENSP on CDS lines)
gtf = os.path.join(BASE, "external_test_data", "gencode.v26.annotation.gtf.gz")
sym2ensp = {}
with gzip.open(gtf, "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[2] != "CDS":
            continue
        attrs = dict(kv.strip().split(" ", 1) for kv in parts[8].strip().split(";") if kv.strip())
        gname = attrs.get("gene_name", "").strip('"')
        pid = attrs.get("protein_id", "").strip('"')
        if gname and pid and gname not in sym2ensp:
            sym2ensp[gname] = pid.split(".")[0]
meta = json.load(open(r"C:/Users/TS/WorkBuddy/PIACE/data/embeddings/human/esm2_650M/meta.json"))
ids = meta["ids"]; pos = {g: i for i, g in enumerate(ids)}
rows, matched = [], 0
for g in hvg:
    p = sym2ensp.get(g)
    rows.append({"symbol": g, "ENSP": p or "", "in_tower": bool(p and p in pos)})
    matched += bool(p and p in pos)
mp = pd.DataFrame(rows)
mp.to_csv(os.path.join(OUT, "plm_map.csv"), index=False)
z = np.load(r"C:/Users/TS/WorkBuddy/PIACE/data/embeddings/human/esm2_650M/embeddings.npz")
ZARR = z[z.files[0]]
sel = [sym2ensp.get(g) for g, ok in zip(mp["symbol"], mp["in_tower"]) if ok]
X = ZARR[np.array([pos[s] for s in sel])]
np.savez_compressed(os.path.join(OUT, "plm_esm2_650M_hvg.npz"),
                    ids=np.array(sel), X=X.astype(np.float32))
report["plm_slice"] = {"hvg_total": len(hvg), "matched_enp": matched,
                       "coverage": round(matched / len(hvg), 4), "dim": int(X.shape[1])}

# ---------------------------------------------------------------- 5. protocol_lock v4
lock = {
 "version": "TAIL-CERT v4.0", "date": "2026-09-26",
 "labels": {"bands": "Y<=50 / M51-89 / L>=90", "evidence": "sebastiani2012+deelen2019+labels_v1>=90+D1a-LLI",
            "weights": "band inverse-frequency + 10y age-bucket inverse-frequency (regression)",
            "balanced_subsampling_162_162": "RETIRED"},
 "cv": {"outer": "324-pool 5-fold OOF (reuse balanced_folds vals)", "inner_train": "1715 minus outer-val",
        "seeds": [42, 7, 2024, 2025, 12345], "nested": "H8"},
 "alpha": {"fused_long_prob_alpha": 1.0, "note": "O-1 closed: legacy ensemble.csv=mixed-alpha artifact"},
 "loss": {"age": "SmoothL1(z) x age-bucket weights",
          "band": "ordinal cumulative logits + label smoothing 0.1 x band weights",
          "bandpos": "SmoothL1, L-band dilated span 1.4x (Tv screen: 1.4 vs 1.0)",
          "gate": "L1 on modality gate; MoE load-balance aux",
          "screen_arms": ["focal=NO (DSR pit)", "pairwise-rank age term (arm)", "triple-view contrastive (arm)"]},
 "metrics_locked": ["pace_ratio_90p_over_70_89", "sig_decel_fraction(band; one-sided conformal p<0.10, per-cohort q)",
                    "tail_gap_molecular_years", "ordinal_ECE", "RMAE", "AUC_L_vs_Y", "decay_pct", "CIQ<=0.3"],
 "family": {"primary": ["H1_tail_gap", "H2_pace_ratio", "H3_noise_v2", "H4_sig_decel", "H8_transfer"],
            "secondary": ["H5_activity_vs_composition", "H6_sex_routes", "H7_wiring"], "correction": "BH"},
 "externals": {"ext1": "D-1a China 811 LLI+940 YC (GSE pending)", "ext2": "cohort2 GSE123696 n=66",
               "reference_only": "GTEx blood/muscle", "rule": "externals evaluated once after lock; never in pretrain pool"},
 "placeholders": {"H3_estimator": "predicted-age-decile matched + heteroscedastic GLS (v2)",
                  "ssgsea_channel": "W1-pending (reuse p5a pipeline)", "plm_towers": "esm2_650M sliced; esm_c/prot_t5_xl pending W1"},
}
json.dump(lock, open(os.path.join(OUT, "protocol_lock_v4.json"), "w"), indent=1, default=str)
report["protocol_lock"] = "written"

json.dump(report, open(os.path.join(OUT, "build_report.json"), "w"), indent=1, default=str)
print(json.dumps(report, indent=1, default=str))
print("[saved]", OUT)
