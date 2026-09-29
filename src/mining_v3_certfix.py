# -*- coding: utf-8 -*-
"""mining_v3_certfix.py - recompute L-band certified flags with the POOLED q
(117.88, same as cadence_sex_full / headlines_full) instead of per-fold q
(133.14, what mining_v3.py used) -> consistent M3 certified-vs-uncertified
contrast. Reuses w2-free OOF predictions from mining_v3_results pipeline:
rebuilds 1715 per-sample 5-seed preds via the 25 ckpts (CPU+GPU-light,
forward only, ~2 min) and writes runs_v3/mining_v3_certfix.json with
L certified n (pooled-q caliber) + top-10 gene GIP contrast cert vs uncert
(reuses per-band GIP from mining_v3_M2_M3_genes.csv is NOT sample-level, so
recompute per-sample GIP on L subset only: 25 ckpts x L val samples x 3000,
batched, ~5 min GPU-light). Writes results + appends to mining_v3_results.json.
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
from sklearn.linear_model import Ridge
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 2024, 2025, 12345]

def log(m): print(f"[certfix] {m}", flush=True)

def build_model(n_immune, n_genes, assignment):
    return CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                    n_immune=n_immune, use_moe=False, use_gate=False,
                    use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)

def main():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    all_ids = [str(i) for i in pd.read_csv(os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
                                           sep="\t", index_col=0, compression=None).T.index]
    pos_by_id = {i: k for k, i in enumerate([str(j) for j in lab.index])}
    oof_p = np.zeros(1715); oof_s = np.zeros(1715); oof_age = np.zeros(1715); oof_band = np.zeros(1715, int)
    gene_gip_L_cert = np.zeros(len(cfg_genes)); gene_gip_L_unc = np.zeros(len(cfg_genes))
    n_c = n_u = 0
    for f in range(5):
        km = np.load(os.path.join(DATA3, "km256_f0.npz"))
        assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
        z = np.load(os.path.join(DATA3, f"fe_cache_f{f}_full.npz"))
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        lab_va = lab.loc[sorted(str(i) for i in folds[str(f)])]
        age = lab_va["age"].to_numpy(np.float32)
        band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
        val_ids = [str(x) for x in lab_va.index]
        inner = [i for i in all_ids if i not in set(val_ids)]
        Xq_tr, Xs_tr = z["Xq_tr"], z["Xs_tr"]
        age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
        rda = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - lab["age"].mean()) / lab["age"].std()
        mu, sd = lab["age"].mean(), lab["age"].std()
        q_t = torch.from_numpy(Xq_va).float()
        s_t = torch.from_numpy(Xs_va).float()
        imm_t = torch.from_numpy(imm_va.astype(np.float32))
        sex_t = torch.from_numpy(C.encode_sex(lab_va["sex"].to_numpy()))
        rda_t = torch.from_numpy(rda).float()
        BV = torch.zeros(len(Xq_va), 3, device=DEV)
        preds = []
        gip_L = None
        for s in SEEDS:
            ck = os.path.join(RUNS, "ckpt_cg", f"cg_k_f{f}_s{s}.pt")
            d = torch.load(ck, map_location=DEV, weights_only=False)
            model = build_model(imm_va.shape[1], len(cfg_genes), assign)
            model.load_state_dict(d["model"]); model.eval()
            Xq0 = q_t.clone().requires_grad_(True)
            with torch.enable_grad():
                out = model(Xq0.to(DEV), s_t.to(DEV), sex_t.to(DEV), imm_t.to(DEV),
                            rda_t.to(DEV), BV.to(DEV), plm=None)
            model.zero_grad()
            out["age"].sum().backward()
            P = out["age"].detach().cpu().numpy() * sd + mu
            preds.append(P)
            g = (Xq0.grad.abs() * q_t.abs()).numpy()
            gip_L = g[band == 2] if gip_L is None else gip_L + g[band == 2]
            del model
        P_ens = np.mean(preds, 0); sig = np.std(preds, 0) + 1e-6
        idx = np.array([pos_by_id[i] for i in val_ids])
        oof_p[idx] = P_ens; oof_s[idx] = sig; oof_age[idx] = age; oof_band[idx] = band
        gL = gip_L / 5.0
        if not hasattr(main, "_st"): main._st = []
        main._st.append((idx, band, gL))
    # pooled q
    cal = oof_band < 2
    stud = (oof_p - oof_age) / oof_s
    q = float(np.quantile(stud[cal], np.ceil((cal.sum() + 1) * 0.9) / (cal.sum() + 1)))
    cert = (oof_p + q * oof_s) < oof_age
    Lcert_mask = cert & (oof_band == 2)
    # split L GIP cert vs uncert
    gc = np.zeros_like(gene_gip_L_cert); gu = np.zeros_like(gene_gip_L_unc)
    for idx, band, gL in main._st:
        locL = np.where(band == 2)[0]
        cm = Lcert_mask[idx[locL]]; um = ~cm
        gc += gL[cm].sum(0) if cm.sum() else 0
        gu += gL[um].sum(0) if um.sum() else 0
    nc = int(Lcert_mask.sum()); nu = int((~Lcert_mask & (oof_band == 2)).sum())
    zcu = (gc / max(nc, 1)) - (gu / max(nu, 1))
    top = np.argsort(zcu)[::-1][:15]
    res = {"q_pooled": q, "n_L_cert": nc, "n_L_uncert": nu,
           "cert_frac_L_pooled": float(Lcert_mask.mean()),
           "top15_GIP_cert_minus_uncert": [str(cfg_genes[i]) for i in top],
           "z_top": [float(zcu[i]) for i in top]}
    json.dump(res, open(os.path.join(RUNS, "mining_v3_certfix.json"), "w"), indent=1, default=float)
    log(f"WROTE mining_v3_certfix.json: q={q:.2f} n_L_cert={nc} (vs per-fold {n_c=}) top={res['top15_GIP_cert_minus_uncert'][:8]}")

if __name__ == "__main__":
    main()
