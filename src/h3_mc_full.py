# -*- coding: utf-8 -*-
"""h3_mc_full.py - complete the pre-registered H3 menu: MC-dropout K=20
noise-ratio on the FULL cohort (1715), using the Phase-2 cg_k ckpts.
Per sample: mc_sigma = mean over the 5 seed-ckpts of its val fold of
std over K=20 MC-dropout forward passes (p=0.1 on transformer FFN+heads,
eval-mode single-head dropout). Then the pre-registered H3 formula:
  nr = sigma_L(decile-matched) / sigma_Y(decile-matched)
Pass = nr <= 0.85 and perm p < 0.01 (disclosure-grade; H3 already failed
on the 5-seed variant at nr=1.50 p=0.9995 - this closes the menu row).
Runs GPU-light. Writes runs_v3/h3_mc_full_results.json.
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
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 2024, 2025, 12345]
K = 20
RNG = np.random.default_rng(20260931)

def log(m): print(f"[h3mcf] {m}", flush=True)

def enable_mcdropout(model, p=0.1):
    # single-head dropout: transformer attention/value dropout to eval-deterministic
    # but keep one stochastic path alive on FFN dropouts
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
            m.inference = True      # force stochastic in eval
    return model

def build_model(n_immune, n_genes, assignment):
    m = CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                 n_immune=n_immune, use_moe=False, use_gate=False,
                 use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)
    return m

def main():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    all_ids = [str(i) for i in pd.read_csv(
        os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
        sep="\t", index_col=0, compression=None).T.index]
    mc_sigma = np.zeros(len(lab))
    for f in range(5):
        km = np.load(os.path.join(DATA3, f"km256_f{os.environ.get('V3_KMFOLD', '0')}.npz"))
        assign = {int(k_): int(v) for k_, v in zip(km["g"], km["m"])}
        z = np.load(os.path.join(DATA3, f"fe_cache_f{f}_full.npz"))
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        lab_va = lab.loc[sorted(str(i) for i in folds[str(f)])]
        age_va = lab_va["age"].to_numpy(np.float32)
        val_set = set(lab_va.index)
        inner = [i for i in all_ids if i not in val_set]
        Xq_tr, Xs_tr, imm_tr = z["Xq_tr"], z["Xs_tr"], z["imm_tr"]
        age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
        from sklearn.linear_model import Ridge
        rda_va = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - lab["age"].mean()) / lab["age"].std()
        BV = torch.zeros(len(Xq_va), 3, device=DEV)
        XVq = torch.from_numpy(Xq_va).float().to(DEV); XVs = torch.from_numpy(Xs_va).float().to(DEV)
        SV = torch.from_numpy(C.encode_sex(lab_va["sex"].to_numpy())).to(DEV)
        IV = torch.from_numpy(imm_va.astype(np.float32)).to(DEV)
        RV = torch.from_numpy(rda_va).float().to(DEV)
        per_seed = []
        for s in SEEDS:
            ck = os.path.join(RUNS, "ckpt_cg", f"cg_k_f{f}_s{s}.pt")
            d = torch.load(ck, map_location=DEV, weights_only=False)
            model = enable_mcdropout(build_model(imm_va.shape[1], len(cfg_genes), assign), 0.1)
            model.load_state_dict(d["model"])
            with torch.no_grad():
                draws = [float(model(XVq, XVs, SV, IV, RV, BV, plm=None)["age"].std())
                         for _ in range(K)]
            per_seed.append(float(np.mean(draws)))
            del model
        row = np.mean(per_seed)
        idx = lab.index.get_indexer(lab_va.index)
        mc_sigma[idx] = row
        log(f"fold{f}: K={K} mc-draws/seed done, mean mc_row={row:.3f}")
    age = lab["age"].to_numpy(np.float64)
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    resid = np.abs(age - np.mean(age))  # placeholder-free decile on |age-70|
    resid = np.abs(age - 70.0)
    dec = np.clip((np.argsort(np.argsort(resid)) / len(resid) * 10).astype(int), 0, 9)
    def nr():
        ratios = []
        for d_ in range(10):
            mL = (band == 2) & (dec == d_); mY = (band == 0) & (dec == d_)
            if mL.sum() >= 3 and mY.sum() >= 3:
                ratios.append(mc_sigma[mL].mean() / mc_sigma[mY].mean())
        return float(np.mean(ratios)) if ratios else None
    obs = nr()
    nulls = []
    pool = (band == 2) | (band == 0)
    b2 = band.copy()
    for _ in range(2000):
        b2[pool] = RNG.permutation(b2[pool])
        r = nr()
        if r is not None:
            nulls.append(r)
    out = {"n": int(len(age)), "K": K, "nr_L_over_Y": obs,
           "null_mean": float(np.mean(nulls)), "p_perm_less": float((np.array(nulls) <= obs).mean()),
           "gate": "nr<=0.85 & p<0.01 (pre-registered H3 menu, MC-dropout K=20 row)"}
    json.dump(out, open(os.path.join(RUNS, "h3_mc_full_results.json"), "w"), indent=1, default=float)
    log(f"WROTE h3_mc_full_results.json: {json.dumps(out, default=float)}")

if __name__ == "__main__":
    main()
