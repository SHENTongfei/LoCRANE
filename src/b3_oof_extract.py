# -*- coding: utf-8 -*-
"""b3_oof_extract.py - OOF measurement array on the 25 b3tail ckpts (H1/H2/H3/H4 + top-2 seed ensemble).
Single process, OMP=2, GPU inference. Writes runs_v3/b3_oof_report.json."""
import os, sys, json, time
os.environ.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "CRANE_DATA_DIR": "C:/Users/TS/Desktop/crane"})
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda"
SEEDS = [42, 7, 2024, 2025, 12345]
CKPT = os.environ.get("OOF_CKPT", "tailft_b3tail")   # arm_tag prefix
OTAG = os.environ.get("OOF_TAG", "b3tail")           # output suffix

def log(m): print(f"[oof] {m}", flush=True)

def build_model(n_immune, n_genes, assignment):
    return CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                    n_immune=n_immune, use_moe=False, use_gate=False,
                    use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)

def main():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3.json")))
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    km = np.load(os.path.join(DATA3, "km256_f0.npz"))
    assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
    p0 = C.load_fold(0)[2]
    mu, sd = float(p0["age"].mean()), max(1e-6, float(p0["age"].std()))

    # global OOF arrays (324 pool)
    val_ids_all = []
    M_mean_all, M_seeds_all = [], []
    top2_all = []
    per_fold_rows = []
    mc_sig_all = []
    all_ids = [str(i) for i in pd.read_csv(os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
                                           sep="\t", index_col=0, compression=None).T.index]
    for f in range(5):
        z = np.load(os.path.join(DATA3, f"fe_cache_f{f}.npz"))
        Xq_tr, Xs_tr, imm_tr = z["Xq_tr"], z["Xs_tr"], z["imm_tr"]
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        val_ids = [str(x) for x in folds[str(f)]]
        lab_va = lab.loc[val_ids]
        age_va = lab_va["age"].to_numpy(np.float32)
        b_va = np.where(age_va >= 90, 2, np.where(age_va > 50, 1, 0))
        sex_va = C.encode_sex(lab_va["sex"].to_numpy())
        # ridge anchor on this fold's inner train (same caliber as trainer)
        from sklearn.linear_model import Ridge
        val_set = set(val_ids)
        inner = [i for i in all_ids if i not in val_set]
        age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
        rda_va = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - mu) / sd
        BV = torch.from_numpy(np.eye(3, dtype=np.float32)[b_va]).to(DEV)
        XVq = torch.from_numpy(Xq_va).float().to(DEV); XVs = torch.from_numpy(Xs_va).float().to(DEV)
        SV = torch.from_numpy(sex_va).to(DEV); IV = torch.from_numpy(imm_va.astype(np.float32)).to(DEV)
        RV = torch.from_numpy(rda_va).float().to(DEV)
        per_seed_M = []
        mc_sig_seeds = []
        for s in SEEDS:
            ck = os.path.join(RUNS, "ckpt", f"{CKPT}_f{f}_s{s}_best.pt")
            if not os.path.exists(ck):
                log(f"WARN missing ckpt {os.path.basename(ck)}"); continue
            d = torch.load(ck, map_location=DEV, weights_only=False)
            model = build_model(imm_va.shape[1], len(cfg_genes), assign)
            model.load_state_dict(d["model"]); model.eval()
            with torch.no_grad():
                out = model(XVq, XVs, SV, IV, RV, BV, plm=None)
            M = out["age"].cpu().numpy() * sd + mu
            per_seed_M.append(M)
            if os.environ.get("OOF_MC", "1") == "1":  # H3 second estimator: MC-dropout K=20
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.train()
                outs = []
                with torch.no_grad():
                    for _ in range(20):
                        o = model(XVq, XVs, SV, IV, RV, BV, plm=None)
                        outs.append(o["age"].cpu().numpy() * sd + mu)
                mc_sig_seeds.append(np.std(outs, axis=0))
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.eval()
        per_seed_M = np.array(per_seed_M)            # (S, nval)
        mc_sig_f = np.mean(mc_sig_seeds, axis=0) if mc_sig_seeds else np.zeros(len(val_ids))
        val_ids_all += val_ids
        M_mean_all.append(per_seed_M.mean(0))
        M_seeds_all.append(per_seed_M)
        # top-2 seed ensemble: rank by per-seed val MAE
        maes = np.abs(per_seed_M - age_va[None, :]).mean(1)
        top2 = np.mean(per_seed_M[np.argsort(maes)[:2]], 0)
        top2_all.append(top2)
        per_fold_rows.append({"fold": f, "nval": len(val_ids), "top2_MAE": float(np.abs(top2 - age_va).mean())})
        mc_sig_all.append(mc_sig_f)
        del model
        log(f"fold{f}: seeds={len(per_seed_M)} top2 MAE={np.abs(top2-age_va).mean():.2f}")
    M_mean_all = np.concatenate(M_mean_all, axis=0)
    M_seeds_all = np.concatenate(M_seeds_all, axis=1)
    top2_all = np.concatenate(top2_all)
    chrono = lab.loc[val_ids_all, "age"].to_numpy(float)
    M = M_mean_all
    band = np.where(chrono >= 90, 2, np.where(chrono > 50, 1, 0))
    rep = compute_hypotheses(chrono, M, M_seeds_all, band)
    rep["top2_ensemble_MAE"] = float(np.abs(top2_all - chrono).mean())
    rep["plain_5seed_MAE"] = float(np.abs(M - chrono).mean())
    rep["per_seed_MAEs"] = [float(np.abs(M_seeds_all[k] - chrono).mean()) for k in range(M_seeds_all.shape[0])] if M_seeds_all.ndim == 3 else None
    rep["n_oof"] = len(chrono)
    with open(os.path.join(RUNS, f"b3_oof_report_{OTAG}.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    np.savez(os.path.join(RUNS, f"oof_predictions_{OTAG}.npz"),
             chrono=chrono, M=M, seeds=M_seeds_all, band=band,
             mc_sigma=np.concatenate(mc_sig_all, axis=0),
             top2=top2_all, ids=np.array(val_ids_all, dtype="U16"))
    log(f"WROTE b3_oof_report_{OTAG}.json + oof_predictions_{OTAG}.npz")
    print(json.dumps({k: rep.get(k) for k in ["H1_tail_gap", "H2_pace_ratio", "H3_noise_ratio",
                                               "H4_sig_decel_frac", "top2_ensemble_MAE", "plain_5seed_MAE"]}, indent=1))


def compute_hypotheses(chrono, M, M_seeds, band):
    res = {}
    resid = M - chrono
    hi = resid[band == 2]; lo = resid[band == 1]
    res["H1_tail_gap"] = float(np.median(hi) - np.median(lo))
    # H2 pace-ratio: |dM/dchrono| via pairwise within-band, per-year
    def velocity(b):
        m = band == b
        c = chrono[m]; mm = M[m]
        order = np.argsort(c)
        dc = np.abs(np.diff(c[order]))
        dm = np.abs(np.diff(mm[order]))
        good = dc > 0.5
        return float(dm[good].sum() / dc[good].sum())
    v90, v70 = velocity(2), velocity(1)
    res["H2_pace_ratio"] = float(v90 / v70) if v70 > 1e-6 else None
    # H3 noise-ratio: seed-sigma within predicted-age decile, L vs Y band
    def seed_sigma(b, dec=5):
        m = band == b
        qq = np.quantile(M[m], np.linspace(0, 1, dec + 1))
        per = []
        for i in range(dec):
            sel = m & (M >= qq[i]) & (M <= qq[i + 1])
            if sel.sum() >= 3:
                per.append(M_seeds[:,sel].std(0).mean())
        return float(np.mean(per)) if per else None
    sL, sY = seed_sigma(2), seed_sigma(0)
    res["H3_noise_ratio"] = float(sL / sY) if (sL and sY and sY > 1e-6) else None
    # H4 significant-decelerator fraction: split-conformal one-sided UPPER bound on M
    res["H4_sig_decel_frac"] = per_band_sigfrac(chrono, M, band)
    return res

def per_band_sigfrac(chrono, M, band, target_q=0.90):
    # fit conformal on younger calibration pool, apply to L band
    cal = band < 2
    resid = M[cal] - chrono[cal]
    q = np.quantile(resid, np.ceil((len(resid) + 1) * target_q) / (len(resid) + 1))
    out = {}
    for b in [0, 1, 2]:
        m = band == b
        # decelerated = predicted molecular age UPPER bound still below chronological
        sig = (M[m] + q) < chrono[m]
        out[f"frac_band{b}"] = float(sig.mean()) if m.sum() else None
    return out

if __name__ == "__main__":
    main()
