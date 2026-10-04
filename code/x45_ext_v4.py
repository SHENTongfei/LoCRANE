# -*- coding: utf-8 -*-
"""x14_ext_battle.py - NEW model (LoCRANE-Ens) vs all baselines on the two
independent external sets, clean rebuild (hvg8000 space identical to the locked
internal battle x12).

ext1 = GTEx blood balanced (n=64, RNA-seq, bracket-mid ages, RYC/RLL balanced)
ext2 = cohort2 GSE123696 Israeli blood (n=66, microarray, ages 23-96, Y/M/L)

Protocol:
- gene universe = hvg8000 ∩ ext1 genes ∩ ext2 genes (ONE list for both ext sets)
- top-40/top-500 by |age-corr| on internal YC train ONLY (760 samples)
- members trained on internal YC (all 760): EN-40, EN-500, SVR-500, XGB-500
- MT deep member x10 config (L-aux band CE, cross-fitted EN anchor, real sex, 5 seeds)
- external features: per-gene quantile harmonization onto INTERNAL scaled-train
  distribution (fit internal only), same matrix fed to every method
- metrics: MAE/PCC/SP + 5000-bootstrap 95% CI on MAE; per-band breakdown
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v4")
RUNS3 = os.path.join(HERE, "..", "runs_v3")
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
EXT = r"C:/Users/TS/Desktop/crane/external_test_data"
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEV={DEV}", flush=True)

# ---------------- internal ----------------
p_all = pd.read_csv(os.path.join(CRANE, "training_data", "balanced_folds", "fold_0", "train_expr.csv"), index_col=0)
v = p_all.var(axis=0).sort_values(ascending=False)
hvg8 = list(v.index[:8000])
expr = pd.read_csv(os.path.join(CRANE, "expression_matrix.tsv.gz"), sep="\t", index_col=0, compression=None)
expr_hvg = expr.loc[expr.index.isin(hvg8)]  # rows=genes(present), cols=samples
lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv"))
lab["ID"] = lab["ID"].astype(str); lab = lab.set_index("ID")
yc_lab = lab[(lab["age"] >= 50) & (lab["age"] < 90)]
l_lab = lab[lab["age"] >= 90]

# ---------------- external ----------------
def load_ext(csv_rel, npz_tag):
    df = pd.read_csv(os.path.join(EXT, csv_rel), index_col=0)
    z = np.load(os.path.join(RUNS3, f"w2_ext_fe_{npz_tag}.npz"), allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    df = df.loc[ids]
    genes = [c for c in df.columns if c not in ("group",)]
    return df[genes].astype(np.float32), z["age"].astype(float), (z["sex"].astype(np.float32)), z["band"].astype(int), z["group"].astype(str)

gt_df, gt_age, gt_sex, gt_band, gt_grp = load_ext("gtex_validation_balanced.csv", "gtex")
c2_df, c2_age, c2_sex, c2_band, c2_grp = load_ext(os.path.join("cohort2", "cohort2_validation_blood.csv"), "cohort2")

# gene universe: hvg8000 rank order ∩ internal ∩ BOTH ext sets
universe = [g for g in hvg8 if g in expr_hvg.index and g in gt_df.columns and g in c2_df.columns]
Xi_full = expr_hvg.loc[universe]  # genes(universe order) × samples
print(f"gene universe (hvg8000 ∩ ext1 ∩ ext2 ∩ internal): {len(universe)}", flush=True)

yc_ids = [s for s in yc_lab.index if s in Xi_full.columns]
l_ids = [s for s in l_lab.index if s in Xi_full.columns]
Xc = np.nan_to_num(Xi_full[yc_ids].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
XL = np.nan_to_num(Xi_full[l_ids].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
yc = yc_lab.loc[yc_ids]
ytr = yc["age"].to_numpy(float)
sextr = (yc["sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
yL = l_lab.loc[l_ids, "age"].to_numpy(float)
sexL = (l_lab.loc[l_ids, "sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
print(f"internal YC n={len(ytr)}, L n={len(yL)}", flush=True)

def topk_on_internal(k):
    ac = np.zeros(Xc.shape[1], dtype=np.float32)
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        if col.std() > 0:
            cc = np.corrcoef(col, ytr)[0, 1]
            ac[j] = abs(cc) if np.isfinite(cc) else 0.0
    order = np.argsort(-ac)
    return np.sort(order[:40]), np.sort(order[:500])

sel40, sel500 = topk_on_internal(500)
print("selected top40/top500 (positions in universe)", flush=True)

# scaled internal reference (StandardScaler fit on internal YC train)
sc40 = StandardScaler().fit(Xc[:, sel40]); sc500 = StandardScaler().fit(Xc[:, sel500])

def harmonize(ext_vals, ref_scaled):
    """per-gene quantile map: ext raw -> internal scaled-train distribution."""
    out = np.empty_like(ext_vals, dtype=np.float64)
    n_r = ref_scaled.shape[0]
    qpos = (np.arange(n_r) + 0.5) / n_r
    ref_sorted = np.sort(ref_scaled, axis=0)
    pos = (np.argsort(np.argsort(ext_vals, axis=0), axis=0) + 0.5) / ext_vals.shape[0]
    for j in range(ext_vals.shape[1]):
        out[:, j] = np.interp(pos[:, j], qpos, ref_sorted[:, j])
    return out.astype(np.float32)

def build_ext_matrix(df, sel, scaler):
    raw = df[[universe[j] for j in sel]].to_numpy(np.float32)
    ref = scaler.transform(Xc[:, sel])
    return harmonize(raw, ref)

def band_of(age_arr):
    return np.where(age_arr <= 50, 0, np.where(age_arr < 90, 1, 2)).astype(np.int64)

# ---------------- train members on internal ----------------
def make(kind):
    if kind == "EN":
        return ElasticNet(alpha=0.5, l1_ratio=0.5)
    if kind == "SVR":
        return SVR(C=10.0, epsilon=0.1, gamma="scale")
    return xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.4, random_state=42,
                            verbosity=0, n_jobs=8)

t0 = time.time()
mem = {}
Xs40 = sc40.transform(Xc[:, sel40])
Xs500 = sc500.transform(Xc[:, sel500])
for tag, Xs in (("EN-40", Xs40), ("EN-500", Xs500)):
    mem[tag] = make("EN").fit(Xs, ytr)
mem["SVR-500"] = make("SVR").fit(Xs500, ytr)
mem["XGB-500"] = make("XGB").fit(Xs500, ytr)
print(f"linear members trained ({time.time()-t0:.0f}s)", flush=True)

# MT deep member (x10 config)
def make_model(n_genes, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    n_mod = 64
    assign = {int(i): int(i * n_mod // n_genes) for i in range(n_genes)}
    return CraneZV3(n_genes=n_genes, assignment=assign, n_modules=n_mod, n_immune=0,
                    d_model=64, n_heads=4, n_layers=2, ff_dim=128,
                    ridge_residual=1.0).to(DEV)

def train_mt(Xtr, ytr_, sex_tr, Xlp, ylp, sex_lp, seed, epochs=150, patience=20):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    ntr = len(ytr_)
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr).astype(np.float32)
    kf = KFold(5, shuffle=True, random_state=seed)
    anchor_raw = np.zeros(ntr)
    for itr, ivl in kf.split(np.arange(ntr)):
        en_i = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[itr], ytr_[itr])
        anchor_raw[ivl] = en_i.predict(Xs[ivl])
    rg_full = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs, ytr_)
    mu, sd = float(ytr_.mean()), max(1e-6, float(ytr_.std()))
    anchor = ((anchor_raw - mu) / sd).astype(np.float32)
    model = make_model(Xtr.shape[1], seed)
    tq = torch.from_numpy(Xs).to(DEV)
    ty = torch.from_numpy(((ytr_ - mu) / sd).astype(np.float32)).to(DEV)
    ta = torch.from_numpy(anchor).to(DEV)
    tsx = torch.from_numpy(sex_tr).to(DEV)
    tband = torch.from_numpy(band_of(ytr_)).to(DEV)
    Xlq = torch.from_numpy(sc.transform(Xlp).astype(np.float32)).to(DEV)
    alq = torch.from_numpy(((rg_full.predict(sc.transform(Xlp)) - mu) / sd).astype(np.float32)).to(DEV)
    slq = torch.from_numpy(sex_lp).to(DEV)
    blq = torch.from_numpy(band_of(ylp)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    nl = len(ylp)
    best_loss, best_state, pat = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=DEV)
        for i in range(0, ntr, 128):
            idx = perm[i:i + 128]
            lidx = torch.randint(0, nl, (len(idx),), device=DEV)
            xq = torch.cat([tq[idx], Xlq[lidx]])
            sx = torch.cat([tsx[idx], slq[lidx]])
            rz = torch.cat([ta[idx], alq[lidx]])
            out = model(xq, xq, sx, ridge_z=rz)
            band_t = torch.cat([tband[idx], blq[lidx]])
            loss = F.l1_loss(out["age"][:len(idx)], ty[idx]) + 0.3 * F.cross_entropy(out["band_logits"], band_t)
            # V4-3 adopted monotonicity prior (RankNet age-order, lam=0.05)
            if len(idx) > 8:
                pi = torch.randint(0, len(idx), (len(idx),), device=DEV)
                pj = torch.randint(0, len(idx), (len(idx),), device=DEV)
                sgn = torch.sign(ty[idx][pi] - ty[idx][pj])
                mm = sgn != 0
                l_mon = F.softplus(-sgn[mm] * (out["age"][:len(idx)][pi][mm] - out["age"][:len(idx)][pj][mm])).mean()
                loss = loss + 0.05 * l_mon
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(tq, tq, tsx, ridge_z=ta)
            vl = F.l1_loss(out["age"], ty).item()
        if vl < best_loss:
            best_loss, best_state, pat = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat > patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model, sc, rg_full, mu, sd

DEEP_SEEDS = (42, 7, 2024, 2025, 12345)
X500_raw = Xc[:, sel500]
deep_models = []
t1 = time.time()
for s in DEEP_SEEDS:
    m, sc_, rg_, mu_, sd_ = train_mt(X500_raw, ytr, sextr, XL[:, sel500], yL, sexL, s)
    deep_models.append((m, sc_, rg_, mu_, sd_))
    print(f"[deep] seed{s} done ({time.time()-t1:.0f}s)", flush=True)

def deep_predict(Xext_scaled, sex_ext):
    ps = []
    for (m, sc_, rg_, mu_, sd_) in deep_models:
        with torch.no_grad():
            an = ((rg_.predict(Xext_scaled) - mu_) / sd_).astype(np.float32)
            q = torch.from_numpy(Xext_scaled.astype(np.float32)).to(DEV)
            a = torch.from_numpy(an).to(DEV)
            sx = torch.as_tensor(sex_ext, dtype=torch.float32, device=DEV)
            out = m(q, q, sx, ridge_z=a)
            ps.append(out["age"].cpu().numpy() * sd_ + mu_)
    return np.mean(np.stack(ps), axis=0)

# ---------------- evaluate on both ext sets ----------------
def eval_set(name, df, age, sex, band, grp):
    out = {}
    E40 = build_ext_matrix(df, sel40, sc40)
    E500 = build_ext_matrix(df, sel500, sc500)
    preds = {}
    preds["EN-40"] = mem["EN-40"].predict(E40)
    preds["EN-500"] = mem["EN-500"].predict(E500)
    preds["SVR-500"] = mem["SVR-500"].predict(E500)
    preds["XGB-500"] = mem["XGB-500"].predict(E500)
    preds["LoCRANE-MT-ens5"] = deep_predict(E500, sex)
    preds["LoCRANE-Ens-eq3"] = (preds["LoCRANE-MT-ens5"] + preds["EN-40"] + preds["EN-500"]) / 3.0
    preds["LinEns-noDeep"] = (preds["EN-40"] + preds["EN-500"]) / 2.0
    rng = np.random.default_rng(0)
    for tag, pr in preds.items():
        err = np.abs(age - pr)
        bs = [float(err[rng.integers(0, len(err), len(err))].mean()) for _ in range(5000)]
        row = {"MAE": round(float(err.mean()), 2), "MAE_CI95": [round(float(np.percentile(bs, 2.5)), 2),
                                                                 round(float(np.percentile(bs, 97.5)), 2)],
               "PCC": round(float(pearsonr(age, pr).statistic), 3),
               "SP": round(float(spearmanr(age, pr).statistic), 3)}
        # per-band MAE
        pb = {}
        for b in np.unique(band):
            m_ = band == b
            pb[str(int(b))] = round(float(err[m_].mean()), 2)
        row["MAE_by_band"] = pb
        out[tag] = row
        print(f"[{name}] {tag:18s} MAE={row['MAE']:6.2f} CI={row['MAE_CI95']} PCC={row['PCC']} SP={row['SP']} by_band={pb}", flush=True)
    return out, preds

res = {}
res["ext1_GTEX"], p1 = eval_set("ext1_GTEX", gt_df, gt_age, gt_sex, gt_band, gt_grp)
res["ext2_cohort2"], p2 = eval_set("ext2_cohort2", c2_df, c2_age, c2_sex, c2_band, c2_grp)

meta = {"universe_genes": len(universe), "sel40": [universe[j] for j in sel40],
        "internal_YC_n": int(len(ytr)), "internal_L_n": int(len(yL)),
        "protocol": "clean rebuild hvg8000∩ext space; train-only selection; QN harmonization fit on internal; ALL methods see identical harmonized matrices; members trained on full internal YC (760)"}
json.dump({"meta": meta, "results": res},
          open(os.path.join(RUNS, "x45_ext_battle_v4.json"), "w"), indent=1)
np.savez_compressed(os.path.join(RUNS, "x14_ext_preds_v4.npz"),
                    gtex_age=gt_age, gtex_pred_ours=p1["LoCRANE-Ens-eq3"], gtex_pred_en40=p1["EN-40"],
                    gtex_pred_en500=p1["EN-500"], gtex_pred_svr=p1["SVR-500"], gtex_pred_xgb=p1["XGB-500"],
                    gtex_pred_mt=p1["LoCRANE-MT-ens5"], gtex_pred_lin=p1["LinEns-noDeep"],
                    c2_age=c2_age, c2_pred_ours=p2["LoCRANE-Ens-eq3"], c2_pred_en40=p2["EN-40"],
                    c2_pred_en500=p2["EN-500"], c2_pred_svr=p2["SVR-500"], c2_pred_xgb=p2["XGB-500"],
                    c2_pred_mt=p2["LoCRANE-MT-ens5"], c2_pred_lin=p2["LinEns-noDeep"])
print("saved x45_ext_battle_v4.json + x14_ext_preds_v4.npz", flush=True)
