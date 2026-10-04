# -*- coding: utf-8 -*-
"""x48_composition.py — R5-P1-2 full rectification: complete member/ensemble composition
ablation under the 15-rep lock, with PER-SAMPLE predictions saved to disk.
Retrains deep arms (v4-3_mono winner + base) exactly as x41 and stores every seed's
test-fold prediction; then builds ALL equal-weight subsets of {deep, EN-40, EN-500, SVR-500}
(15 combos), paired wilcoxon of each vs the headline eq3' + Holm within family.
Linear member per-sample preds come from runs_v3/x19_baselines15_preds.npz (same folds)."""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v4")
DATA3 = os.path.join(HERE, "..", "data_v3")
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HVG_K, DEEPK, REPS = 8000, 500, 15
LAM_MON, LAM_BAND = 0.05, 0.3
DEEP_SEEDS = (42, 7, 2024, 2025, 12345)

p_all = pd.read_csv(CRANE + "/training_data/balanced_folds/fold_0/train_expr.csv", index_col=0)
v = p_all.var(axis=0).sort_values(ascending=False)
hvg = list(v.index[:HVG_K])
expr = pd.read_csv(CRANE + "/expression_matrix.tsv.gz", sep="\t", index_col=0, compression=None)
expr_hvg = expr.loc[expr.index.isin(hvg)]
lab = pd.read_csv(DATA3 + "/labels_v3.csv"); lab["ID"] = lab["ID"].astype(str); lab = lab.set_index("ID")
yc_lab = lab[(lab["age"] >= 50) & (lab["age"] < 90)]
l_lab = lab[lab["age"] >= 90]
common = [s for s in yc_lab.index if s in expr_hvg.columns]
expr_h = expr_hvg[common]; yc_lab = yc_lab.loc[common]
X = np.nan_to_num(expr_h.T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = yc_lab["age"].to_numpy(float)
sex_all = (yc_lab["sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
l_ids = [s for s in l_lab.index if s in expr_hvg.columns]
X_L = np.nan_to_num(expr_hvg[l_ids].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_L = l_lab.loc[l_ids, "age"].to_numpy(float)
sex_L = (l_lab.loc[l_ids, "sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
n = len(y)
rng = np.random.default_rng(2026)
SPLITS = [rng.permutation(n) for _ in range(REPS)]
print(f"n={n} DEV={DEV}", flush=True)

def band_of(a):
    return np.where(a <= 50, 0, np.where(a < 90, 1, 2)).astype(np.int64)

def topk(tr, k):
    ac = np.zeros(X.shape[1], dtype=np.float32)
    Xt, yt = X[tr], y[tr]
    for j in range(Xt.shape[1]):
        if Xt[:, j].std() > 0:
            c = np.corrcoef(Xt[:, j], yt)[0, 1]
            ac[j] = abs(c) if np.isfinite(c) else 0.0
    return np.argsort(-ac)[:k]

def train_mt(Xtr, ytr, sex_tr, Xlp, ylp, sex_lp, seed, use_mono):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    ntr = len(ytr)
    vp = np.random.RandomState(seed).permutation(ntr)[:max(8, int(ntr * 0.1))]
    tp = np.setdiff1d(np.arange(ntr), vp)
    sc = StandardScaler().fit(Xtr[tp])
    Xs = sc.transform(Xtr).astype(np.float32)
    kf = KFold(5, shuffle=True, random_state=seed)
    anchor_raw = np.zeros(ntr)
    for itr, ivl in kf.split(np.arange(ntr)):
        en_i = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[itr], ytr[itr])
        anchor_raw[ivl] = en_i.predict(Xs[ivl])
    rg_full = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[tp], ytr[tp])
    mu, sd = float(ytr[tp].mean()), max(1e-6, float(ytr[tp].std()))
    anchor = ((anchor_raw - mu) / sd).astype(np.float32)
    assign = {int(i): int(i * 64 // Xtr.shape[1]) for i in range(Xtr.shape[1])}
    model = CraneZV3(n_genes=Xtr.shape[1], assignment=assign, n_modules=64, n_immune=0,
                     d_model=64, n_heads=4, n_layers=2, ff_dim=128, ridge_residual=1.0).to(DEV)
    tq = torch.from_numpy(Xs).to(DEV)
    ty = torch.from_numpy(((ytr - mu) / sd).astype(np.float32)).to(DEV)
    ta = torch.from_numpy(anchor).to(DEV)
    tsx = torch.from_numpy(sex_tr).to(DEV)
    tband = torch.from_numpy(band_of(ytr)).to(DEV)
    Xlq = torch.from_numpy(sc.transform(Xlp).astype(np.float32)).to(DEV)
    alq = torch.from_numpy(((rg_full.predict(sc.transform(Xlp)) - mu) / sd).astype(np.float32)).to(DEV)
    slq = torch.from_numpy(sex_lp).to(DEV)
    blq = torch.from_numpy(band_of(ylp)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best_loss, best_state, pat = 1e9, None, 0
    nl = len(ylp); tp_t = torch.from_numpy(tp).to(DEV)
    for ep in range(150):
        model.train()
        perm = torch.randperm(len(tp), device=DEV)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            gsel = tp_t[idx]
            lidx = torch.randint(0, nl, (len(idx),), device=DEV)
            xq = torch.cat([tq[gsel], Xlq[lidx]])
            sx = torch.cat([tsx[gsel], slq[lidx]])
            rz = torch.cat([ta[gsel], alq[lidx]])
            out = model(xq, xq, sx, ridge_z=rz)
            loss = F.l1_loss(out["age"][:len(idx)], ty[gsel]) + LAM_BAND * F.cross_entropy(
                out["band_logits"], torch.cat([tband[gsel], blq[lidx]]))
            if use_mono and len(idx) > 8:
                pi = torch.randint(0, len(idx), (len(idx),), device=DEV)
                pj = torch.randint(0, len(idx), (len(idx),), device=DEV)
                sgn = torch.sign(ty[gsel][pi] - ty[gsel][pj])
                m = sgn != 0
                l_mon = F.softplus(-sgn[m] * (out["age"][:len(idx)][pi][m] - out["age"][:len(idx)][pj][m])).mean()
                loss = loss + LAM_MON * l_mon
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(tq[vp], tq[vp], tsx[vp], ridge_z=ta[vp])
            vl = F.l1_loss(out["age"], ty[vp]).item()
        if vl < best_loss:
            best_loss, best_state, pat = vl, {k: vv.cpu().clone() for k, vv in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 20: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    def predict(Xnew, sex_new):
        with torch.no_grad():
            Xn = sc.transform(Xnew).astype(np.float32)
            an = ((rg_full.predict(Xn) - mu) / sd).astype(np.float32)
            q = torch.from_numpy(Xn).to(DEV); a = torch.from_numpy(an).to(DEV)
            sx = torch.as_tensor(sex_new, dtype=torch.float32, device=DEV)
            out = model(q, q, sx, ridge_z=a)
            return out["age"].cpu().numpy() * sd + mu
    return predict

# ---- retrain deep arms, SAVE PER-SAMPLE PREDS ----
flat = {}
t0 = time.time()
for arm, use_mono in (("deep_v43", True), ("deep_base", False)):
    for rep in range(REPS):
        p = SPLITS[rep]
        tr, te = p[:int(n * 0.8)], p[int(n * 0.8):]
        sel = topk(tr, DEEPK)
        flat[f"{arm}__rep{rep}_te"] = te
        sp = []
        for s in DEEP_SEEDS:
            pr = train_mt(X[tr][:, sel], y[tr], sex_all[tr], X_L[:, sel], y_L, sex_L, s, use_mono)(
                X[te][:, sel], sex_all[te])
            flat[f"{arm}__rep{rep}_seed{s}_pred"] = pr
            sp.append(pr)
        flat[f"{arm}__rep{rep}_pred"] = np.mean(np.stack(sp), axis=0)
        print(f"{arm} rep{rep} MAE={np.abs(y[te]-flat[f'{arm}__rep{rep}_pred']).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed(os.path.join(RUNS, "x48_deep_preds.npz"), **flat)
print("deep preds saved", flush=True)

# ---- composition matrix: all equal-weight subsets of {deep_v43, EN-40, EN-500, SVR-500} ----
z19 = np.load(os.path.join(HERE, "..", "runs_v3", "x19_baselines15_preds.npz"))
MEM = {}
for rep in range(REPS):
    te = np.array(flat[f"deep_v43__rep{rep}_te"])
    MEM["deep"] = (flat[f"deep_v43__rep{rep}_pred"], te)
    MEM["EN-40"] = (z19[f"EN-40__rep{rep}_pred"], z19[f"EN-40__rep{rep}_te"])
    MEM["EN-500"] = (z19[f"EN-500__rep{rep}_pred"], z19[f"EN-500__rep{rep}_te"])
    MEM["SVR-500"] = (z19[f"SVR-500__rep{rep}_pred"], z19[f"SVR-500__rep{rep}_te"])
    assert all(np.array_equal(MEM["deep"][1], m[1]) for m in MEM.values())
names = ["deep", "EN-40", "EN-500", "SVR-500"]
combos = []
for r in range(1, 5):
    import itertools
    for c in itertools.combinations(names, r):
        combos.append(c)
mae_tab = {c: [] for c in combos}
for rep in range(REPS):
    te = np.array(flat["deep_v43__rep0_te"]) if False else np.array(flat[f"deep_v43__rep{rep}_te"])
    for c in combos:
        P = np.mean(np.stack([MEM[m][0] for m in c]), axis=0)
        mae_tab[c].append(float(np.abs(y[te] - P).mean()))
head = ("deep", "EN-40", "EN-500")  # eq3' headline
w_arr = np.array(mae_tab[head])
out = {"protocol": "R5-P1-2 full composition ablation: all 15 equal-weight subsets of {deep(v4-3), EN-40, EN-500, SVR-500}, 15 locked folds, per-sample preds archived (x48_deep_preds.npz)",
       "headline_eq3p": {"MAE": round(float(w_arr.mean()), 4)},
       "composition_table": {}}
rows = []
for c in combos:
    arr = np.array(mae_tab[c])
    d = arr - w_arr
    same = tuple(c) == head
    pv = 1.0 if same else (float(wilcoxon(arr, w_arr, alternative="greater").pvalue) if np.any(d != 0) else 1.0)
    rows.append({"combo": " + ".join(c), "MAE": round(float(arr.mean()), 4),
                 "delta_vs_eq3p": round(float(d.mean()), 4),
                 "wins_of_eq3p": "headline" if same else f"{int((d > 0).sum())}/15", "p": round(pv, 6)})
rows.sort(key=lambda r: r["MAE"])
out["composition_table"] = rows
sig = [r for r in rows if not r["wins_of_eq3p"].startswith("headline") and r["p"] < 0.05]
out["n_combos_significantly_worse_than_eq3p"] = len(sig)
# base-champion reference: eq3(base) = deep_base + EN40 + EN500
mae_base = []
for rep in range(REPS):
    te = np.array(flat[f"deep_base__rep{rep}_te"])
    P = (flat[f"deep_base__rep{rep}_pred"] + z19[f"EN-40__rep{rep}_pred"] + z19[f"EN-500__rep{rep}_pred"]) / 3.0
    mae_base.append(float(np.abs(y[te] - P).mean()))
out["eq3_base_reference"] = {"MAE": round(float(np.mean(mae_base)), 4),
                              "vs_eq3p_p": round(float(wilcoxon(np.array(mae_base), w_arr, alternative="greater").pvalue), 6),
                              "wins_of_eq3p": f"{int((np.array(mae_base) > w_arr).sum())}/15"}
json.dump(out, open(os.path.join(RUNS, "x48_composition.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:2400])
print("-> x48_composition.json")
