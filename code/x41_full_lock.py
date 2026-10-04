# -*- coding: utf-8 -*-
"""x41_full_lock.py — full 15-rep lock for base vs adopted INNO winners (V4-3 mono, V4-7 combo).
Mirrors x12_final_lock exactly (15 reps rng(2026), 5 seeds, eq3 blend with EN-40/EN-500).
Adoption gate: per-rep paired wilcoxon eq3'(winner) vs eq3(base), point MAE improvement."""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v4"); os.makedirs(RUNS, exist_ok=True)
DATA3 = os.path.join(HERE, "..", "data_v3")
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HVG_K, DEEPK, REPS = 8000, 500, 15
LAM_BAND, LOSS = 0.3, "L1"
LAM_MON, LAM_SP = 0.05, 0.05
DEEP_SEEDS = (42, 7, 2024, 2025, 12345)
ARMS = ["base", "v4-3_mono", "v4-7_combo"]

p_all = pd.read_csv(os.path.join(CRANE, "training_data", "balanced_folds", "fold_0", "train_expr.csv"), index_col=0)
v = p_all.var(axis=0).sort_values(ascending=False)
hvg = list(v.index[:HVG_K])
expr = pd.read_csv(os.path.join(CRANE, "expression_matrix.tsv.gz"), sep="\t", index_col=0, compression=None)
expr_hvg = expr.loc[expr.index.isin(hvg)]
lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv"))
lab["ID"] = lab["ID"].astype(str); lab = lab.set_index("ID")
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
print(f"n={n} L={len(y_L)} DEV={DEV} REPS={REPS}", flush=True)

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

def train_mt(Xtr, ytr, sex_tr, Xlp, ylp, sex_lp, seed, arm):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    ntr = len(ytr)
    vp = np.random.RandomState(seed).permutation(ntr)[:max(8, int(ntr * 0.1))]
    tp = np.setdiff1d(np.arange(ntr), vp)
    sc = StandardScaler().fit(Xtr[tp])
    Xs = sc.transform(Xtr).astype(np.float32)
    use_mono = arm in ("v4-3_mono", "v4-7_combo")
    use_sparse = arm == "v4-7_combo"
    kf = KFold(5, shuffle=True, random_state=seed)
    anchor_raw = np.zeros(ntr)
    for itr, ivl in kf.split(np.arange(ntr)):
        en_i = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[itr], ytr[itr])
        anchor_raw[ivl] = en_i.predict(Xs[ivl])
    rg_full = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[tp], ytr[tp])
    mu, sd = float(ytr[tp].mean()), max(1e-6, float(ytr[tp].std()))
    anchor = ((anchor_raw - mu) / sd).astype(np.float32)
    n_mod = 64
    assign = {int(i): int(i * n_mod // Xtr.shape[1]) for i in range(Xtr.shape[1])}
    model = CraneZV3(n_genes=Xtr.shape[1], assignment=assign, n_modules=n_mod, n_immune=0,
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
    lf = (lambda o, t: F.l1_loss(o, t))
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
            l_age = lf(out["age"][:len(idx)], ty[gsel])
            l_band = F.cross_entropy(out["band_logits"], torch.cat([tband[gsel], blq[lidx]]))
            loss = l_age + LAM_BAND * l_band
            if use_mono and len(idx) > 8:
                pi = torch.randint(0, len(idx), (len(idx),), device=DEV)
                pj = torch.randint(0, len(idx), (len(idx),), device=DEV)
                sgn = torch.sign(ty[gsel][pi] - ty[gsel][pj])
                m = sgn != 0
                l_mon = F.softplus(-sgn[m] * (out["age"][:len(idx)][pi][m] - out["age"][:len(idx)][pj][m])).mean()
                loss = loss + LAM_MON * l_mon
            if use_sparse:
                loss = loss + LAM_SP * ((out["age"][:len(idx)] - ta[gsel]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(tq[vp], tq[vp], tsx[vp], ridge_z=ta[vp])
            vl = lf(out["age"], ty[vp]).item()
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

# also need anchors/scale for OOF attribution? not here. Run lock.
JOURNAL = os.path.join(RUNS, "x41_journal.jsonl")
def jwrite(rec):
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

t0 = time.time()
table = {}
ens_preds = {a: {} for a in ARMS}   # arm -> rep -> ens pred
te_info = {}
for rep in range(REPS):
    p = SPLITS[rep]
    tr, te = p[:int(n * 0.8)], p[int(n * 0.8):]
    sel = topk(tr, DEEPK)
    te_info[rep] = te.tolist()
    # CPU members on this split
    def en_pred(k):
        if k == 40:
            ac = np.zeros(X.shape[1], dtype=np.float32)
            for j in range(X.shape[1]):
                if X[tr][:, j].std() > 0:
                    c = np.corrcoef(X[tr][:, j], y[tr])[0, 1]
                    ac[j] = abs(c) if np.isfinite(c) else 0.0
            cols = np.argsort(-ac)[:40]
        else:
            cols = sel[:k] if k <= DEEPK else sel
        s2 = StandardScaler().fit(X[tr][:, cols])
        m2 = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(s2.transform(X[tr][:, cols]), y[tr])
        return m2.predict(s2.transform(X[te][:, cols]))
    en40, en500 = en_pred(40), en_pred(500)
    for arm in ARMS:
        sp = []
        for s in DEEP_SEEDS:
            key = f"{arm}|{rep}|{s}"
            pr = train_mt(X[tr][:, sel], y[tr], sex_all[tr], X_L[:, sel], y_L, sex_L, s, arm)(
                X[te][:, sel], sex_all[te])
            sp.append(pr)
            jwrite({"key": key, "arm": arm, "rep": rep, "seed": s,
                    "MAE": float(np.abs(y[te] - pr).mean())})
        ens = np.mean(np.stack(sp), axis=0)
        ens_preds[arm][rep] = ens
        blend = (ens + en40 + en500) / 3.0
        mae_ens = float(np.abs(y[te] - ens).mean())
        mae_eq3 = float(np.abs(y[te] - blend).mean())
        pcc_eq3 = float(pearsonr(y[te], blend).statistic)
        print(f"rep{rep} [{arm}] deep={mae_ens:.3f} eq3={mae_eq3:.3f} pcc={pcc_eq3:.3f} ({time.time()-t0:.0f}s)", flush=True)

# per-rep arrays + wilcoxon
rep_mae = {a: [] for a in ARMS}
for rep in range(REPS):
    te = np.array(te_info[rep])
    for arm in ARMS:
        rep_mae[arm].append(float(np.abs(y[te] - ens_preds[arm][rep]).mean()))
out = {"protocol": "YC ALL 50-89 hvg8000 rng(2026) 15 reps 80/20 5 seeds, eq3 blend; INNO V4 full lock",
       "per_rep_MAE": rep_mae,
       "mean_MAE": {a: float(np.mean(v)) for a, v in rep_mae.items()}}
for arm in ARMS[1:]:
    d = np.array(rep_mae[arm]) - np.array(rep_mae["base"])
    w_p = float(wilcoxon(rep_mae[arm], rep_mae["base"]).pvalue) if np.any(d != 0) else 1.0
    out[f"{arm}_vs_base"] = {"mean_gain": float(np.mean(rep_mae["base"]) - np.mean(rep_mae[arm])),
                              "rep_wins": f"{int((d < 0).sum())}/{REPS}", "wilcoxon_p": w_p}
    print(arm, "gain", out[f"{arm}_vs_base"], flush=True)
json.dump(out, open(os.path.join(RUNS, "x41_lock.json"), "w"), indent=1)
print("-> x41_lock.json", flush=True)
