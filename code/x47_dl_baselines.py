# -*- coding: utf-8 -*-
"""x47_dl_baselines.py — R5-P1-1 full rectification: fair DL baseline suite.
Same 15 locked folds (rng 2026), same top-500 train-only selection, same standardization,
same internal-val early stopping as our deep member (patience 20), y standardized.
DL baselines get 3 seeds (MORE budget than classical baselines' 1 fit — conservative for us):
  MLP-wide 1024-256-32 | CNN1D over gene axis | FT-Transformer-mini (per-feature tokens)
  CraneZ-vanilla (SAME backbone, ALL domain-bias priors OFF: no anchor, no FiLM, no band-aux, no mono)
Reference: sklearn MLP-500 from audit/x46_dl_baseline_rectify.json.
Saves per-sample preds npz BEFORE verdict computation (AAR lesson)."""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v4")
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p_all = pd.read_csv(CRANE + "/training_data/balanced_folds/fold_0/train_expr.csv", index_col=0)
v = p_all.var(axis=0).sort_values(ascending=False)
hvg = list(v.index[:8000])
expr = pd.read_csv(CRANE + "/expression_matrix.tsv.gz", sep="\t", index_col=0, compression=None)
expr_hvg = expr.loc[expr.index.isin(hvg)]
lab = pd.read_csv(os.path.join(HERE, "..", "data_v3", "labels_v3.csv"))
lab["ID"] = lab["ID"].astype(str); lab = lab.set_index("ID")
yc = lab[(lab.age >= 50) & (lab.age < 90)]
common = [s for s in yc.index if s in expr_hvg.columns]
X = np.nan_to_num(expr_hvg[common].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = yc.loc[common, "age"].to_numpy(float)
sex_all = (yc.loc[common, "sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
n = len(y)
rng = np.random.default_rng(2026)
SPLITS = [rng.permutation(n) for _ in range(15)]
SEEDS3 = (42, 7, 2024)
print(f"X={X.shape} DEV={DEV}", flush=True)

def topk(tr, k=500):
    ac = np.zeros(X.shape[1], dtype=np.float32)
    Xt, yt = X[tr], y[tr]
    for j in range(Xt.shape[1]):
        if Xt[:, j].std() > 0:
            c = np.corrcoef(Xt[:, j], yt)[0, 1]
            ac[j] = abs(c) if np.isfinite(c) else 0.0
    return np.argsort(-ac)[:k]

class MLPWide(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 1024), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(1024, 256), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(8),
                                 nn.Flatten(), nn.Linear(512, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x.unsqueeze(1)).squeeze(-1)

class FTMini(nn.Module):
    def __init__(self, n_feat, d=32, n_heads=4, n_layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=ff,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        h = self.enc(torch.cat([self.cls.expand(len(x), -1, -1), tok], 1))
        return self.head(h[:, 0]).squeeze(-1)

MODELS = ["MLP-wide", "CNN1D", "FT-mini", "CraneZ-vanilla"]

def make(name, n_feat, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    if name == "MLP-wide": return MLPWide(n_feat).to(DEV)
    if name == "CNN1D": return CNN1D().to(DEV)
    if name == "FT-mini": return FTMini(n_feat).to(DEV)
    if name == "CraneZ-vanilla":
        assign = {int(i): int(i * 64 // n_feat) for i in range(n_feat)}
        return CraneZV3(n_genes=n_feat, assignment=assign, n_modules=64, n_immune=0,
                        d_model=64, n_heads=4, n_layers=2, ff_dim=128,
                        ridge_residual=0.0, use_film=False).to(DEV)

def fit_predict(name, Xtr, ytr, sex_tr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    ntr = len(ytr)
    vp = np.random.RandomState(seed).permutation(ntr)[:max(8, int(ntr * 0.1))]
    tp = np.setdiff1d(np.arange(ntr), vp)
    sc = StandardScaler().fit(Xtr[tp])
    Xs = sc.transform(Xtr).astype(np.float32)
    mu, sd = float(ytr[tp].mean()), max(1e-6, float(ytr[tp].std()))
    ty = ((ytr - mu) / sd).astype(np.float32)
    model = make(name, Xtr.shape[1], seed)
    tq = torch.from_numpy(Xs).to(DEV)
    tyt = torch.from_numpy(ty).to(DEV)
    tsx = torch.from_numpy(sex_tr).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, pat = 1e9, None, 0
    vanilla = (name == "CraneZ-vanilla")
    for ep in range(200):
        model.train()
        perm = torch.randperm(len(tp), device=DEV)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            if vanilla:
                out = model(tq[tp][idx], tq[tp][idx], tsx[tp][idx], ridge_z=None)
                loss = F.l1_loss(out["age"], tyt[tp][idx])
            else:
                pr = model(tq[tp][idx])
                loss = F.l1_loss(pr, tyt[tp][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            if vanilla:
                out = model(tq[vp], tq[vp], tsx[vp], ridge_z=None)
                vl = F.l1_loss(out["age"], tyt[vp]).item()
            else:
                vl = F.l1_loss(model(tq[vp]), tyt[vp]).item()
        if vl < best:
            best, best_state, pat = vl, {k: vv.cpu().clone() for k, vv in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 20: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xn = sc.transform(Xte).astype(np.float32)
        q = torch.from_numpy(Xn).to(DEV)
        if vanilla:
            out = model(q, q, torch.zeros(len(q), device=DEV), ridge_z=None)
            pr = out["age"].cpu().numpy() * sd + mu
        else:
            pr = model(q).cpu().numpy() * sd + mu
    return pr

t0 = time.time()
preds = {m: {} for m in MODELS}   # model -> rep -> mean-of-3-seeds pred
maes = {m: [] for m in MODELS}
for rep in range(15):
    p = SPLITS[rep]
    tr, te = p[:int(n * 0.8)], p[int(n * 0.8):]
    sel = topk(tr)
    for m in MODELS:
        sp = []
        for s in SEEDS3:
            pr = fit_predict(m, X[tr][:, sel], y[tr], sex_all[tr], X[te][:, sel], s)
            sp.append(pr)
        ens = np.mean(np.stack(sp), axis=0)
        preds[m][rep] = ens
        maes[m].append(float(np.abs(y[te] - ens).mean()))
    print(f"rep{rep} " + " ".join(f"{m}={maes[m][-1]:.2f}" for m in MODELS) + f" ({time.time()-t0:.0f}s)", flush=True)

# save preds BEFORE verdicts (AAR lesson)
flat = {}
for m in MODELS:
    for rep in range(15):
        flat[f"{m}__rep{rep}_pred"] = preds[m][rep]
        flat[f"{m}__rep{rep}_te"] = SPLITS[rep][int(n * 0.8):]
np.savez_compressed(os.path.join(RUNS, "x47_dl_preds.npz"), **flat)

eq3 = json.load(open(os.path.join(RUNS, "x41_eq3_from_log.json")))
w = np.array(eq3["v4-3_mono"]["eq3"])
out = {"protocol": "R5-P1-1 full DL baseline suite: 15 locked folds, top500 train-only, 3 seeds averaged (more budget than classical baselines), internal-val early stopping identical to our deep member",
       "per_rep_MAE": {m: [round(x, 3) for x in v_] for m, v_ in maes.items()},
       "table": {}}
ps = []
for m in MODELS:
    arr = np.array(maes[m])
    d = arr - w
    pv = float(wilcoxon(arr, w, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    ps.append((m, pv))
    out["table"][m] = {"MAE": round(float(arr.mean()), 4), "delta_vs_winner": round(float(-d.mean()), 4),
                        "wins_of_winner": f"{int((d > 0).sum())}/15", "p": round(pv, 6)}
ps.sort(key=lambda x: x[1])
holm = {m: min(1.0, p * (len(ps) - i)) for i, (m, p) in enumerate(ps)}
out["holm_p"] = {m: round(p, 6) for m, p in holm.items()}
x46 = json.load(open(os.path.join(RUNS, "audit", "x46_dl_baseline_rectify.json")))
out["reference_sklearn_MLP-500"] = {"MAE": x46["MLP-500"]["MAE"], "wins_of_winner": x46["MLP-500"]["wins_of_winner"],
                                     "p": x46["MLP-500"]["vs_winner_p"]}
json.dump(out, open(os.path.join(RUNS, "x47_dl_baselines.json"), "w"), indent=1)
print(json.dumps(out["table"], indent=1))
print("Holm:", out["holm_p"])
print("-> x47_dl_baselines.json")
