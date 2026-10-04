# -*- coding: utf-8 -*-
"""run_demo.py — one-command demonstration of the LoCRANE recipe on the released
real data slice (300 variance-selected genes x 150 samples from the Chinese
longevity cohort; 100 working-age M-band + 50 verified 90+ L-band samples).

What it does (demo-scale protocol: 5 locked splits instead of the paper's 15,
reduced epochs — same architecture and hyperparameters):
  1. cross-fitted ridge-residual anchor
  2. FiLM sex conditioning
  3. three-band auxiliary supervision (the L band enters the auxiliary task)
  4. RankNet-style monotonicity regularizer (lambda = 0.05)
  5. seed ensemble blended with EN-40 / EN-500 elastic nets -> eq3' composition
  6. ablation: priors-off same-backbone control (the +0.30y-style prior value)

Runtime: ~1-2 minutes on CPU. Requires: numpy pandas scikit-learn torch.
"""
import gzip, hashlib, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "demo")
LAM_MON, LAM_SP = 0.05, 0.05
SEEDS = (42, 7, 2024)
SPLITS, EPOCHS = 5, 60

class PriorsMLP(nn.Module):
    """Residual age network with the four LoCRANE priors."""
    def __init__(self, d_in, use_film=True, use_band=True):
        super().__init__()
        self.use_film, self.use_band = use_film, use_band
        self.fc1 = nn.Linear(d_in, 128)
        self.block = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, 1)
        self.band = nn.Linear(128, 3)
        self.film = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 256))
    def forward(self, x, sex):
        h = F.relu(self.fc1(x))
        if self.use_film:
            gb = self.film(sex.view(-1, 1))
            gamma, beta = gb[:, :128], gb[:, 128:]
            h = h * (1 + torch.tanh(gamma)) + torch.tanh(beta)
        h = h + self.block(h)
        return self.head(h).squeeze(-1), self.band(h)

def band_of(a):
    return np.where(a <= 50, 0, np.where(a < 90, 1, 2)).astype(np.int64)

def train_deep(Xtr, ytr, sex_tr, ap_tr, s, Xte, sex_te, ap_te,
               yL, XL, sexL, use_film, use_band, use_mono, seed, epochs=EPOCHS):
    """Residual-on-anchor deep member: net outputs a tanh-bounded correction,
    final prediction = anchor + s * tanh(g(x)). s = cross-fitted residual SD."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = PriorsMLP(Xtr.shape[1], use_film, use_band)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    st = torch.tensor(sex_tr, dtype=torch.float32)
    at = torch.tensor((ytr - ap_tr) / s, dtype=torch.float32)   # bounded correction target
    bt = torch.tensor(band_of(ytr), dtype=torch.int64)
    XLt = torch.tensor(XL, dtype=torch.float32); sL = torch.tensor(sexL, dtype=torch.float32)
    bL = torch.tensor(band_of(yL), dtype=torch.int64)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    n = len(yt)
    for ep in range(epochs):
        net.train()
        for i in torch.split(torch.randperm(n), 64):
            corr, band_logits = net(Xt[i], st[i])
            g = torch.tanh(corr)
            pred = at[i] + g                       # normalised residual scale
            loss = F.smooth_l1_loss(pred, torch.zeros_like(pred))
            if use_band:
                loss = loss + LAM_SP * F.cross_entropy(band_logits, bt[i])
            if use_mono and i.numel() > 1:
                d = pred.view(-1, 1) - pred.view(1, -1)
                sign = (yt[i].view(-1, 1) - yt[i].view(1, -1)) < 0
                loss = loss + LAM_MON * (F.softplus(d) * sign.float()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if use_band and len(yL):
            net.train()
            _, bl = net(XLt, sL)
            loss = 0.3 * F.cross_entropy(bl, bL)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        corr, _ = net(Xe, torch.tensor(sex_te, dtype=torch.float32))
        return ap_te + s * torch.tanh(corr).numpy()

def main():
    X_e = os.path.join(DATA, "expression_demo.csv.gz")
    h = hashlib.sha256(open(X_e, "rb").read()).hexdigest()
    ref = open(os.path.join(DATA, "SHA256SUMS")).read().split()[0]
    print(f"[data] expression_demo.csv.gz sha256 {h[:12]} (expected {ref[:12]}) {'OK' if h == ref else 'MISMATCH'}")
    with gzip.open(X_e, "rt") as f:
        E = pd.read_csv(f, sep="\t", index_col=0)
    P = pd.read_csv(os.path.join(DATA, "phenotype_demo.csv")).set_index("sample_id")
    common = [s for s in P.index if s in E.columns]
    E, P = E[common], P.loc[common]
    X = np.nan_to_num(E.T.values, nan=0.0).astype(np.float32)
    y = P["age"].to_numpy(float)
    sex = (P["sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
    is_L = (y >= 90)
    print(f"[data] n={len(y)} (working-age {int((~is_L).sum())}, 90+ {int(is_L.sum())}) genes={X.shape[1]}")

    rng = np.random.default_rng(2026)
    n = len(y)
    SPL = [rng.permutation(np.where(~is_L)[0]) for _ in range(SPLITS)]
    err = {"eq3p": [], "nopriors": [], "ridge": []}
    for r, perm in enumerate(SPL):
        k = max(1, len(perm) // 5)
        te, tr = perm[:k], perm[k:]
        trL = np.where(is_L)[0]  # the 90+ tail: auxiliary supervision only
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype(np.float32)
        r0 = Ridge(alpha=10.0).fit(Xs[tr], y[tr])
        # cross-fitted ridge anchor + residual scale
        ap_kf = np.zeros(len(tr), dtype=np.float32)
        for a, b in KFold(5, shuffle=True, random_state=42).split(tr):
            r_ = Ridge(alpha=10.0).fit(Xs[tr[a]], y[tr[a]])
            ap_kf[b] = r_.predict(Xs[tr[b]])
        s = float(np.std(y[tr] - ap_kf)) + 1e-6
        ap_te = r0.predict(Xs[te])
        # eq3': priors-on deep seed ensemble (anchor-bounded corrections)
        pd_ = np.mean([train_deep(Xs[tr], y[tr], sex[tr], ap_kf, s, Xs[te], sex[te], ap_te,
                                  y[trL], Xs[trL], sex[trL], True, True, True, s_) for s_ in SEEDS], axis=0)
        # elastic-net members (Kong-protocol budgets)
        fc = np.array([abs(np.corrcoef(Xs[tr, j], y[tr])[0, 1]) if Xs[tr, j].std() > 0 else 0
                       for j in range(Xs.shape[1])])
        top40, top500 = np.argsort(-fc)[:40], np.argsort(-fc)[:500]
        en40 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000).fit(Xs[tr][:, top40], y[tr]).predict(Xs[te][:, top40])
        en500 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000).fit(Xs[tr][:, top500], y[tr]).predict(Xs[te][:, top500])
        eq3p = np.mean([pd_, en40, en500], axis=0)
        # priors-off same-backbone control
        pn = np.mean([train_deep(Xs[tr], y[tr], sex[tr], ap_kf, s, Xs[te], sex[te], ap_te,
                                 y[trL], Xs[trL], sex[trL], False, False, False, s_) for s_ in SEEDS], axis=0)
        rg = Ridge(alpha=10.0).fit(Xs[tr], y[tr]).predict(Xs[te])
        err["eq3p"].append(np.abs(eq3p - y[te]).mean())
        err["nopriors"].append(np.abs(pn - y[te]).mean())
        err["ridge"].append(np.abs(rg - y[te]).mean())
        print(f"[split {r}] eq3' MAE {err['eq3p'][-1]:.2f} | priors-off {err['nopriors'][-1]:.2f} | ridge {err['ridge'][-1]:.2f}")
    a, b_ = np.array(err["eq3p"]), np.array(err["nopriors"])
    p = wilcoxon(a, b_).pvalue if SPLITS >= 5 else float("nan")
    print("\n==== demo summary ({} splits, demo-scale) ====".format(SPLITS))
    print(f"ridge (300-gene, tuned) reference MAE  : {np.mean(err['ridge']):.3f} y")
    print(f"eq3' (four priors + hybrid blend) MAE : {a.mean():.3f} y")
    print(f"same-backbone priors-off control MAE : {b_.mean():.3f} y")
    print(f"prior value on this slice            : {b_.mean() - a.mean():+.3f} y (paired p={p:.3f})")
    print("NOTE: demo-scale numbers are for pipeline verification only;")
    print("      paper numbers come from the 15-split locked protocol on the full cohort.")

if __name__ == "__main__":
    main()
