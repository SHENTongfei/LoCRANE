# -*- coding: utf-8 -*-
"""x44_mining.py — v4 model-internal mining matrix (25 seeds, forward-probe axes).
Loads x43 npz for IG axes (M1/M2L/M2Y/M7/M8); retrains 25 winner models cheaply for:
  C1 causal levers (top-60 M1 genes, do-perturb -> delta age, forward-only)
  C3 module do-ablation (64 modules)
  A2 sex-divergent correction spectrum (per-sex delta-IG, 25 seeds)
  A4 nonlinearity map (correction curve by 5y age bins)
  B2 geometry velocity (CLS manifold spread ratio Y vs L)
  G1 linear core coefficients (EN-500, deterministic)
Gates: split-half rho>=0.6 within the 25 seeds; M-axes also carry x43 perm-null counts.
Output: runs_v4/MINING_V4.json + FINDINGS_V4.md skeleton."""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v4")
MINV2 = os.path.join(HERE, "..", "runs_v3", "mining_v2")
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM_MON = 0.05

p_all = pd.read_csv(os.path.join(CRANE, "training_data", "balanced_folds", "fold_0", "train_expr.csv"), index_col=0)
v = p_all.var(axis=0).sort_values(ascending=False)
hvg8 = list(v.index[:8000])
expr = pd.read_csv(os.path.join(CRANE, "expression_matrix.tsv.gz"), sep="\t", index_col=0, compression=None)
expr_hvg = expr.loc[expr.index.isin(hvg8)]
lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv"))
lab["ID"] = lab["ID"].astype(str); lab = lab.set_index("ID")
yc_lab = lab[(lab["age"] >= 50) & (lab["age"] < 90)]
l_lab = lab[lab["age"] >= 90]
ids = [s for s in yc_lab.index if s in expr_hvg.columns]
X = np.nan_to_num(expr_hvg[ids].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = yc_lab.loc[ids, "age"].to_numpy(float)
sex_all = (yc_lab.loc[ids, "sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
l_ids = [s for s in l_lab.index if s in expr_hvg.columns]
XL = np.nan_to_num(expr_hvg[l_ids].T.values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
yL = l_lab.loc[l_ids, "age"].to_numpy(float)
sexL = (l_lab.loc[l_ids, "sex"].astype(str).str.upper().str[0] == "F").to_numpy().astype(np.float32)
gene_names = np.array(list(expr_hvg.index))
ac = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0.0
               for j in range(X.shape[1])], dtype=np.float32)
SEL = np.sort(np.argsort(-ac)[:500])
gene_sel = gene_names[SEL]
X = X[:, SEL]; XL = XL[:, SEL]
x43 = json.load(open(os.path.join(RUNS, "x43_xai25.json")))
M1top = x43["M1_top30"]
print(f"YC n={len(y)} DEV={DEV}", flush=True)

SEEDS = [42, 7, 2024, 2025, 12345, 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 131, 139, 147, 155]

def band_of(a):
    return np.where(a <= 50, 0, np.where(a < 90, 1, 2)).astype(np.int64)

def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X).astype(np.float32)
    kf = KFold(5, shuffle=True, random_state=seed)
    anchor_raw = np.zeros(len(y))
    for itr, ivl in kf.split(np.arange(len(y))):
        en_i = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[itr], y[itr])
        anchor_raw[ivl] = en_i.predict(Xs[ivl])
    rg_full = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs, y)
    mu, sd = float(y.mean()), max(1e-6, float(y.std()))
    anchor = ((anchor_raw - mu) / sd).astype(np.float32)
    assign = {int(i): int(i * 64 // 500) for i in range(500)}
    model = CraneZV3(n_genes=500, assignment=assign, n_modules=64, n_immune=0,
                     d_model=64, n_heads=4, n_layers=2, ff_dim=128, ridge_residual=1.0).to(DEV)
    tq = torch.from_numpy(Xs).to(DEV)
    ty = torch.from_numpy(((y - mu) / sd).astype(np.float32)).to(DEV)
    ta = torch.from_numpy(anchor).to(DEV)
    tsx = torch.from_numpy(sex_all).to(DEV)
    tband = torch.from_numpy(band_of(y)).to(DEV)
    Xlq = torch.from_numpy(sc.transform(XL).astype(np.float32)).to(DEV)
    alq = torch.from_numpy(((rg_full.predict(sc.transform(XL)) - mu) / sd).astype(np.float32)).to(DEV)
    slq = torch.from_numpy(sexL).to(DEV)
    blq = torch.from_numpy(band_of(yL)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best_loss, best_state, pat = 1e9, None, 0
    for ep in range(150):
        model.train()
        perm = torch.randperm(len(y), device=DEV)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            lidx = torch.randint(0, len(yL), (len(idx),), device=DEV)
            xq = torch.cat([tq[idx], Xlq[lidx]])
            sx = torch.cat([tsx[idx], slq[lidx]])
            rz = torch.cat([ta[idx], alq[lidx]])
            out = model(xq, xq, sx, ridge_z=rz)
            band_t = torch.cat([tband[idx], blq[lidx]])
            loss = F.l1_loss(out["age"][:len(idx)], ty[idx]) + 0.3 * F.cross_entropy(out["band_logits"], band_t)
            if len(idx) > 8:
                pi = torch.randint(0, len(idx), (len(idx),), device=DEV)
                pj = torch.randint(0, len(idx), (len(idx),), device=DEV)
                sgn = torch.sign(ty[idx][pi] - ty[idx][pj])
                m = sgn != 0
                l_mon = F.softplus(-sgn[m] * (out["age"][:len(idx)][pi][m] - out["age"][:len(idx)][pj][m])).mean()
                loss = loss + LAM_MON * l_mon
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(tq, tq, tsx, ridge_z=ta)
            vl = F.l1_loss(out["age"], ty).item()
        if vl < best_loss:
            best_loss, best_state, pat = vl, {k: vv.cpu().clone() for k, vv in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat > 20: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model, sc, rg_full, mu, sd

def scaled(model_pack, Xpop):
    _, sc, rg_full, mu, sd = model_pack
    Xn = sc.transform(Xpop).astype(np.float32)
    an = ((rg_full.predict(Xn) - mu) / sd).astype(np.float32)
    return (torch.from_numpy(Xn).to(DEV), torch.from_numpy(an).to(DEV),
            torch.as_tensor((sexL if Xpop is XL else sex_all), dtype=torch.float32, device=DEV))

def read_age(model, q, a, sx, gene_idx=None, set_value=None, zero_modules=None):
    with torch.no_grad():
        qq = q.clone()
        if gene_idx is not None and set_value is not None:
            qq[:, gene_idx] = set_value
        mod = model.pool(qq).unsqueeze(-1)
        h = model.module_embed(mod) + model.pos_embed
        h = model.film(h, sx)
        if zero_modules is not None:
            h = h.clone(); h[:, zero_modules] = 0.0
        h_all = torch.cat([model.cls_token.expand(len(qq), -1, -1), h], dim=1)
        for blk in model.blocks:
            h_all = blk(h_all, None)
        h_all = model.norm(h_all)
        cls = h_all[:, 0] + model.gene_proj(qq)
        delta = model.reg_head(torch.cat([cls, model.reg_gene_proj(qq), a.unsqueeze(-1)], -1)).squeeze(-1)
        return (a + model.ridge_residual * torch.tanh(delta)).cpu().numpy()

# ---------- 25-seed probes ----------
t0 = time.time()
C1 = []; C3 = []; A2F = []; A2M = []; A4 = []; B2 = []
gidx_top = np.array([int(np.where(gene_sel == g)[0][0]) for g in M1top[:60] if g in set(gene_sel)])
med = np.median(X, axis=0)
for k, s in enumerate(SEEDS):
    mp = train_seed(s)
    model = mp[0]
    q, a, sx = scaled(mp, X)
    base = read_age(model, q, a, sx)
    # C1: do-perturb top-60 genes to median (forward-only)
    c1 = {}
    for gi in gidx_top:
        pa = read_age(model, q, a, sx, gene_idx=int(gi), set_value=float(med[gi]))
        c1[int(gi)] = float(np.mean(np.abs(pa - base)))
    C1.append(c1)
    # C3: module do-ablation
    c3 = {}
    for mmod in range(64):
        pa = read_age(model, q, a, sx, zero_modules=mmod)
        c3[mmod] = float(np.mean(np.abs(pa - base)))
    C3.append(c3)
    # A2: per-sex correction spectra (mean tanh-delta contribution per gene via reg_gene_proj path? use do-diff on delta instead)
    f_mask = sex_all == 1; m_mask = sex_all == 0
    aF = read_age(model, q[f_mask], a[f_mask], sx[f_mask])
    aM = read_age(model, q[m_mask], a[m_mask], sx[m_mask])
    dF = []; dM = []
    for gi in gidx_top:
        pF = read_age(model, q[f_mask], a[f_mask], sx[f_mask], gene_idx=int(gi), set_value=float(med[gi]))
        pM = read_age(model, q[m_mask], a[m_mask], sx[m_mask], gene_idx=int(gi), set_value=float(med[gi]))
        dF.append(float((aF - pF).mean())); dM.append((float((aM - pM).mean())))
    A2F.append(dF); A2M.append(dM)
    # A4: correction curve by 5y bins
    curve = []
    for lo in range(50, 90, 5):
        msk = (y >= lo) & (y < lo + 5)
        if msk.sum() >= 5:
            curve.append(float(np.mean(base[msk] - a.cpu().numpy()[msk])))
    A4.append(curve)
    # B2: CLS geometry — spread ratio of representation radius L vs Y
    with torch.no_grad():
        def cls_of(Xpop):
            qq, aa, ss = scaled(mp, Xpop)
            mod = model.pool(qq).unsqueeze(-1)
            h = model.module_embed(mod) + model.pos_embed
            h = model.film(h, ss)
            h_all = torch.cat([model.cls_token.expand(len(qq), -1, -1), h], dim=1)
            for blk in model.blocks:
                h_all = blk(h_all, None)
            return model.norm(h_all)[:, 0]
        cL = cls_of(XL); cY = cls_of(X)
        rL = float(cL.std(0).norm()); rY = float(cY.std(0).norm())
    B2.append(rL / max(rY, 1e-9))
    print(f"probe seed{k+1}/25 ({time.time()-t0:.0f}s)", flush=True)
    del mp
    torch.cuda.empty_cache()

def sh(M):
    a, b = M[:13].mean(0), M[13:].mean(0)
    return round(float(spearmanr(a, b).statistic), 3)

keys60 = [int(i) for i in gidx_top]
C1m = np.array([[c[k] for k in keys60] for c in C1])
A2Fm = np.array(A2F); A2Mm = np.array(A2M)
C3m = np.array([[c[j] for j in range(64)] for c in C3])
gates = {"C1_lever_splithalf": sh(C1m),
         "C3_module_splithalf": sh(C3m),
         "A2_sexdiv_splithalf": sh(np.abs(A2Fm - A2Mm)),
         "A4_curve_splithalf": round(float(spearmanr(np.mean(A4[:13], 0), np.mean(A4[13:], 0)).statistic), 3),
         "B2_ratio_mean": round(float(np.mean(B2)), 3),
         "B2_frac_lt1": round(float(np.mean(np.array(B2) < 1.0)), 3)}
top_levers = [gene_sel[keys60[i]] for i in np.argsort(-C1m.mean(0))[:20]]
top_mods = [int(j) for j in np.argsort(-C3m.mean(0))[:8]]
json.dump({"arm": "v4-3_mono", "gates": gates,
           "C1_top_levers": top_levers,
           "C1_lever_years": {gene_sel[keys60[i]]: round(float(C1m.mean(0)[i]), 3) for i in np.argsort(-C1m.mean(0))[:20]},
           "C3_top_modules": top_mods,
           "A4_curve_by5y": np.round(np.mean(A4, 0), 3).tolist(),
           "A2_top_divergent": [gene_sel[keys60[i]] for i in np.argsort(-np.abs(A2Fm.mean(0) - A2Mm.mean(0)))[:20]],
           "B2_per_seed": [round(b, 3) for b in B2]},
          open(os.path.join(RUNS, "MINING_V4.json"), "w"), indent=1)
print("GATES:", json.dumps(gates, indent=1), flush=True)
print("-> MINING_V4.json", flush=True)
