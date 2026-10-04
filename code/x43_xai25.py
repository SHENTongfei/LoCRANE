# -*- coding: utf-8 -*-
"""x43_xai25.py — v4 winner 25-seed ensemble attribution (x24 machinery + 2 NEW axes).
Axes: M1 delta-IG, M2Y/M2L band-L IG, M3 FiLM divergence, M4 module do-ablation,
      M7 pairwise-order IG (NEW, tied to monotonicity prior),
      M8 correction-budget IG (NEW, tied to sparsity prior).
Gates (pre-registered, same as x24): split-half rho>=0.6 at gene & module level;
      attribution must exceed p99 of 5 permuted-label seeds.
ARM env var selects winner config (default v4-3_mono)."""
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
RUNS = os.path.join(HERE, "..", "runs_v4"); os.makedirs(RUNS, exist_ok=True)
CRANE = os.environ.get("CRANE_DATA_DIR", os.path.join(HERE, "..", "data"))
sys.path.insert(0, HERE)
from model_v3 import CraneZV3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARM = os.environ.get("V4_ARM", "v4-3_mono")
USE_MONO = ARM in ("v4-3_mono", "v4-7_combo")
USE_SPARSE = ARM == "v4-7_combo"
LAM_MON, LAM_SP = 0.05, 0.05

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
print(f"YC n={len(y)} L n={len(yL)} ARM={ARM} DEV={DEV}", flush=True)

SEEDS = [42, 7, 2024, 2025, 12345, 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 131, 139, 147, 155]

def band_of(a):
    return np.where(a <= 50, 0, np.where(a < 90, 1, 2)).astype(np.int64)

def train_seed(seed, y_target=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    yt = y if y_target is None else y_target
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X).astype(np.float32)
    kf = KFold(5, shuffle=True, random_state=seed)
    anchor_raw = np.zeros(len(yt))
    for itr, ivl in kf.split(np.arange(len(yt))):
        en_i = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs[itr], yt[itr])
        anchor_raw[ivl] = en_i.predict(Xs[ivl])
    rg_full = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(Xs, yt)
    mu, sd = float(yt.mean()), max(1e-6, float(yt.std()))
    anchor = ((anchor_raw - mu) / sd).astype(np.float32)
    assign = {int(i): int(i * 64 // 500) for i in range(500)}
    model = CraneZV3(n_genes=500, assignment=assign, n_modules=64, n_immune=0,
                     d_model=64, n_heads=4, n_layers=2, ff_dim=128, ridge_residual=1.0).to(DEV)
    tq = torch.from_numpy(Xs).to(DEV)
    ty = torch.from_numpy(((yt - mu) / sd).astype(np.float32)).to(DEV)
    ta = torch.from_numpy(anchor).to(DEV)
    tsx = torch.from_numpy(sex_all).to(DEV)
    tband = torch.from_numpy(band_of(yt)).to(DEV)
    Xlq = torch.from_numpy(sc.transform(XL).astype(np.float32)).to(DEV)
    alq = torch.from_numpy(((rg_full.predict(sc.transform(XL)) - mu) / sd).astype(np.float32)).to(DEV)
    slq = torch.from_numpy(sexL).to(DEV)
    blq = torch.from_numpy(band_of(yL)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best_loss, best_state, pat = 1e9, None, 0
    for ep in range(150):
        model.train()
        perm = torch.randperm(len(yt), device=DEV)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            lidx = torch.randint(0, len(yL), (len(idx),), device=DEV)
            xq = torch.cat([tq[idx], Xlq[lidx]])
            sx = torch.cat([tsx[idx], slq[lidx]])
            rz = torch.cat([ta[idx], alq[lidx]])
            out = model(xq, xq, sx, ridge_z=rz)
            band_t = torch.cat([tband[idx], blq[lidx]])
            loss = F.l1_loss(out["age"][:len(idx)], ty[idx]) + 0.3 * F.cross_entropy(out["band_logits"], band_t)
            if USE_MONO and len(idx) > 8:
                pi = torch.randint(0, len(idx), (len(idx),), device=DEV)
                pj = torch.randint(0, len(idx), (len(idx),), device=DEV)
                sgn = torch.sign(ty[idx][pi] - ty[idx][pj])
                m = sgn != 0
                l_mon = F.softplus(-sgn[m] * (out["age"][:len(idx)][pi][m] - out["age"][:len(idx)][pj][m])).mean()
                loss = loss + LAM_MON * l_mon
            if USE_SPARSE:
                loss = loss + LAM_SP * ((out["age"][:len(idx)] - ta[idx]) ** 2).mean()
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

def _pack_inputs(raw_pack, Xpop, sex_pop):
    model, sc, rg_full, mu, sd = raw_pack
    Xn = sc.transform(Xpop).astype(np.float32)
    an = ((rg_full.predict(Xn) - mu) / sd).astype(np.float32)
    q_all = torch.from_numpy(Xn).to(DEV)
    a_all = torch.from_numpy(an).to(DEV)
    sx_all = torch.as_tensor(sex_pop, dtype=torch.float32, device=DEV)
    return model, q_all, a_all, sx_all

def ig_vec(raw_pack, Xpop, sex_pop, target, steps=48, batch=384):
    model, q_all, a_all, sx_all = _pack_inputs(raw_pack, Xpop, sex_pop)
    acc = torch.zeros_like(q_all)
    for st in range(1, steps + 1):
        alpha = st / steps
        for i in range(0, len(q_all), batch):
            q = (alpha * q_all[i:i+batch]).clone().requires_grad_(True)
            out = model(q, q, sx_all[i:i+batch], ridge_z=a_all[i:i+batch])
            if target == "delta":
                score = (model.ridge_residual * torch.tanh(model.reg_head(
                    torch.cat([out["cls"], model.reg_gene_proj(q), a_all[i:i+batch].unsqueeze(-1)], -1)).squeeze(-1))).sum()
            elif target == "delta_sq":
                d = model.ridge_residual * torch.tanh(model.reg_head(
                    torch.cat([out["cls"], model.reg_gene_proj(q), a_all[i:i+batch].unsqueeze(-1)], -1)).squeeze(-1))
                score = (d ** 2).sum()
            else:
                score = out["band_logits"][:, 2].sum()
            g, = torch.autograd.grad(score, q)
            acc[i:i+batch] += g.detach()
    return np.abs((q_all * acc / steps).cpu().numpy()).mean(axis=0)

def ig_pairwise(raw_pack, n_pairs=6000, steps=32, seed=0):
    """NEW M7: IG of the pairwise score (pred_i - pred_j) wrt both inputs,
    averaged over age-ordered random pairs -> genes carrying ORDER information."""
    model, q_all, a_all, sx_all = _pack_inputs(raw_pack, X, sex_all)
    rng = np.random.default_rng(seed)
    order = np.argsort(y)
    hi = order[rng.integers(len(order) // 2, len(order), n_pairs)]
    lo = order[rng.integers(0, len(order) // 2, n_pairs)]
    qi = q_all[hi]; qj = q_all[lo]; ai = a_all[hi]; aj = a_all[lo]; si = sx_all[hi]; sj = sx_all[lo]
    acc_i = torch.zeros_like(qi); acc_j = torch.zeros_like(qj)
    B = 256
    for st in range(1, steps + 1):
        alpha = st / steps
        for b in range(0, len(qi), B):
            qqi = (alpha * qi[b:b+B]).clone().requires_grad_(True)
            qqj = (alpha * qj[b:b+B]).clone().requires_grad_(True)
            out_i = model(qqi, qqi, si[b:b+B], ridge_z=ai[b:b+B])
            out_j = model(qqj, qqj, sj[b:b+B], ridge_z=aj[b:b+B])
            def dlt(o, qq, aa):
                return model.ridge_residual * torch.tanh(model.reg_head(
                    torch.cat([o["cls"], model.reg_gene_proj(qq), aa.unsqueeze(-1)], -1)).squeeze(-1))
            score = (dlt(out_i, qqi, ai[b:b+B]) - dlt(out_j, qqj, aj[b:b+B])).sum()
            gi, = torch.autograd.grad(score, qqi, retain_graph=True)
            gj, = torch.autograd.grad(score, qqj)
            acc_i[b:b+B] += gi.detach(); acc_j[b:b+B] += gj.detach()
    v = (np.abs((qi * acc_i / steps).cpu().numpy()) + np.abs((qj * acc_j / steps).cpu().numpy())).mean(axis=0)
    return v

def film_div(model):
    with torch.no_grad():
        gbs = {}
        for sexv, tag in ((0.0, "M"), (1.0, "F")):
            gb = model.film.mlp(torch.tensor([[sexv]], dtype=torch.float32, device=DEV))
            gb = gb.view(-1, 2, model.film.n_modules, model.film.rank)
            gb = (gb.reshape(-1, model.film.rank) @ model.film.basis.t()).view(-1, 2, model.n_modules, model.d_model)
            gbs[tag] = gb[0, 0].cpu().numpy()
    return np.abs(gbs["F"] - gbs["M"]).mean(axis=1)

def age_zero_mod(raw_pack, Xpop, sex_pop, zm):
    model, sc, rg_full, mu, sd = raw_pack
    Xn = sc.transform(Xpop).astype(np.float32)
    an = ((rg_full.predict(Xn) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        q = torch.from_numpy(Xn).to(DEV)
        a = torch.from_numpy(an).to(DEV)
        sx = torch.as_tensor(sex_pop, dtype=torch.float32, device=DEV)
        mod = model.pool(q).unsqueeze(-1)
        h = model.module_embed(mod) + model.pos_embed
        h = model.film(h, sx)
        if zm is not None:
            h = h.clone(); h[:, zm] = 0.0
        h_all = torch.cat([model.cls_token.expand(len(q), -1, -1), h], dim=1)
        for blk in model.blocks:
            h_all = blk(h_all, None)
        h_all = model.norm(h_all)
        cls = h_all[:, 0] + model.gene_proj(q)
        delta = model.reg_head(torch.cat([cls, model.reg_gene_proj(q), a.unsqueeze(-1)], -1)).squeeze(-1)
        return (a + model.ridge_residual * torch.tanh(delta)).cpu().numpy() * sd

# ---------------- 25 seeds ----------------
t0 = time.time()
M1, M2Y, M2L, D3, M7, M8 = [], [], [], [], [], []
for k, s in enumerate(SEEDS):
    raw = train_seed(s)
    M1.append(ig_vec(raw, X, sex_all, "delta"))
    M2Y.append(ig_vec(raw, X, sex_all, "L"))
    M2L.append(ig_vec(raw, XL, sexL, "L"))
    D3.append(film_div(raw[0]))
    M7.append(ig_pairwise(raw, seed=s))
    M8.append(ig_vec(raw, X, sex_all, "delta_sq"))
    print(f"seed{k+1}/25 done ({time.time()-t0:.0f}s)", flush=True)
    del raw
    torch.cuda.empty_cache()
M1 = np.stack(M1); M2Y = np.stack(M2Y); M2L = np.stack(M2L); D3 = np.stack(D3); M7 = np.stack(M7); M8 = np.stack(M8)

# ---------------- permuted null (5 seeds) ----------------
rng = np.random.default_rng(99)
N1, N7 = [], []
for s in (1001, 1002, 1003, 1004, 1005):
    raw = train_seed(s, y_target=rng.permutation(y))
    N1.append(ig_vec(raw, X, sex_all, "delta"))
    N7.append(ig_pairwise(raw, seed=s))
    print(f"null seed{s} done ({time.time()-t0:.0f}s)", flush=True)
    del raw
    torch.cuda.empty_cache()
N1 = np.stack(N1); N7 = np.stack(N7)

# save FIRST — verdict code must never be able to destroy 16 min of attribution
np.savez(os.path.join(RUNS, "xai25_attribution.npz"), M1=M1, M2Y=M2Y, M2L=M2L, D3=D3, M7=M7, M8=M8,
         N1=N1, N7=N7, gene_sel=gene_sel, allow_pickle=True)
print("npz saved", flush=True)

BINS64 = [slice(i * 500 // 64, (i + 1) * 500 // 64) for i in range(64)]
def split_half(M, level):
    a, b = M[:13].mean(0), M[13:].mean(0)
    if level == "gene":
        return spearmanr(a, b).statistic
    if level == "mod":
        am = np.stack([np.array([M[i][sl].mean() for sl in BINS64]) for i in range(len(M))])
        return spearmanr(am[:13].mean(0), am[13:].mean(0)).statistic
    return spearmanr(M[:13].mean(0), M[13:].mean(0)).statistic

def null_p99_count(M, N, level="gene"):
    if level == "gene":
        m = M.mean(0); thr = np.percentile(np.abs(N.mean(0)), 99)
        return int((m > thr).sum()), float(thr)
    return -1, np.nan

gates = {
    "M1_delta_gene_splithalf": round(float(split_half(M1, "gene")), 3),
    "M1_delta_mod_splithalf": round(float(split_half(M1, "mod")), 3),
    "M2L_gene_splithalf": round(float(split_half(M2L, "gene")), 3),
    "M2Y_gene_splithalf": round(float(split_half(M2Y, "gene")), 3),
    "M3_film_splithalf": round(float(split_half(D3, "vec")), 3),
    "M7_pair_gene_splithalf": round(float(split_half(M7, "gene")), 3),
    "M8_budget_gene_splithalf": round(float(split_half(M8, "gene")), 3),
}
c1, t1 = null_p99_count(M1, N1); c7, t7 = null_p99_count(M7, N7)
gates["M1_genes_over_null_p99"] = c1
gates["M7_genes_over_null_p99"] = c7

np.savez(os.path.join(RUNS, "xai25_attribution.npz"), M1=M1, M2Y=M2Y, M2L=M2L, D3=D3, M7=M7, M8=M8,
         N1=N1, N7=N7, gene_sel=gene_sel, allow_pickle=True)
json.dump({"arm": ARM, "gates": gates,
           "M1_top30": [g for g in gene_sel[np.argsort(-M1.mean(0))][:30]],
           "M7_top30": [g for g in gene_sel[np.argsort(-M7.mean(0))][:30]],
           "M8_top30": [g for g in gene_sel[np.argsort(-M8.mean(0))][:30]],
           "M2L_top20": [g for g in gene_sel[np.argsort(-M2L.mean(0))][:20]],
           "M4_note": "module do-ablation moved to x44 (uses saved npz models? no - retrain there)"},
          open(os.path.join(RUNS, "x43_xai25.json"), "w"), indent=1)
print("GATES:", json.dumps(gates, indent=1), flush=True)
print("-> x43_xai25.json", flush=True)
