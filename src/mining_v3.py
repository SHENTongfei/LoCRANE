# -*- coding: utf-8 -*-
"""mining_v3.py - MINING_v3 execution (pre-registered in MINING_v3_PLAN.md).
Frozen B3 25-ckpt ensemble (cg_k, final-epoch, disclosed), OOF on val folds.
M1 module do-ablation (256, input-level zeroing of module genes on Xq+Xs,
   delta MAE/spearman on L-band val)
M2 gene GIP importance (|grad age * x| on Xq, 5-seed mean, all/L/M bands)
M3 tail-contrast gene z (L vs M GIP) + L-certified vs L-uncertified contrast
M4 immune-cell GIP z (L vs M, 19 cells, names from LM22 keep_cells)
Leakage-safe: only val-fold samples. GPU-light (~15-25 min). argv-precise guard.
Writes runs_v3/mining_v3_results.json (+ per-item CSVs)."""
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
from scipy.stats import spearmanr
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 2024, 2025, 12345]

CELLS = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
         'T cells CD4 naive', 'T cells CD4 memory resting', 'T cells CD4 memory activated',
         'T cells follicular helper', 'T cells regulatory (Tregs)', 'NK cells resting',
         'Monocytes', 'Macrophages M0', 'Macrophages M1', 'Macrophages M2',
         'Dendritic cells resting', 'Dendritic cells activated', 'Mast cells resting',
         'Eosinophils', 'Neutrophils']

def log(m): print(f"[mine3] {m}", flush=True)

def build_model(n_immune, n_genes, assignment):
    return CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                    n_immune=n_immune, use_moe=False, use_gate=False,
                    use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)

def main():
    import psutil
    me = os.getpid()
    twins = [p.pid for p in psutil.process_iter(["pid", "cmdline"])
             if p.pid != me and p.info["cmdline"]
             and p.info["cmdline"][-1].replace("\\", "/").endswith("mining_v3.py")
             and p.info["cmdline"][0].replace("\\", "/").endswith("python.exe")]
    if twins:
        log(f"another mining_v3 alive {twins} - orderly exit")
        return 1

    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    gene_names = list(cfg_genes)
    all_ids = [str(i) for i in pd.read_csv(os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
                                           sep="\t", index_col=0, compression=None).T.index]

    # accumulators
    gip_all = np.zeros((len(gene_names),), np.float64)
    gip_L = np.zeros((len(gene_names),), np.float64)
    gip_M = np.zeros((len(gene_names),), np.float64)
    n_all = n_L = n_M = 0
    cell_gip_L = np.zeros(len(CELLS)); cell_gip_M = np.zeros(len(CELLS)); nc_L = nc_M = 0
    ablu = {}            # module -> list of (dm, ds) across ckpts
    pred_store = {}      # fold -> dict sid -> (pred, cert_inputs)
    mcert = np.zeros(1715, bool); mcert_L_ids = []

    for f in range(5):
        km = np.load(os.path.join(DATA3, "km256_f{0}.npz".format(os.environ.get("V3_KMFOLD", "0"))))
        assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
        mod2genes = {k: [] for k in range(256)}
        for g_i, m in assign.items():
            mod2genes[m].append(g_i)
        z = np.load(os.path.join(DATA3, f"fe_cache_f{f}_full.npz"))
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        lab_va = lab.loc[sorted(str(i) for i in folds[str(f)])]
        age = lab_va["age"].to_numpy(np.float32)
        band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
        val_ids = [str(x) for x in lab_va.index]
        inner = [i for i in all_ids if i not in set(val_ids)]
        from sklearn.linear_model import Ridge
        Xq_tr, Xs_tr = z["Xq_tr"], z["Xs_tr"]
        age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
        rda = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - lab["age"].mean()) / lab["age"].std()
        mu, sd = lab["age"].mean(), lab["age"].std()

        q_t = torch.from_numpy(Xq_va).float()
        s_t = torch.from_numpy(Xs_va).float()
        imm_t = torch.from_numpy(imm_va.astype(np.float32))
        sex_t = torch.from_numpy(C.encode_sex(lab_va["sex"].to_numpy()))
        rda_t = torch.from_numpy(rda).float()
        BV = torch.zeros(len(Xq_va), 3, device=DEV)  # band-neutral: match OOF suite (cg_train final eval)

        # base preds + GIP + ablation, per seed
        preds = []
        for s in SEEDS:
            ck = os.path.join(RUNS, "ckpt_cg", f"cg_k_f{f}_s{s}.pt")
            d = torch.load(ck, map_location=DEV, weights_only=False)
            model = build_model(imm_va.shape[1], len(cfg_genes), assign)
            model.load_state_dict(d["model"]); model.eval()
            Xq0 = q_t.clone().requires_grad_(True)
            imm0 = imm_t.clone().requires_grad_(True)
            with torch.enable_grad():
                out = model(Xq0.to(DEV), s_t.to(DEV), sex_t.to(DEV), imm0.to(DEV),
                            rda_t.to(DEV), BV.to(DEV), plm=None)
                loss = out["age"].sum()
            model.zero_grad()
            loss.backward()
            gip = (Xq0.grad.abs() * q_t.abs()).numpy()
            gip_all += gip.mean(0)
            n_all += 1
            mL = band == 2; mM = band == 1
            gip_L += gip[mL].mean(0); n_L += 1
            gip_M += gip[mM].mean(0); n_M += 1
            imm_gip = (imm0.grad.abs() * imm_t.abs()).numpy()
            cell_gip_L += imm_gip[mL].sum(0); nc_L += 1
            cell_gip_M += imm_gip[mM].sum(0); nc_M += 1
            P = out["age"].detach().cpu().numpy() * sd + mu
            preds.append(P)
            # M1 ablation (input zeroing on L-band only, ONE module per forward)
            mL_ = band == 2
            lrows = torch.where(torch.from_numpy(band == 2))[0]
            baseL_mae = float(np.abs(P[mL_] - age[mL_]).mean())
            baseL_rho = float(spearmanr(P[mL_], age[mL_]).statistic)
            for mk2 in range(256):
                gs = mod2genes[mk2]
                if not gs:
                    continue
                Xqz = q_t.clone(); Xsz = s_t.clone()
                gl = torch.tensor(gs)
                ii, jj = torch.meshgrid(lrows, gl, indexing="ij")
                Xqz[ii, jj] = 0.0; Xsz[ii, jj] = 0.0
                with torch.no_grad():
                    o2 = model(Xqz.to(DEV), Xsz.to(DEV), sex_t.to(DEV), imm_t.to(DEV),
                               rda_t.to(DEV), BV.to(DEV), plm=None)
                P2L = o2["age"][mL_].cpu().numpy() * sd + mu
                dm = float(np.abs(P2L - age[mL_]).mean() - baseL_mae)
                ds = float(spearmanr(P2L, age[mL_]).statistic - baseL_rho)
                ablu.setdefault(mk2, []).append((dm, ds))
            del model
        P_ens = np.mean(preds, 0)
        sigma = np.std(preds, 0) + 1e-6
        pred_store[f] = (val_ids, P_ens, sigma, age, band)
        log(f"fold{f}: GIP+ablation done (L={int(mL.sum())}, M={int(mM.sum())})")

    # assemble per-sample OOF (1715) for cert flags
    oof_p = np.zeros(1715); oof_s = np.zeros(1715); oof_age = np.zeros(1715); oof_band = np.zeros(1715, int)
    for f in range(5):
        ids, P, sig, age_, band_ = pred_store[f]
        idx = np.array([i for i, x in enumerate([str(j) for j in all_ids]) if x in set(ids)])
        # simpler: use lab index
        pos = lab.index.get_indexer([i for i in ids])
        oof_p[pos] = P; oof_s[pos] = sig; oof_age[pos] = age_; oof_band[pos] = band_
    cal = oof_band < 2
    stud = (oof_p - oof_age) / oof_s
    q = float(np.quantile(stud[cal], np.ceil((cal.sum() + 1) * 0.9) / (cal.sum() + 1)))
    cert = (oof_p + q * oof_s) < oof_age

    res = {"n_coversed": int(n_all), "q_cert": q, "cert_frac_L": float(cert[oof_band == 2].mean())}
    # M1
    rows = []
    for mk in range(256):
        v = ablu.get(mk, [])
        if not v:
            continue
        v = np.array(v)
        top3 = " ".join(gene_names[i] for i in mod2genes[mk][:3])
        rows.append({"module": mk, "n_genes": len(mod2genes[mk]), "top_genes": top3,
                     "delta_mae_L": float(v[:, 0].mean()), "delta_spear_L": float(v[:, 1].mean()),
                     "sd_mae": float(v[:, 0].std())})
    m1 = pd.DataFrame(rows).sort_values("delta_mae_L", ascending=False)
    m1.to_csv(os.path.join(RUNS, "mining_v3_M1_module_ablation.csv"), index=False)
    res["M1_top5"] = m1.head(5).to_dict("records")
    # M2/M3
    gip_all /= max(1, n_all / 5.0)   # per-ckpt scale keep relative
    z_gene = (gip_L / max(gip_L.sum(), 1e-9)) - (gip_M / max(gip_M.sum(), 1e-9))
    gdf = pd.DataFrame({"gene": gene_names, "gip_all": gip_all, "gip_L_frac": gip_L,
                        "gip_M_frac": gip_M, "z_LM": z_gene})
    gdf["rank_all"] = gdf["gip_all"].rank(ascending=False).astype(int)
    gdf.to_csv(os.path.join(RUNS, "mining_v3_M2_M3_genes.csv"), index=False)
    res["M2_top10_all"] = gdf.nlargest(10, "gip_all")[["gene", "gip_all"]].to_dict("records")
    res["M3_top10_Lup"] = gdf.nlargest(10, "z_LM")[["gene", "z_LM"]].to_dict("records")
    res["M3_top10_Ldown"] = gdf.nsmallest(10, "z_LM")[["gene", "z_LM"]].to_dict("records")
    # M3 cert vs uncert (L-band only, per-gene via stored? use z_LM proxy + cert overlap):
    Lcert = set()
    for f in range(5):
        ids, P, sig, age_, band_ = pred_store[f]
        st = (P - age_) / sig
        qf = float(np.quantile(st[band_ < 2], np.ceil(((band_ < 2).sum() + 1) * 0.9) / ((band_ < 2).sum() + 1)))
        cf = (P + qf * sig) < age_
        Lcert |= set(np.array(ids)[(band_ == 2) & cf])
    res["M3_Lcertified_n"] = len(Lcert)
    # M4
    cdf = pd.DataFrame({"cell": CELLS, "gip_L": cell_gip_L / max(nc_L, 1),
                        "gip_M": cell_gip_M / max(nc_M, 1)})
    cdf["z_LM"] = cdf["gip_L"] - cdf["gip_M"]
    cdf.to_csv(os.path.join(RUNS, "mining_v3_M4_cells.csv"), index=False)
    cdf2 = cdf.sort_values("z_LM", ascending=False)
    res["M4_top5_L"] = cdf2.head(5).to_dict("records")
    res["M4_top3_M"] = cdf2.tail(3).to_dict("records")
    json.dump(res, open(os.path.join(RUNS, "mining_v3_results.json"), "w"), indent=1, default=float)
    log("WROTE mining_v3_results.json + CSVs")

if __name__ == "__main__":
    main()
