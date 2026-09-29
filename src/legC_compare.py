# -*- coding: utf-8 -*-
"""legC_compare.py - Leg C strict parity: eager train_v3 (stage0_cgpar, best-val
ckpt) vs CUDA-graph bucketed cg_train (cg_full, final-epoch state), fold0 seed42
full cohort. Pre-registered tolerances (documented here, chosen BEFORE reading
the numbers):
  PASS = pair_spearman(M_eager, M_cg) >= 0.995 AND |MAE_eager - MAE_cg| <= 0.20
Disclosed caveat: best-val state (eager) vs final-epoch state (cg) - cosine LR
plateau makes the two close but not identical; the tolerance covers this.
GPU-light (one forward pass over 345 val samples). Writes runs_v3/legC_results.json.
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    km = np.load(os.path.join(DATA3, "km256_f0.npz"))
    assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
    z = np.load(os.path.join(DATA3, "fe_cache_f0_full.npz"))
    Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
    Xq_tr, Xs_tr = z["Xq_tr"], z["Xs_tr"]
    lab_va = lab.loc[sorted(str(i) for i in folds["0"])]
    age = lab_va["age"].to_numpy(np.float64)
    all_ids = [str(i) for i in pd.read_csv(os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
                                           sep="\t", index_col=0, compression=None).T.index]
    inner = [i for i in all_ids if i not in set(lab_va.index)]
    age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
    rda = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - lab["age"].mean()) / lab["age"].std()
    mu, sd = lab["age"].mean(), lab["age"].std()

    def eval_ckpt(ckpath):
        d = torch.load(ckpath, map_location=DEV, weights_only=False)
        model = CraneZV3(n_genes=len(cfg_genes), assignment=assign, n_modules=256,
                          n_immune=imm_va.shape[1], use_moe=False, use_gate=False,
                          use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)
        model.load_state_dict(d["model"]); model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(Xq_va).float().to(DEV),
                        torch.from_numpy(Xs_va).float().to(DEV),
                        torch.from_numpy(C.encode_sex(lab_va["sex"].to_numpy())).to(DEV),
                        torch.from_numpy(imm_va.astype(np.float32)).to(DEV),
                        torch.from_numpy(rda).float().to(DEV),
                        torch.zeros(len(Xq_va), 3, device=DEV), plm=None)
        M = out["age"].cpu().numpy() * sd + mu
        return M, d.get("seed", None)

    eager_ckpt = os.path.join(RUNS, "ckpt", "stage0_cgpar_f0_s42_best.pt")
    assert os.path.exists(eager_ckpt), f"eager ckpt missing: {eager_ckpt} (run Leg C eager step first)"
    M_eager, _ = eval_ckpt(eager_ckpt)

    zc = np.load(os.path.join(RUNS, "cg_full_M_f0_s42.npz"), allow_pickle=True)
    ids_npz = [str(x) for x in zc["ids"]]
    true_ids = [t[:16] for t in sorted(str(i) for i in folds["0"])]
    assert ids_npz == true_ids, "cg npz row-order misaligned (AAR #6 guard)"
    M_cg = zc["M"].astype(np.float64)

    res = {"n_val": int(len(lab_va)),
           "MAE_eager": float(np.abs(M_eager - age).mean()),
           "MAE_cg": float(np.abs(M_cg - age).mean()),
           "MAE_diff": float(abs(np.abs(M_eager - age).mean() - np.abs(M_cg - age).mean())),
           "rho_eager": float(spearmanr(M_eager, age).statistic),
           "rho_cg": float(spearmanr(M_cg, age).statistic),
           "pair_spearman": float(spearmanr(M_eager, M_cg).statistic),
           "demean_corr": float(np.corrcoef(M_eager - M_eager.mean(), M_cg - M_cg.mean())[0, 1]),
           "note": "eager=best-val state / cg=final-epoch state (cosine plateau); tol: pair_spearman>=0.995 & |dMAE|<=0.20"}
    res["PASS"] = bool(res["pair_spearman"] >= 0.995 and res["MAE_diff"] <= 0.20)
    json.dump(res, open(os.path.join(RUNS, "legC_results.json"), "w"), indent=1, default=float)
    print("[legC]", json.dumps(res, default=float), flush=True)

if __name__ == "__main__":
    main()
