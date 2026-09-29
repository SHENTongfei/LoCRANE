# -*- coding: utf-8 -*-
"""h2_full_repr.py - H2 L2 escalation: full-cohort (1715, L=624) pace-ratio
in representation space, from the Phase-2 ckpt chain (cg_ckpt_chain.py,
prefix cg_k, CG_SAVECKPT=1). Per sample: cls ensemble = mean of the 5 seed
models on its val fold. Estimators + CIRCULAR-SHIFT null (L1-fixed design,
pool H2 p-values were invalidated by the global-permutation null bias).
M-band upweighted variant: fine M bins 70/75/80/85/90.
Model note: FINAL-epoch states (not best-val) — disclosed.
Writes runs_v3/h2_full_results.json + runs_v3/repr_full_b3tail.npy.
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CRANE_DATA_DIR", "C:/Users/TS/Desktop/crane")  # same as h2_repr_pace
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(HERE, "..", "data_v3")
RUNS = os.path.join(HERE, "..", "runs_v3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 7, 2024, 2025, 12345]
RNG = np.random.default_rng(20260930)
N_SHIFT = 2000

def log(m): print(f"[h2full] {m}", flush=True)

def build_model(n_immune, n_genes, assignment):
    return CraneZV3(n_genes=n_genes, assignment=assignment, n_modules=256,
                    n_immune=n_immune, use_moe=False, use_gate=False,
                    use_contra=False, n_plm=0, plm_bottleneck=128).to(DEV)

def main():
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, "folds_v3_full.json")))
    cache = os.path.join(RUNS, "repr_full_b3tail.npz")
    if os.path.exists(cache):
        zc = np.load(cache)
        repr_all, ids, age = zc["repr"], zc["ids"], zc["age"]
        log(f"CACHE HIT {os.path.basename(cache)} - skip 25-ckpt extraction")
        main_test(repr_all, ids, age)
        return
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    cfg_genes = C.select_hvg(p_all, 3000)
    all_ids = [str(i) for i in pd.read_csv(
        os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"),
        sep="\t", index_col=0, compression=None).T.index]
    reps = {}
    for f in range(5):
        km = np.load(os.path.join(DATA3, f"km256_f{os.environ.get('V3_KMFOLD', '0')}.npz"))
        assign = {int(k): int(v) for k, v in zip(km["g"], km["m"])}
        z = np.load(os.path.join(DATA3, f"fe_cache_f{f}_full.npz"))
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        lab_va = lab.loc[sorted(str(i) for i in folds[str(f)])]
        age_va = lab_va["age"].to_numpy(np.float32)
        mu, sd = lab_va["age"].mean(), max(1e-6, lab_va["age"].std())
        val_set = set(lab_va.index)
        inner = [i for i in all_ids if i not in val_set]
        Xq_tr, Xs_tr, imm_tr = z["Xq_tr"], z["Xs_tr"], z["imm_tr"]
        age_tr = lab.loc[inner, "age"].to_numpy(np.float32)[: Xq_tr.shape[0]]
        rda_va = (Ridge(alpha=1.0).fit(Xq_tr, age_tr).predict(Xq_va) - lab["age"].mean()) / lab["age"].std()
        BV = torch.zeros(len(Xq_va), 3, device=DEV)  # band-neutral: raw repr for drift
        XVq = torch.from_numpy(Xq_va).float().to(DEV); XVs = torch.from_numpy(Xs_va).float().to(DEV)
        SV = torch.from_numpy(C.encode_sex(lab_va["sex"].to_numpy())).to(DEV)
        IV = torch.from_numpy(imm_va.astype(np.float32)).to(DEV)
        RV = torch.from_numpy(rda_va).float().to(DEV)
        acc = []
        for s in SEEDS:
            ck = os.path.join(RUNS, "ckpt_cg", f"cg_k_f{f}_s{s}.pt")
            d = torch.load(ck, map_location=DEV, weights_only=False)
            model = build_model(imm_va.shape[1], len(cfg_genes), assign)
            model.load_state_dict(d["model"]); model.eval()
            with torch.no_grad():
                out = model(XVq, XVs, SV, IV, RV, BV, plm=None)
            acc.append(out["cls"].cpu().numpy())
            del model
        reps[f] = (np.array(lab_va.index.astype(str)), np.mean(acc, axis=0))
        log(f"fold{f}: {reps[f][0].shape[0]} samples x {reps[f][1].shape[1]}d")
    ids = np.concatenate([reps[f][0] for f in range(5)])
    repr_all = np.concatenate([reps[f][1] for f in range(5)])
    age = lab.loc[ids, "age"].to_numpy(np.float64)
    np.savez(os.path.join(RUNS, "repr_full_b3tail.npz"),
             repr=repr_all, ids=ids.astype("U40"), age=age)
    assert len(ids) == 1715

    main_test(repr_all, ids, age)

def main_test(repr_all, ids, age):
    def bin_ratio(edges_m, edges_l):
        def vels(edges):
            seg = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (age >= lo) & (age < hi)
                if m.sum() >= 3:
                    seg.append(((lo + hi) / 2, repr_all[m].mean(0), float(hi - lo)))
            return [float(np.linalg.norm(b - a)) / sp
                    for (ma, a, sa), (mb, b, sp) in zip(seg[:-1], seg[1:])]
        vm, vl = vels(edges_m), vels(edges_l)
        return float(np.mean(vl) / np.mean(vm)) if vm and vl else None

    def circ_null(obs, edges_m, edges_l):
        order = np.argsort(age, kind="stable")
        rs, as_ = repr_all[order], age[order]
        nulls = []
        for _ in range(N_SHIFT):
            k = int(RNG.integers(1, len(rs)))
            v = bin_ratio_ns(np.roll(rs, k, axis=0), as_, edges_m, edges_l)
            if v is not None and np.isfinite(v):
                nulls.append(v)
        nulls = np.array(nulls)
        return float((nulls <= obs).mean()), float((np.abs(nulls - 1) >= abs(obs - 1)).mean()), float(nulls.mean())

    def bin_ratio_ns(rs, as_, em, el):
        def vels(edges):
            seg = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (as_ >= lo) & (as_ < hi)
                if m.sum() >= 3:
                    seg.append(((lo + hi) / 2, rs[m].mean(0), float(hi - lo)))
            return [float(np.linalg.norm(b - a)) / sp
                    for (ma, a, sa), (mb, b, sp) in zip(seg[:-1], seg[1:])]
        vm, vl = vels(em), vels(el)
        return float(np.mean(vl) / np.mean(vm)) if vm and vl else None

    out = {"n": int(len(ids)),
           "nM_70_90": int(((age >= 70) & (age < 90)).sum()),
           "nL_90p": int((age >= 90).sum())}
    for tag, em in [("std", [70, 80, 90]), ("fine_m", [70, 75, 80, 85, 90])]:
        obs = bin_ratio(em, [90, 97, 105])
        out[f"obs_{tag}"] = obs
        if obs is not None:
            p1, p2, nm = circ_null(obs, em, [90, 97, 105])
            out.update({f"p_circ_one_{tag}": p1, f"p_circ_two_{tag}": p2, f"null_mean_{tag}": nm})
        log(f"{tag}: obs={obs} " + (f"p_one={out.get(f'p_circ_one_{tag}')}" if obs is not None else ""))
    with open(os.path.join(RUNS, "h2_full_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log("WROTE h2_full_results.json")

if __name__ == "__main__":
    main()
