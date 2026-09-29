# -*- coding: utf-8 -*-
"""h2_l1_rescue.py - H2 L1 rescue on cached 324-pool repr (CPU-only, no GPU).

Two fixes over the flawed v1 null:
  (1) CIRCULAR-SHIFT null: sort samples by chronological age, rotate the age
      labels by k positions. Preserves the marginal age distribution AND the
      local age->repr gradient structure; breaks only the exact correspondence.
      The v1 global chrono-permutation destroyed both, so its null ratios
      clustered low (null_mean 0.375 vs obs 0.675 -> p_perm_less 0.937,
      uninformative direction).
  (2) M-band upweighting variant: finer M bins 70-75/75-80/80-85/85-90 so the
      starved M leg (n=18) contributes 4 velocity segments instead of 2.
Two-sided p on |ratio-1| (deceleration = ratio<1). Also per-segment
velocities for the record. Writes runs_v3/h2_l1_results.json.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
DATA3 = os.path.join(HERE, "..", "data_v3")
RNG = np.random.default_rng(20260930)
N_SHIFT = 2000

def log(m): print(f"[h2l1] {m}", flush=True)

def load():
    r = np.load(os.path.join(RUNS, "repr_b3tail.npy"))
    o = np.load(os.path.join(RUNS, "oof_predictions_b3tail.npz"))
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    ids = [str(x) for x in o["ids"]]
    age = lab.loc[ids, "age"].to_numpy(np.float64)
    return r, age

def bin_vel_ratio(repr, ages, m_edges, l_edges):
    def seg_means(edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ages >= lo) & (ages < hi)
            if m.sum() >= 3:
                out.append(((lo + hi) / 2.0, repr[m].mean(0), float(hi - lo)))
        return out
    vm = seg_means(m_edges); vl = seg_means(l_edges)
    if len(vm) < 1 or len(vl) < 1:
        return None, None, None
    # velocity between consecutive centroids (per-year displacement)
    vmv = [float(np.linalg.norm(b - a)) / sp for (ma, a, sa), (mb, b, sp) in zip(vm[:-1], vm[1:])]
    vlv = [float(np.linalg.norm(b - a)) / sp for (ma, a, sa), (mb, b, sp) in zip(vl[:-1], vl[1:])]
    if not vmv or not vlv:
        return None, None, None
    return float(np.mean(vlv) / np.mean(vmv)), vmv, vlv

def main():
    repr, age = load()
    n = len(age)
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    log(f"pool n={n}  M(>50<90)={int((band==1).sum())} L(>=90)={int((band==2).sum())}")

    # --- observed estimators ---
    r_std, vmv_std, vlv_std = bin_vel_ratio(repr, age, [70, 80, 90], [90, 97, 105])
    r_fine, vmv_fine, _ = bin_vel_ratio(repr, age, [70, 75, 80, 85, 90], [90, 97, 105])
    out = {"n": n, "obs_std_m_edges": r_std, "m_vels_std": vmv_std, "l_vels_std": vlv_std,
           "obs_fine_m_bins": r_fine, "m_vels_fine": vmv_fine}

    # --- circular-shift null ---
    order = np.argsort(age, kind="stable")
    repr_s = repr[order]; age_s = age[order]
    for tag, m_edges in [("std", [70, 80, 90]), ("fine", [70, 75, 80, 85, 90])]:
        obs = out[f"obs_{tag}_m_" + ("edges" if tag == "std" else "bins")]
        if obs is None:
            out[f"p_circ_{tag}"] = None
            continue
        nulls = []
        for _ in range(N_SHIFT):
            k = int(RNG.integers(1, n))  # shift by 1..n-1 (k=0 = observed)
            ns = np.roll(repr_s, k, axis=0)
            v = bin_vel_ratio(ns, age_s, m_edges, [90, 97, 105])[0]
            if v is not None and np.isfinite(v):
                nulls.append(v)
        nulls = np.array(nulls)
        p_one = float((nulls <= obs).mean())
        p_two = float((np.abs(nulls - 1.0) >= abs(obs - 1.0)).mean())
        out[f"null_{tag}_mean"] = float(nulls.mean())
        out[f"null_{tag}_p25"] = float(np.quantile(nulls, 0.25))
        out[f"null_{tag}_p75"] = float(np.quantile(nulls, 0.75))
        out[f"p_circ_one_{tag}"] = p_one
        out[f"p_circ_two_{tag}"] = p_two
        log(f"{tag}: obs={obs:.4f} null_mean={nulls.mean():.4f} "
            f"[p25 {np.quantile(nulls,.25):.3f}, p75 {np.quantile(nulls,.75):.3f}] "
            f"p_one={p_one:.4f} p_two={p_two:.4f}")

    with open(os.path.join(RUNS, "h2_l1_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log("WROTE h2_l1_results.json")

if __name__ == "__main__":
    main()
