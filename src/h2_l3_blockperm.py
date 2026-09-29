# -*- coding: utf-8 -*-
"""h2_l3_blockperm.py - H2 L3 rescue (pre-registered in the H2 verdict):
within-band stratified BLOCK-permutation null on the full-cohort repr.

Rationale: the circular-shift null (p_one=0.124) was conservative because a
global cyclic rotate PRESERVES the age->repr drift structure (it just
re-anchors it), so the null ratios stayed low (null_mean 0.55) and obs 0.287
landed only at p~0.12. The L3 block-perm instead BREAKS the within-band drift:
per band (Y/M/L) we sort by age, split into K contiguous blocks, and randomly
permute the block ORDER (in-block order preserved). Under H0 (no tail
deceleration) each band's drift is randomized -> both L-leg and M-leg
velocities become noise -> ratio ~ O(1). A genuinely decelerated tail keeps
its L-leg velocity low even under M-leg scrambling, so obs should fall in the
left null tail.

statistic = L-leg bin velocity / M-leg bin velocity (same bins as
h2_full_repr: M [70,80,90], L [90,97,105]). 2000 block perms.
p_one = frac(nulls <= obs). Gate (pre-registered H2 menu): ratio <= 0.85
AND p < 0.01. Writes runs_v3/h2_l3_results.json.
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_v3")
RNG = np.random.default_rng(20260931)
N_PERM = 2000
N_BLOCKS = 12

def log(m): print(f"[h2l3] {m}", flush=True)

def bin_vel_ratio(repr, age, m_edges, l_edges):
    def segs(edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (age >= lo) & (age < hi)
            if m.sum() >= 3:
                out.append(((lo + hi) / 2, repr[m].mean(0), float(hi - lo)))
        return out
    vm = segs(m_edges); vl = segs(l_edges)
    vmv = [float(np.linalg.norm(b - a)) / sp for (x, a, sa), (y, b, sp) in zip(vm[:-1], vm[1:])]
    vlv = [float(np.linalg.norm(b - a)) / sp for (x, a, sa), (y, b, sp) in zip(vl[:-1], vl[1:])]
    if not vmv or not vlv:
        return None
    return float(np.mean(vlv) / np.mean(vmv))

def block_permute(repr, age, band, nb=N_BLOCKS):
    """within-band block permutation: return a repr row-ordering that, applied,
    scrambles within-band drift. Operates on a band-sorted index layout."""
    perm = np.arange(len(repr))
    for b in (0, 1, 2):
        idx = np.where(band == b)[0]
        if len(idx) < nb:
            continue
        # age-sort within band
        aorder = idx[np.argsort(age[idx], kind="stable")]
        blocks = np.array_split(aorder, nb)
        RNG.shuffle(blocks)
        newseq = np.concatenate(blocks)
        # place back: the positions held by this band now hold the shuffled order
        perm[aorder] = newseq
    return repr[perm]  # shuffled representation

def main():
    z = np.load(os.path.join(RUNS, "repr_full_b3tail.npz"))
    repr_all, ids, age = z["repr"], [str(x) for x in z["ids"]], z["age"]
    band = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    M_EDGES = [70, 80, 90]; L_EDGES = [90, 97, 105]
    obs = bin_vel_ratio(repr_all, age, M_EDGES, L_EDGES)
    nulls = []
    for _ in range(N_PERM):
        rp = block_permute(repr_all, age, band)
        v = bin_vel_ratio(rp, age, M_EDGES, L_EDGES)
        if v is not None and np.isfinite(v):
            nulls.append(v)
    nulls = np.array(nulls)
    p_one = float((nulls <= obs).mean())
    p_two = float((np.abs(nulls - 1) >= abs(obs - 1)).mean())
    out = {"n": int(len(age)), "nL": int((band == 2).sum()), "nM": int((band == 1).sum()),
           "obs_ratio": obs, "null_mean": float(nulls.mean()),
           "null_p10": float(np.quantile(nulls, 0.10)), "null_p90": float(np.quantile(nulls, 0.90)),
           "p_one": p_one, "p_two": p_two, "n_perm": N_PERM, "n_blocks": N_BLOCKS,
           "gate": "ratio<=0.85 & p<0.01 (pre-registered H2 menu)"}
    out["PASS"] = bool(obs is not None and obs <= 0.85 and p_one < 0.01)
    json.dump(out, open(os.path.join(RUNS, "h2_l3_results.json"), "w"), indent=1, default=float)
    log(f"obs={obs:.4f} null_mean={nulls.mean():.4f} [p10 {np.quantile(nulls,.1):.3f}, p90 {np.quantile(nulls,.9):.3f}] "
        f"p_one={p_one:.4f} p_two={p_two:.4f} PASS={out['PASS']}")
    log("WROTE h2_l3_results.json")

if __name__ == "__main__":
    main()
