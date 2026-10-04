# LoCRANE

**Lo**ngevity-**C**ertified **R**esidual **A**ge **N**etwork **E**nsemble — a
locked-protocol transcriptomic aging clock for the Chinese longevity cohort.

LoCRANE wraps a compact residual age network in four longevity-anchored domain
priors, blends it with two elastic-net members, and prices every claim under a
frozen 15-split paired protocol. The shipped hybrid reaches **5.740 years MAE**
on the working-age validation range, ahead of ten classical regressors, four
matched deep architectures, and the source cohort study's published clock
(6.52 years), with the priors' **+0.30-year** value quantified by a
same-backbone control.

This repository carries the **code, a real runnable data slice, the frozen
result tables, the protocol lock, and the supplement**. It does not carry the
paper manuscript, the figures, or the full cohort (see
[Data](#2-data) below).

---

## 1. Quickstart (2 minutes)

```bash
git clone https://github.com/SHENTongfei/LoCRANE.git
cd LoCRANE
pip install -r requirements.txt
python code/run_demo.py
```

The demo trains the full recipe — cross-fitted ridge-residual anchor, FiLM sex
conditioning, three-band auxiliary supervision with the 90+ tail, monotonicity
regularizer, hybrid deep+linear blend — on the released **real** slice
(300 variance-selected genes × 150 samples: 100 working-age + 50 verified 90+)
and prints a per-split table plus the prior-value summary:

```
ridge (300-gene, tuned) reference MAE  : 6.640 y
eq3' (four priors + hybrid blend) MAE : 6.239 y
same-backbone priors-off control MAE : 6.768 y
prior value on this slice            : +0.529 y (paired p=0.438)
```

Demo-scale numbers verify the **pipeline** only (5 splits, small n). All paper
numbers come from the 15-split locked protocol on the full cohort.

## 2. Data

| Item | What it is | Access |
|---|---|---|
| `data/demo/` | Real slice of the cohort: 300 genes × 150 samples + phenotype (age, sex), SHA256-locked | in this repository |
| Full cohort (1,715 blood transcriptomes, Hainan/Hubei/Hunan) | The complete training cohort | Collaborative sharing — contact the corresponding author of Xiao et al., *Cell Reports Medicine* 7:102767 (2026) |
| GTEx whole blood (ext1, n=64) | External RNA-seq cohort | public, [GTEx Portal](https://gtexportal.org/) |
| GSE123696 (ext2, n=66) | External Israeli PrimeView microarray cohort | public, [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123696) |
| GSE65218 (Vitality 90+, n=151) | Independent 90+ survival probe | public, [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65218) |

The demo slice was drawn with `numpy.default_rng(2026)`: 100 random working-age
(50–89) samples and 50 random verified 90+ samples, restricted to the 300
highest-variance genes of the full matrix. `data/demo/SHA256SUMS` pins the
exact bytes; `run_demo.py` verifies the hash before training.

Point every full-cohort script at your copy of the data:

```bash
export CRANE_DATA_DIR=/path/to/cohort        # expects expression_matrix.tsv + labels
python code/x41_full_lock.py                 # 15-split internal lock (GPU recommended)
```

## 3. Repository map

```
LoCRANE/
├── protocol_lock.json          # frozen protocol + every headline number's source
├── code/
│   ├── run_demo.py             # one-command demo (this is the entry point)
│   ├── model_v3.py             # network definition (residual blocks, FiLM, band head)
│   ├── x41_full_lock.py        # 15-split internal lock: base vs monotonicity-adopted
│   ├── x42_ablation.py         # component removal costs (anchor/FiLM/band-aux/sex)
│   ├── x43_xai25.py            # 25-seed gated attribution suite (6 axes)
│   ├── x44_mining.py           # causal levers, module do-costs, geometry
│   ├── x45_ext_v4.py           # external zero-shot battles (GTEx + GSE123696)
│   ├── x47_dl_baselines.py     # matched deep suite incl. priors-off control
│   └── x48_composition.py      # exhaustive 15-subset composition matrix
├── results/                    # frozen result tables (the paper's only number sources)
│   ├── x20b_winner_vs_baselines.json   # 10/10 classical baselines, Holm-adjusted
│   ├── x41_eq3_from_log.json           # per-split lock: eq3' vs base
│   ├── x42_ablation.json               # component costs with paired p-values
│   ├── x47_dl_baselines.json           # deep suite + priors-off (+0.30 y prior value)
│   ├── x48_composition.json            # 15-combination matrix (eq3' ranks 1st)
│   ├── x15c_core3_kshot_v4.json        # external few-shot (linear core wins all k)
│   └── x45_ext_battle_v4.json          # external zero-shot numbers
├── supplement/LoCRANE_supplement.pdf   # figure source map, per-split table,
│                                       # disclosure ledger, survival-probe detail
└── data/demo/                          # real slice + SHA256SUMS
```

## 4. Reproducing the paper (full cohort required)

With `CRANE_DATA_DIR` pointing at the cohort, the locked chain is:

| Step | Command | Produces | Paper claim |
|---|---|---|---|
| 1 | `python code/x41_full_lock.py` | 15-split lock, adoption gate | headline 5.740 vs base 5.757 (10/15, p=0.035) |
| 2 | `python code/x42_ablation.py` | component costs | anchor +0.078 > FiLM +0.019 > mono +0.017 |
| 3 | `python code/x47_dl_baselines.py` | deep suite + control | priors worth +0.30 y (6.044 vs 5.740, 15/15) |
| 4 | `python code/x48_composition.py` | composition matrix | eq3' ranks 1/15; deep −0.16 y irreplaceable |
| 5 | `python code/x45_ext_v4.py` | external zero/k-shot | linear core lowest at every k |
| 6 | `python code/x43_xai25.py` | 25-seed attribution | 5/6 axes pass gates; M3 fails honestly |
| 7 | `python code/x44_mining.py` | levers/modules/geometry | lever stability 0.938; velocity 0.93 |

Runtime: steps 1–4 need a GPU for comfort (a few hours total at 15 splits ×
5 seeds); steps 5–7 are minutes-to-an-hour on CPU/GPU. Every step writes its
frozen JSON next to the released ones — compare before citing.

## 5. Protocol lock (what "locked" means)

`protocol_lock.json` pins, before any evaluation: the 15 random 80/20 splits
(master seed 2026, identical rows for every method), the HVG8,000
training-fold feature screen, the two pre-registered comparison families
(10 classical + 4 deep) with paired Wilcoxon + Holm correction, the
monotonicity weight λ=0.05 and its adoption rule, the ensemble composition
rule (exhaustive matrix, no post-hoc search), and the external
relative-ordering scope with support-coverage disclosure. Every headline
number in the paper maps to one file in `results/`; nothing is recomputed
after evaluation.

## 6. Honesty ledger (read before citing)

* The deep member's per-seed direction rate is 0.56 — claims live at the
  ensemble level.
* EN-500 is an ensemble member; that comparison is member-vs-frame.
* 32/64 (ext1) and 27/66 (ext2) external samples sit outside the training age
  support — external claims are relative orderings.
* The one non-significant external cell (ext1, k=5 vs SVR-500) is printed on
  the figure.
* The FiLM attribution axis fails its pre-registered gate (ρ=0.435) and is
  reported as failed.
* The longevity signature does **not** stratify survival inside the 90+ band
  (HR 1.14, p=0.31 on Vitality 90+); gene-level pilot associations
  (GAL3ST4 p=0.007, PFKP directional) are graded pilot.

Full ledger: `supplement/LoCRANE_supplement.pdf`, §4.

## 7. Citation

```bibtex
@article{shen2026locrane,
  author = {Shen, Tim},
  title  = {LoCRANE: a longevity-certified multi-view deep ensemble for
            cross-population transcriptomic aging clocks and biological discovery},
  journal = {Engineering Applications of Artificial Intelligence},
  year   = {2026}
}
```

## 8. License

MIT (see `LICENSE`). The cohort data itself remains under the source study's
access terms; the released slice is for pipeline verification.
