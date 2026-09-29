# LoCRANE — Longevity-Certified Residual Age Network Ensemble

**LoCRANE** (中文名: 长寿鹤) is a transcriptomic aging model with a twist: instead of only
predicting age, it issues **one-sided statistical certificates of decelerated aging**
(**CADENCE** — studentized split-conformal certificates with binomial enrichment tests)
for the extreme-old tail of a 1,715-sample whole-blood RNA-seq cohort (795 individuals
aged ≥90). The model was **pre-registered, certified under a frozen out-of-fold (OOF)
5-fold × 5-seed ensemble, and adversarially audited** — every number in this README is
traceable to a JSON file in [`results/`](results/).

> **Identity.** LoCRANE = **Lo**ngevity-**C**ertified **R**esidual **A**ge **N**etwork
> **E**nsemble. Its certificate protocol is **CADENCE** / **CADENCE-S** (sex-conditional).
> "Lo" is the initial syllable of *Longevity* (as "Epi" in EpiCENSUS); the model runs on
> frozen representations — a LoRA-tuned sequence tower was screened during design and
> failed its pre-registered gate, so the final model deliberately keeps frozen features.

---

## Headline results (all real, all traceable)

| Claim | Value | Source JSON |
|---|---|---|
| OOF age prediction (n=1715, 25-ckpt ensemble) | **MAE 13.60 y, Spearman 0.696** | `results/headlines_full.json` |
| H1 healthy-tail gap (residual, L vs M/Y; chrono-permuted) | **−6.64 y**, 0/2000 permutations (p < 0.001) | `results/headlines_full.json` |
| Residual mean by band (Y / M / L) | +7.72 / +2.33 / **−4.63** y | `results/headlines_full.json` |
| H4 certified-decelerator rate (Y / M / L, α=0.10) | 0.55% / 4.20% / **8.81%**, monotone; pooled L enrichment 2.53× | `results/headlines_full.json` |
| CADENCE-S, female L band | **6.73% certified, 2.65× enrichment, p = 9.2e-8** | `results/cadence_sex_full.json` |
| CADENCE-S, male L band | **10.43% certified, 2.41× enrichment, p = 7.0e-5** | `results/cadence_sex_full.json` |
| CUDA-graphs vs eager parity (Leg C) | pair-Spearman 0.9997, ΔMAE 0.018 y → **PASS** | `results/legC_results.json` |
| W2 cohort fast-adaptation (cohort2, Israel chip, 20 anchors, zero retraining) | MAE 17.66 → **14.13 y** (Δ20 = **+3.53 y**, monotone, **WIN**) | `results/w2_probe_results.json` |
| External H1 tail-gap (cohort2, disclosure n_L=10) | −6.27 y (same direction as internal) | `results/ext_leg_results.json` |
| H2 velocity-ratio / H3 width-ratio | **not certified** (0.29 / 1.50) — honest disclosure | `results/h2_*_results.json`, `results/h3_*_results.json` |

Honesty notes baked into the protocol: GTEx serves as a cross-platform *disclosure* leg
only (zero-shot MAE 77.7, Spearman 0.22 — tissue mismatch, reported not hidden);
cohort2 n_L=10 is disclosure-grade; H2/H3 non-certification is part of the pre-registered
verdict system, not a defect.

## Interpretability (M1–M4 + certificate contrast, `results/mining_v3_*.json`)

* **M1** module do-ablation: tail-band discrimination is transcriptome-wide, not
  module-concentrated (strongest module mod122, A2M/HIST2H2BF/PZP, +0.041 y, below fold noise).
* **M2** gene-level GIP: ATF6B / GCNT7 / TENC1 / POU5F2 / CKMT2 / P2RY8 / GPR156 /
  C20orf24 / LRRC25 / ABCA13 (ER-stress, mitochondrial, lipid-transport axes).
* **M3** tail contrast (L vs M): L-up C20orf195 / TBX21 / CLDN20 / ZNF460 / TEDDM1 / ANGPT1 /
  TRIM63 / PRRC2A / LRRC32 / SMIM20; L-down HIST1H2BE / HIST1H3C (H3.3 family) / XKR9 /
  HELLS / FGFR2 / CNTNAP3B / ZNF320 / GPX2 / C1RL.
* **M4** 19-cell immune channel: L band read through Neutrophils / M2 macrophages /
  Plasma / CD8 T / resting NK.
* **Certificate contrast** (94 certified vs 701 uncertified 90+): HLA-F / CD83 / DNAJB1 /
  RRAS / RPS26 / HBEGF / BRD2 / VSTM1 / MICA / NR4A2 … — an immune–stress signature.
* Literature grading (PubMed abstract level): 8 hits grade-B support; **5 novel candidates
  (GCNT7 / C22orf46 / CLDN20 / TEDDM1 / ZNF320) — we flag, not claim.**

---

## Repository layout

```
LoCRANE/
├── run_demo.py              # self-contained demo (real 120-sample slice, ~2 min CPU)
├── demo_results.json        # output of a verified demo run on this machine
├── src/                     # model, trainer, CUDA-graphs trainer, chains, collectors, mining
│   ├── model_v3.py          # CraneZV3 (= LoCRANE architecture)
│   ├── train_v3.py          # full pre-registered trainer (H58 arms, registry, heartbeat)
│   ├── cg_train.py          # bucketed CUDA-graphs trainer (production speed)
│   ├── cg_full_chain.py     # single-owner 25-job launcher (certification grade)
│   ├── cg_ckpt_chain.py     # 25-job launcher that also saves per-job checkpoints
│   ├── cadence_full_collect.py  # CADENCE-S certificate collector
│   ├── headlines_full.py    # headline metrics (H1/H4/OOF)
│   ├── mining_v3*.py        # interpretability M1–M4 + certificate contrast + litcheck
│   ├── w2_*.py, ext_leg_suite.py # external legs (cohort2 GSE123696 / GTEx)
│   ├── h2_*.py, h3_*.py, legC_*.py, h_placeco_full.py # verification suite
│   └── common.py            # shared feature utilities (HVG, LM22 deconvolution, folds)
├── data/
│   ├── demo_bundle.npz      # REAL 120-sample slice (see provenance below)
│   ├── demo_labels.csv      # human-readable labels for the slice
│   └── demo_genes.txt       # the 3,000 HVG gene names (pipeline-identical selection)
├── data_v3/                 # full-cohort metadata (IDs, folds, 256-module assignment)
├── results/                 # 18 pre-registered verdict/result JSONs (the paper numbers)
├── docs/                    # TAIL_PROTOCOL_v4.md (pre-registration), MINING_v3_PLAN.md
├── tools/make_demo_data.py  # regenerates demo_bundle.npz from full local data
├── DATA_LICENSE.md          # data provenance & licenses
├── EXCLUDED.md              # SHA256 ledger of everything NOT shipped (and why)
└── requirements.txt
```

## Installation

```bash
python -m pip install -r requirements.txt
```

* Python ≥ 3.10; PyTorch ≥ 2.1 (CPU is enough for the demo; the full chain was run on
  CUDA 12.8, torch 2.11).
* Developed and tested on **Windows** (`train_v3.py` uses `msvcrt` file locks for the
  multi-worker registry). Linux users: replace the lock in `train_v3.py::_reg_lock`
  with `fcntl` — that is the only Windows-specific dependency.

## Quickstart (2 minutes, no download)

```bash
python run_demo.py
```

This trains the **real architecture** (CraneZV3: 3000 HVG → 256 gene modules →
FiLM-sex Transformer + 19-cell immune token + ridge-residual age head + ordinal band
head) on a **real, provenance-recorded 120-sample slice** (13 Y / 51 M / 56 L, ages
24–106) with a reduced objective, then reports validation MAE / Spearman / band
accuracy and a mini one-sided conformal certificate rate.

A verified run on this machine produced (`demo_results.json`):

```
val_MAE_years: 23.334   val_spearman: 0.192   val_band_acc: 0.583
conformal_q90_years: 20.04   certified_rate_val: 0.458
```

**Do not compare demo numbers to the paper numbers** — 96 training samples with a
reduced objective is a pipeline check, not a certified result. The certified numbers
(13.60 y MAE etc.) require the full 1,715-sample cohort and the 25-checkpoint OOF
ensemble below.

## Full reproduction

### 0. Data placement

Download the **release assets** of this repository and place them next to the repo:

```
<CRANE_DATA_DIR>/                          # any folder, exported as CRANE_DATA_DIR
├── expression_matrix.tsv.gz               # 1715 x 11030 log-expression (release asset)
├── LM22_signature.tsv                     # see DATA_LICENSE.md (Newman et al. 2015)
└── training_data/balanced_folds/fold_0/
    ├── train_expr.csv                     # HVG-reference fold CSV (release asset)
    ├── val_expr.csv
    ├── train_pheno.csv
    └── val_pheno.csv
```

`data_v3/` (labels, 5-fold split `folds_v3_full.json`, 256-module assignments
`km256_f*.npz`, LM22 QC) is already in this repository.

### 1. Sanity run (one fold, one seed)

```bash
export CRANE_DATA_DIR=/abs/path/to/data
export V3_FOLDS=folds_v3_full.json        # full-cohort 5-fold split (in data_v3/)
export V3_FESUF=_full                     # feature-cache suffix (full cohort)
export V3_EPOCHS=300
python src/train_v3.py --arm stage0 --folds 0 --seeds 42 --tag repro
```

First run builds the fold-0 feature cache (`data_v3/fe_cache_f0_full.npz`, ~60 MB;
QuantileTransformer fit on the fold's own inner-train — leakage-free by construction).
Registry entries land in `runs_v3/registry_stage0_repro.json` (done-flag resume:
re-running skips finished jobs).

### 2. Certified chain (25 checkpoints, single owner)

```bash
python src/cg_full_chain.py               # 5 folds x 5 seeds, serialized, CUDA graphs
python src/cadence_full_collect.py        # -> runs_v3/cadence_sex_full.json
python src/headlines_full.py              # -> runs_v3/headlines_full.json
```

`cg_full_chain.py` holds ONE process that owns the whole queue (no shell loops —
that was the double-launch root cause), resumes from done-flags, and stops the chain
orderly on any job failure. On a single consumer GPU (~16 GB) the chain takes a few
hours. `cg_ckpt_chain.py` additionally saves per-job weights to `runs_v3/ckpt_cg/`
for the interpretability stage.

### 3. Interpretability (needs Stage-2 checkpoints)

```bash
python src/mining_v3.py                   # M1-M4 -> runs_v3/mining_v3_results.json
python src/mining_v3_certfix.py           # certificate-contrast GIP
python src/mining_v3_litcheck.py          # PubMed abstract-level A/B/C/NONE grading
```

### 4. Verification suite (each script = one pre-registered verdict)

| Script | Verdict it produces |
|---|---|
| `src/legC_compare.py` / `src/cg_legC.py` | CUDA-graphs ↔ eager parity (Leg C) |
| `src/h_placeco_full.py` | label-permuted placebo (H1 null) |
| `src/h2_full_repr.py` + `h2_l1_rescue.py` + `h2_l3_blockperm.py` | H2 velocity-ratio (not certified) |
| `src/h3_l5_full.py` / `src/h3_mc_full.py` | H3 conformal width ratio (disclosure) |
| `src/w2_external_fe.py` → `w2_ensemble.py` → `w2_probe.py` | W2 fast-adaptation (cohort2 WIN / gtex KILL) |
| `src/ext_leg_suite.py` | external-leg H1/H4/CADENCE-S (disclosure grades) |

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `CRANE_DATA_DIR` | `C:/Users/TS/Desktop/crane` | root containing `expression_matrix.tsv.gz`, `training_data/`, `LM22_signature.tsv` |
| `V3_FOLDS` | `folds_v3.json` | fold-split JSON in `data_v3/` (use `folds_v3_full.json` for the full cohort) |
| `V3_FESUF` | `""` | feature-cache suffix (`_full` for the full cohort) |
| `V3_EPOCHS` | `300` | training epochs |
| `V3_BCAP` | `512` | CUDA-graph batch cap (buckets {256, 512}) |
| `V3_RR` | `1.0` | ridge-residual gate for the age head |
| `CG_FLAGPRE` / `CG_SAVECKPT` | `cg_full` / unset | checkpoint prefix / enable weight saving |
| `CRANE_PY` | local conda python | interpreter pinned by the chains |
| `DEMO_EPOCHS` / `DEMO_SEED` | `60` / `42` | demo knobs |

## Data provenance & licenses

* **Internal cohort (n=1715)** = the analysis cohort of *"Transcriptomic Analysis of
  Chinese Longevity Cohorts"* (Cell Reports Medicine, 2026; PMC13198318), downloaded as
  the authors' Mendeley Data release `fpf4y72kfz` (DOI 10.17632/fpf4y72kfz.1),
  license **CC BY 4.0**. IDs are the paper's de-identified numeric codes.
* **cohort2** = GSE123696 (Israel, whole-blood chip, 66 samples incl. 10 aged ≥90).
* **GTEx** = GTEx v8 whole blood (64-sample balanced subset) — cross-platform
  disclosure leg only; GTEx terms apply.
* **LM22** = Newman et al. 2015 signature matrix — free for research; not redistributed
  here (see `DATA_LICENSE.md`).
* Code: MIT (see `LICENSE`). Everything excluded from this repo, with SHA256 and the
  reason, is listed in `EXCLUDED.md`.

## Citation

Preprint in preparation. Until then, please cite this repository:

> LoCRANE: Longevity-Certified Residual Age Network Ensemble — one-sided conformal
> deceleration certificates for the 90+ healthy tail (2026).
