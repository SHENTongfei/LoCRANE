# Data provenance & licenses

All data shipped or referenced by this repository are **real, published,
de-identified research data**. No synthetic or fabricated values are included
anywhere; `data/demo_bundle.npz` is a deterministic row-slice of the frozen
fold-0 feature representation of the full cohort (regenerable with
`tools/make_demo_data.py`).

## 1. Internal cohort — n = 1715 (the "D-1a" cohort)

* Source publication: *Transcriptomic Analysis of Chinese Longevity Cohorts*,
  Cell Reports Medicine (2026), PMC13198318 — 811 long-lived individuals (≥90)
  + 940 younger controls, whole-blood RNA-seq, Hainan/Hunan/Hubei.
* Data release: Mendeley Data `fpf4y72kfz` v1, DOI
  [10.17632/fpf4y72kfz.1](https://doi.org/10.17632/fpf4y72kfz.1),
  license **CC BY 4.0** (verified in the dataset metadata).
* What we redistribute:
  * release asset `expression_matrix.tsv.gz` — the processed 1715 × 11030
    log-expression matrix used by the model (derived work; CC BY 4.0 permits
    redistribution with attribution);
  * `data_v3/labels_v3.csv` — de-identified numeric IDs, age, sex, region,
    band, and pre-computed class weights (from the same release);
  * `data/demo_bundle.npz` — a 120-sample real slice of the frozen feature
    representation (CC BY 4.0, same attribution).
* Attribution requirement: cite the original publication and the Mendeley DOI
  when reusing any of these files.

## 2. cohort2 — Israel whole-blood chip cohort (n = 66, 10 aged ≥ 90)

* Source: GEO **GSE123696** (public). Used as an independent-population
  external leg (zero-shot + k-shot fast adaptation).
* We do **not** redistribute its feature caches; rebuild with
  `src/w2_external_fe.py` after downloading from GEO.

## 3. GTEx (n = 64 balanced whole-blood subset)

* Source: GTEx Portal (v8). Terms of use apply; **not redistributed**.
* Used strictly as a cross-platform *disclosure* leg (reported zero-shot
  mismatch MAE 77.7 y / Spearman 0.22 in `results/ext_leg_results.json`).

## 4. LM22 signature matrix (Newman et al., 2015, Cell)

* Required by the 19-cell immune channel (`common.lm22_deconvolution`).
* Freely available for academic use from the CIBERSORT authors; **not
  redistributed here**. Obtain `LM22_signature.tsv` (22 immune cell types ×
  547 genes), place it under `CRANE_DATA_DIR`, and verify against the SHA256
  recorded in `EXCLUDED.md`.

## 5. Privacy

All subject-level files contain de-identified numeric IDs with age/sex/region
only — no dates, no genomic sequences, no phenotypic free text — as published
by the original authors under CC BY 4.0.
