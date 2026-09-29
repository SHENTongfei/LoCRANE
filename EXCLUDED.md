# EXCLUDED.md — what is NOT in this repository, and why

Everything below is real data/artifact that was excluded from the git tree for
size, license, or rebuildability reasons. SHA256 recorded so any recipient can
verify integrity against the authors' copies. Release assets are marked.

Generated: 2026-09-30. Machine-readable copy: `excluded_sha256.json`.

| File | Size | SHA256 | Reason |
|---|---|---|---|
| `d1a_mendeley_fpf4y72kfz_v1.zip (raw CC-BY package)` | 352.5 MB | `cdbc958187144c29…596e4e78` | raw source package; too large — download from Mendeley DOI 10.17632/fpf4y72kfz.1 (CC BY 4.0) |
| `d1a_readcount.npz (raw counts 19229x1751)` | 136.2 MB | `62b57d79537da20a…9bca30ce` | raw counts; rebuildable from the Mendeley package |
| `expression_matrix.tsv.gz (1715x11030 processed; ALSO release asset)` | 320.1 MB | `4b1f3d8e16aef0c8…cfab2bbf` | **release asset** — download from the GitHub Release of this repo |
| `LM22_signature.tsv (Newman 2015; obtain from CIBERSORT)` | 0.1 MB | `1bae24cfa5fc87bb…7c884a47` | third-party signature (Newman 2015); obtain from CIBERSORT (free academic) |
| `fold_0/train_expr.csv (release asset)` | 62.6 MB | `991192f286b83035…0abd4977` | **release asset** |
| `fold_0/val_expr.csv (release asset)` | 17.3 MB | `54ccb7007cdc050f…5efa55f4` | **release asset** |
| `fold_0/train_pheno.csv (release asset)` | 0.0 MB | `7c02df1e5b1c228e…0e62833a` | **release asset** |
| `fold_0/val_pheno.csv (release asset)` | 0.0 MB | `1c01685b247a4e88…119f4997` | **release asset** |
| `fe_cache_f0.npz (fold feature caches (rebuildable))` | 60.9 MB | `168d5307e0e4e978…8b7030b0` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f0_full.npz (fold feature caches (rebuildable))` | 61.0 MB | `ba24c49e6297333a…0469b914` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f1.npz (fold feature caches (rebuildable))` | 60.9 MB | `783b6d92976916d1…0b6b4ff2` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f1_full.npz (fold feature caches (rebuildable))` | 61.4 MB | `b47d00693ab0f4fc…b2d3e114` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f2.npz (fold feature caches (rebuildable))` | 60.9 MB | `229da6a94f061655…347e80a6` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f2_full.npz (fold feature caches (rebuildable))` | 61.4 MB | `adf1a0586bf16286…30845216` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f3.npz (fold feature caches (rebuildable))` | 60.9 MB | `fe5d9eb00f96c5a7…3c6b9bf6` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f3_full.npz (fold feature caches (rebuildable))` | 61.4 MB | `24c54f1eb12a952b…59b4a7ef` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f4.npz (fold feature caches (rebuildable))` | 60.9 MB | `d1449e09a8a5621e…e7a7cb36` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `fe_cache_f4_full.npz (fold feature caches (rebuildable))` | 61.4 MB | `b8bc9ed402f9a4f9…b0e7773d` | fold feature caches — rebuilt automatically by src/train_v3.py on first run |
| `plm_esm2_650M_hvg.npz (PLM embedding caches (third-party model derived))` | 50.1 MB | `73f0414a650859d4…522811c2` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `plm_esm2_650M_hvg3000.npz (PLM embedding caches (third-party model derived))` | 9.3 MB | `5195ae01bf403b4d…857b2f2b` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `plm_esm2_650M_hvg3000_loraTrue.npz (PLM embedding caches (third-party model derived))` | 10.2 MB | `da046aa66425310e…fc5a76d7` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `plm_esm_c_600M_hvg3000.npz (PLM embedding caches (third-party model derived))` | 8.3 MB | `d114b986af2432fb…89551959` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `plm_prot_t5_xl_hvg3000.npz (PLM embedding caches (third-party model derived))` | 7.4 MB | `8ba14bba2f945303…165dad54` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `plm_map.csv (PLM id map)` | 0.5 MB | `ba25c6e27675763e…855af35d` | PLM embedding caches — third-party-model derived; only needed for the (killed) PLM/LoRA arms |
| `runs_v3/ckpt_cg/*.pt (25 files, weights only)` | 229 MB total | `[25 weight files…history]` | model weights — reproducible via src/cg_ckpt_chain.py |

## Also excluded (not hashed here)

* `runs_v3/` work-logs, registries, and intermediate npz (M-predictions, representation caches) — all conclusion-bearing numbers are re-exported to `results/` as JSON; the raw intermediates are reproducible from the scripts.
* Manuscript sources and PDFs (separate submission channel).
* Old-model (pre-LoCRANE, "v2") artifacts — quarantined during the project and excluded here to prevent cross-era confusion.
