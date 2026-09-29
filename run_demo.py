# -*- coding: utf-8 -*-
"""run_demo.py - self-contained LoCRANE demo on a REAL 120-sample data slice.

The bundle (data/demo_bundle.npz) is a stratified, provenance-recorded slice of the
frozen fold-0 feature representation of the full 1715-sample cohort (D-1a, CC BY 4.0;
see DATA_LICENSE.md). Every number in it is real.

What this demo runs:
  * the REAL model class (src/model_v3.py, CraneZV3): GeneModulePool(3000 HVG -> 256
    modules) + FiLM sex conditioning + Transformer trunk + ordinal 3-band head +
    ridge-residual age head + 19-cell immune token;
  * a REDUCED training objective (age SmoothL1 + ordinal band CE) so it fits in a few
    CPU-minutes. The certified pipeline (src/train_v3.py / cg_train.py) uses the full
    pre-registered objective (dilated band-position head, band/bucket weights,
    early-stop on macro ordinal AUC) - run that for paper numbers.

Usage:
    python run_demo.py                    # ~2-4 min on CPU
    DEMO_EPOCHS=120 DEMO_SEED=7 python run_demo.py
Outputs: demo_results.json + stdout summary.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys_k = os.path.join(HERE, "src")
import sys
sys.path.insert(0, sys_k)
from model_v3 import CraneZV3, count_params  # noqa: E402

EPOCHS = int(os.environ.get("DEMO_EPOCHS", "60"))
SEED = int(os.environ.get("DEMO_SEED", "42"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    z = np.load(os.path.join(HERE, "data", "demo_bundle.npz"), allow_pickle=False)
    Xq, Xs, imm = z["Xq"], z["Xs"], z["imm"]
    age, sex01, is_val = z["age"], z["sex01"], z["is_val"]
    g, m = z["km_g"], z["km_m"]
    tr, va = is_val == 0, is_val == 1

    # ridge anchor (same design as the full pipeline: age ~ Xq)
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(Xq[tr], age[tr])
    mu, sd = float(age[tr].mean()), max(1e-6, float(age[tr].std()))
    rda = (ridge.predict(Xq) - mu) / sd

    # band codes are stored as integers (Y=0, M=1, L=2)
    bi = band = z["band"].astype(np.int64)
    bp = np.eye(3, dtype=np.float32)[bi]

    assignment = {int(gi): int(mi) for gi, mi in zip(g, m)}
    model = CraneZV3(n_genes=3000, assignment=assignment, n_modules=256, n_immune=19).to(DEV)
    print(f"[demo] device={DEV} params={count_params(model)/1e6:.2f}M "
          f"train={tr.sum()} val={va.sum()} epochs={EPOCHS} seed={SEED}")

    tq = torch.from_numpy(Xq[tr]).float().to(DEV)
    ts = torch.from_numpy(Xs[tr]).float().to(DEV)
    tx = torch.from_numpy(sex01[tr]).float().to(DEV)
    ti = torch.from_numpy(imm[tr]).float().to(DEV)
    tr_ = torch.from_numpy(rda[tr]).float().to(DEV)
    tbp = torch.from_numpy(bp[tr]).float().to(DEV)
    ty = torch.from_numpy((age[tr] - mu) / sd).float().to(DEV)  # normalized target (pipeline convention)
    tb = torch.from_numpy(bi[tr]).long().to(DEV)

    vq = torch.from_numpy(Xq[va]).float().to(DEV)
    vs = torch.from_numpy(Xs[va]).float().to(DEV)
    vx = torch.from_numpy(sex01[va]).float().to(DEV)
    vi = torch.from_numpy(imm[va]).float().to(DEV)
    vr = torch.from_numpy(rda[va]).float().to(DEV)
    vbp = torch.from_numpy(bp[va]).float().to(DEV)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    l_age = nn.SmoothL1Loss()
    l_band = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(tq, ts, tx, immune=ti, ridge_z=tr_, band_prior=tbp)
        loss = l_age(out["age"], ty) + l_band(out["band_logits"], tb)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[demo] ep{ep+1} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        out = model(vq, vs, vx, immune=vi, ridge_z=vr, band_prior=vbp)
        pred = out["age"].cpu().numpy() * sd + mu  # denormalize (pipeline convention)
        blogit = out["band_logits"].cpu().numpy()
        with torch.no_grad():
            otr = model(tq, ts, tx, immune=ti, ridge_z=tr_, band_prior=tbp)
            pred_tr = otr["age"].cpu().numpy() * sd + mu
    yva = age[va]
    mae = float(np.mean(np.abs(pred - yva)))
    sp = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(yva)))[0, 1])
    bacc = float((blogit.argmax(1) == bi[va]).mean())

    # mini one-sided split-conformal certificate (simplified CADENCE):
    # calibrate q90 on train residuals of the trained model, certify a val sample
    # as "decelerated" when (true age - prediction) falls below the 90% bound
    res_tr = age[tr] - pred_tr
    q90 = float(np.quantile(res_tr, 0.9, method="higher"))
    certs = (yva - pred) <= q90
    cert_rate = float(certs.mean())
    cert_rate_L = float(certs[bi[va] == 2].mean()) if (bi[va] == 2).any() else float("nan")

    res = {
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "epochs": EPOCHS, "seed": SEED,
        "device": str(DEV), "params_M": round(count_params(model) / 1e6, 3),
        "val_MAE_years": round(mae, 3), "val_spearman": round(sp, 4),
        "val_band_acc": round(bacc, 4),
        "conformal_q90_years": round(q90, 3),
        "certified_rate_val": round(cert_rate, 4),
        "certified_rate_L_band": round(cert_rate_L, 4),
        "note": ("demo = pipeline verification on a 120-sample real slice with a reduced "
                 "objective; NOT comparable to the certified full-cohort numbers "
                 "(results/headlines_full.json: OOF MAE 13.60 y, Spearman 0.696)."),
        "bundle_provenance": {
            "source": "D-1a (Cell Rep Med 2026 companion, Mendeley doi:10.17632/fpf4y72kfz.1, CC BY 4.0)",
            "slice": "stratified from frozen fold-0 feature representation (fe_cache_f0_full.npz)",
            "bands": {b: int((band == v).sum()) for b, v in
                      {"Y": 0, "M": 1, "L": 2}.items()},
            "age_range": [float(age.min()), float(age.max())],
        },
    }
    json.dump(res, open(os.path.join(HERE, "demo_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    print("[demo] wrote demo_results.json")


if __name__ == "__main__":
    main()
