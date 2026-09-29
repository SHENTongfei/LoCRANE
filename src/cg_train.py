# -*- coding: utf-8 -*-
"""cg_train.py - CUDA-Graphs trainer, BUCKETED (R2.5): kills per-step launch gaps
AND keeps the dynamic batch governor (user order: 动态扩批/缩批压满).
One graph per batch-size bucket {128,256,512,1024}; the wgov hysteresis now
SELECTS which bucket to replay instead of resizing tensors - adaptation is
quantized to buckets, capture is one-time per bucket at warmup.
Dual use: intra-step (graphs) cure for launch-bound tiny models; pair with
multi-stream co-scheduling later for inter-job gaps.
Usage: python cg_train.py --fold 2 --seeds 42 --epochs 30 [--gbs 512]
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("CRANE_DATA_DIR", "C:/Users/TS/Desktop/crane")
import common as C
from model_v3 import CraneZV3

DATA3 = os.path.join(os.path.dirname(HERE), "data_v3")
RUNS = os.path.join(os.path.dirname(HERE), "runs_v3")
DEV = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=2)
ap.add_argument("--seeds", default="42")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--gbs", type=int, default=512)          # starting batch target
ap.add_argument("--buckets", default="256,512")  # resident graphs: keep 2 (each holds its pool)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--rr", type=float, default=1.0)
args = ap.parse_args()

lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
folds = json.load(open(os.path.join(DATA3, os.environ.get("V3_FOLDS", "folds_v3.json"))))
kmd = np.load(os.path.join(DATA3, f"km256_f{os.environ.get('V3_KMFOLD', str(args.fold))}.npz"))
assign = {int(k): int(v) for k, v in zip(kmd["g"], kmd["m"])}
n_genes = len(assign)

z = np.load(os.path.join(DATA3, f"fe_cache_f{args.fold}{os.environ.get('V3_FESUF', '')}.npz"))
Xq_tr, Xs_tr, imm_tr = z["Xq_tr"], z["Xs_tr"], z["imm_tr"]
Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]

val_ids = set(folds[str(args.fold)])
inner = [str(i) for i in lab.index if str(i) not in val_ids]
lab_tr = lab.loc[inner]
lab_va = lab.loc[sorted(str(i) for i in folds[str(args.fold)])]  # SORTED = cache row order
age_tr = lab_tr["age"].to_numpy(np.float32)
age_va = lab_va["age"].to_numpy(np.float32)

p0 = C.load_fold(0)[2]
mu, sd = float(p0["age"].mean()), max(1e-6, float(p0["age"].std()))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(Xq_tr, age_tr)
rda_tr = (ridge.predict(Xq_tr) - mu) / sd
rda_va = (ridge.predict(Xq_va) - mu) / sd

def band_cum(age):
    b = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    y = np.zeros((len(age), 2), dtype=np.float32)
    y[:, 0] = (b >= 1).astype(np.float32)
    y[:, 1] = (b == 2).astype(np.float32)
    return torch.from_numpy(y)

def tau_t(age, dil=1.4):
    b = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    t = np.zeros(len(age), dtype=np.float32)
    t[b == 0] = (age[b == 0] - 10) / 40.0
    t[b == 1] = (age[b == 1] - 51) / 39.0
    t[b == 2] = np.clip((age[b == 2] - 90) / (39.0 * dil), 0, 1)
    return torch.from_numpy(t)

DEV_T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(DEV)

Xt = DEV_T(Xq_tr); Xs_ = DEV_T(Xs_tr); It = DEV_T(imm_tr)
age_z = DEV_T((age_tr - mu) / sd)
Yc = band_cum(age_tr).to(DEV)
Tb = tau_t(age_tr).to(DEV)
Rt = DEV_T(rda_tr); St = torch.from_numpy(C.encode_sex(lab_tr["sex"])).to(DEV)
n = len(Xt)

import pynvml
pynvml.nvmlInit()
NV = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_util():
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(NV).gpu
    except Exception:
        return 100  # probe failure must never shrink the batch

def gpu_free_gib():
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(NV).free / 2**30
    except Exception:
        return 99.0

class Bucket:
    """One static shape = one static buffer set + one captured graph."""
    def __init__(self, bs):
        self.bs = bs
        self.idx = torch.zeros(bs, dtype=torch.long, device=DEV)
        self.q = torch.zeros(bs, Xt.shape[1], device=DEV)
        self.s = torch.zeros(bs, Xs_.shape[1], device=DEV)
        self.i = torch.zeros(bs, It.shape[1], device=DEV)
        self.r = torch.zeros(bs, device=DEV)
        self.x = torch.zeros(bs, device=DEV)
        self.sex = torch.zeros(bs, device=DEV)
        self.yc = torch.zeros(bs, 2, device=DEV)
        self.tb = torch.zeros(bs, device=DEV)
        self.w = torch.zeros(bs, device=DEV)
        self.graph = None
        self.loss = None

    def fill(self, perm, i, remaining):
        take = min(self.bs, remaining)
        idx = perm[i:i + take]
        if take < self.bs:  # tail: wraparound pad (once per epoch, <=127 rows)
            idx = torch.cat([idx, perm[:self.bs - take]])
        self.idx.copy_(idx)
        self.q.copy_(Xt[self.idx]); self.s.copy_(Xs_[self.idx])
        self.i.copy_(It[self.idx]); self.r.copy_(Rt[self.idx])
        self.x.copy_(age_z[self.idx]); self.sex.copy_(St[self.idx])
        self.yc.copy_(Yc[self.idx]); self.tb.copy_(Tb[self.idx])
        self.w.copy_(torch.ones(self.bs, device=DEV))
        return take

    def compute(self, model):
        zero_bp = torch.zeros(self.bs, 3, device=DEV)  # no true-label leakage (R1 audit fix)
        out = model(self.q, self.s, self.sex, self.i, self.r, zero_bp)
        loss_reg = F.smooth_l1_loss(out["age"], self.x)
        eps = 5e-2
        t = self.yc.clamp(eps, 1 - eps) * 0.95 + 0.5 * 0.05
        loss_ord = F.binary_cross_entropy_with_logits(out["band_cum"], t).mean()
        loss_tau = F.smooth_l1_loss(out["tau"], self.tb)
        return loss_reg + loss_ord + 0.5 * loss_tau, out

def capture_buckets(model, opt, BUCKETS):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    perm = torch.randperm(n, device=DEV)
    out = []
    with torch.cuda.stream(side):
        for bs in BUCKETS:
            b = Bucket(bs)
            for _ in range(3):  # warmup: allocates workspaces for this shape
                opt.zero_grad(set_to_none=False)
                loss, _ = b.compute(model)
                loss.backward()
            torch.cuda.current_stream().wait_stream(side)
            opt.zero_grad(set_to_none=False)
            for p in model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                opt.zero_grad(set_to_none=False)
                loss, _ = b.compute(model)
                loss.backward()
            b.graph = g
            b.loss = loss   # static output tensor: replay writes into this storage
            out.append(b)
            with torch.cuda.stream(side):
                pass
    torch.cuda.current_stream().wait_stream(side)
    return out

def pick_bucket(buckets, target_bs, remaining):
    """Largest captured bucket <= min(target, remaining); smallest bucket if tail."""
    cap = min(target_bs, remaining)
    fit = [b for b in buckets if b.bs <= cap]
    if fit:
        return max(fit, key=lambda b: b.bs)
    return min(buckets, key=lambda b: b.bs)  # tail < smallest: wraparound pad

FLAGPRE = os.environ.get("CG_FLAGPRE", "cg_full")
for seed in [int(s) for s in args.seeds.split(",")]:
    if os.path.exists(os.path.join(RUNS, f"{FLAGPRE}_f{args.fold}_s{seed}.flag")):
        print(f"[cg] skip f{args.fold} s{seed} (flag)", flush=True); continue
    torch.manual_seed(seed); np.random.seed(seed)
    model = CraneZV3(n_genes=n_genes, assignment=assign, n_modules=256,
                     n_immune=It.shape[1], ridge_residual=args.rr).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, capturable=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    BUCKETS = sorted({min(b, n) for b in (int(x) for x in args.buckets.split(",")) if b <= max(n, 128)})
    buckets = capture_buckets(model, opt, BUCKETS)
    print(f"[cg] captured fold{args.fold} seed{seed} buckets={[b.bs for b in buckets]} n={n}", flush=True)

    # wgov state: batch target in bytes -> bucket index (same hysteresis as train_v3)
    target_bs = min(args.gbs, n)
    win = []
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0; k = 0
        i = 0
        while i < n:
            b = pick_bucket(buckets, target_bs, n - i)
            take = b.fill(perm, i, n - i)
            b.graph.replay()
            opt.step()
            tot += float(b.loss); k += 1
            i += take
            win.append(gpu_util())
            if len(win) % 25 == 0:  # governor: same thresholds as train_v3 wgov
                um = sum(win[-30:]) / min(len(win), 30)
                fg = gpu_free_gib()
                if um < 75 and fg > 1.5 and b.bs < buckets[-1].bs:
                    nxt = [x for x in buckets if x.bs > b.bs]
                    if nxt:
                        target_bs = nxt[0].bs
                elif (um > 97 or fg < 0.5) and b.bs > buckets[0].bs:
                    prv = [x for x in buckets if x.bs < b.bs]
                    if prv:
                        target_bs = prv[-1].bs
        sched.step()
        if ep % 15 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                ov = model(DEV_T(Xq_va), DEV_T(Xs_va), torch.from_numpy(C.encode_sex(lab_va["sex"])).to(DEV),
                           DEV_T(imm_va), DEV_T(rda_va), torch.zeros(len(Xq_va), 3, device=DEV))
                M = ov["age"].cpu().numpy() * sd + mu
                from sklearn.metrics import roc_auc_score
                bv = np.where(age_va >= 90, 2, np.where(age_va > 50, 1, 0))
                bl = ov["band_logits"].cpu().numpy()
                aucs = [roc_auc_score((bv == k2).astype(int), bl[:, k2]) for k2 in range(3)]
                u = gpu_util()
            model.train()
            print(f"[cg] ep{ep} loss={tot / max(1, k):.4f} MAE={np.abs(M - age_va).mean():.2f} "
                  f"rho={float(pd.Series(M).corr(pd.Series(age_va), method='spearman')):.3f} "
                  f"mAUC={np.mean(aucs):.3f} bs={b.bs} util={u}% elapsed={round((time.time() - t0) / 60, 1)}min", flush=True)
    # persist final val predictions for full-cohort OOF certification
    model.eval()
    with torch.no_grad():
        ov = model(DEV_T(Xq_va), DEV_T(Xs_va), torch.from_numpy(C.encode_sex(lab_va["sex"])).to(DEV),
                   DEV_T(imm_va), DEV_T(rda_va), torch.zeros(len(Xq_va), 3, device=DEV))
    np.savez(os.path.join(RUNS, f"{FLAGPRE}_M_f{args.fold}_s{seed}.npz"),
             ids=np.array(lab_va.index.astype(str), dtype="U16"),
             M=ov["age"].cpu().numpy() * sd + mu, age=age_va)
    if os.environ.get("CG_SAVECKPT"):
        ck_dir = os.path.join(RUNS, "ckpt_cg")
        os.makedirs(ck_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "fold": args.fold, "seed": seed},
                   os.path.join(ck_dir, f"{FLAGPRE}_f{args.fold}_s{seed}.pt"))
    open(os.path.join(RUNS, f"{FLAGPRE}_f{args.fold}_s{seed}.flag"), "w").write("done")
    print("[cg] done", flush=True)
