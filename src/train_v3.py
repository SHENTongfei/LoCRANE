# -*- coding: utf-8 -*-
"""train_v3.py - TAIL-CERT trainer with H58 screen mode (stage-0 baseline + arms).

Stage-0 (baseline v3): data chain v3 + low-rank FiLM + ordinal + band-pos(dilated)
                       + age head, NO W1 items (MoE/gate/PLM/contrastive off).
Arms (W1, one variable at a time): --arm gate|plm|moe|tv1.0|contrastive
Imbalance: band inverse-frequency weights + 10y age-bucket regression weights
           (labels_v3.csv); optional --balanced-sampler arm.
Discipline: heartbeat file (60s), run_registry.json append, alpha=1.0 fixed,
early stop = macro ordinal AUC on outer val (pre-registered), GPU-only.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # R2 CPU-reduction preset (2026-09-27): 4->2, backup train_v3.py.bak_cpu
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import sys, json, time, argparse, threading
import msvcrt  # Windows cross-process file locks (registry race fix)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("CRANE_DATA_DIR", "C:/Users/TS/Desktop/crane")
import common as C
from model_v3 import CraneZV3, count_params

ROOT = os.path.dirname(HERE)
DATA3 = os.path.join(ROOT, "data_v3")
RUNS = os.path.join(ROOT, "runs_v3")
os.makedirs(RUNS, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BAND2I = {"Y": 0, "M": 1, "L": 2}
EPOCHS = int(os.environ.get("V3_EPOCHS", 300))
BATCH = 64
LR = 5e-4



class AdaptiveBatcher:
    """EB/NT canon: dynamic batch on step-time + GPU-free feedback."""
    def __init__(self, initial=64, floor=16, cap=512):
        self.bs, self.floor, self.cap = initial, floor, cap
        self._last = None
    def gpu_free_gib(self):
        try:
            f, _ = torch.cuda.mem_get_info()
            return f / 2**30
        except Exception:
            return 99.0
    def update(self, dt, batch):
        per = dt / max(1, batch)
        if self._last is not None:
            slow = per > 2.0 * self._last
            fast = per < 0.6 * self._last
            free = self.gpu_free_gib()
            if slow or free < 2.0:
                self.bs = max(self.floor, int(self.bs * 0.7))
            elif fast and free > 6.0:
                self.bs = min(self.cap, int(self.bs * 1.4))
        self._last = per
        return self.bs


class UtilWindow:
    """60s window GPU-util sampler (NVML, 5s ticks); governor reads window mean.
    If NVML dies, returns None -> governor keeps last batch (caliber-safe)."""
    def __init__(self):
        import threading
        self.samples, self.lock, self.stop = [], threading.Lock(), False
        def loop():
            import time as _t
            try:
                import pynvml
                pynvml.nvmlInit()
                dev = pynvml.nvmlDeviceGetHandleByIndex(0)
                while not self.stop:
                    with self.lock:
                        self.samples.append(pynvml.nvmlDeviceGetUtilizationRates(dev).gpu)
                        self.samples = self.samples[-40:]  # 80s ring at 2s tick
                    _t.sleep(2)
            except Exception:
                pass
        threading.Thread(target=loop, daemon=True).start()
    def window_mean(self, window_s=60):
        with self.lock:
            n = max(1, window_s // 5)
            s_ = self.samples[-n:]
        return (sum(s_) / len(s_)) if s_ else None

def heartbeat(path, stop):
    while not stop.is_set():
        open(path, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        stop.wait(30)


# ---- registry race fix: concurrent same-arm jobs share registry_{arm}_{tag}.json;
# lock the file for read/write and merge on save so no worker's entries are lost
# and a mid-write truncate can never be read (caliber-neutral: IO only). ----
def _reg_lock(reg_path, timeout_s=300.0):
    lf = open(reg_path + ".lock", "a+")
    t0 = time.time()
    while True:
        try:
            lf.seek(0)
            msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
            return lf
        except OSError:
            if time.time() - t0 > timeout_s:
                lf.close()
                raise RuntimeError(f"[reg-lock] timeout waiting on {reg_path}.lock")
            time.sleep(0.5)


def _reg_unlock(lf):
    try:
        lf.seek(0)
        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    lf.close()


def reg_load(reg_path):
    lf = _reg_lock(reg_path)
    try:
        if not os.path.exists(reg_path):
            return []
        for _ in range(10):
            try:
                return json.load(open(reg_path))
            except (json.JSONDecodeError, OSError):
                time.sleep(0.5)
        return []
    finally:
        _reg_unlock(lf)


def reg_save(reg_path, registry):
    key = lambda r: (r.get("fold"), r.get("seed"))
    tmp = f"{reg_path}.tmp.{os.getpid()}"
    lf = _reg_lock(reg_path)
    try:
        cur = []
        if os.path.exists(reg_path):
            try:
                cur = json.load(open(reg_path))
            except (json.JSONDecodeError, OSError):
                cur = []
        seen = {key(r) for r in cur}
        merged = cur + [r for r in registry if key(r) not in seen]
        json.dump(merged, open(tmp, "w"), indent=1)
        os.replace(tmp, reg_path)
    finally:
        _reg_unlock(lf)


PLM_ARR = None; PLM_IDS = None; PLM_IDSET = None

def load_all():
    global PLM_ARR, PLM_IDS, PLM_IDSET
    lab = pd.read_csv(os.path.join(DATA3, "labels_v3.csv")).set_index("ID")
    folds = json.load(open(os.path.join(DATA3, os.environ.get("V3_FOLDS", "folds_v3.json"))))
    z = np.load(os.path.join(DATA3, "plm_esm2_650M_hvg3000.npz"))
    PLM_ARR = z["X"]; PLM_IDS = z["ids"]; PLM_IDSET = set(PLM_IDS)
    if os.environ.get("V3_LORA_TRUE", "0") == "1":
        lt = os.path.join(DATA3, "plm_esm2_650M_hvg3000_loraTrue.npz")
        if os.path.exists(lt):
            zt = np.load(lt); PLM_ARR = zt["X"].astype(np.float32)
            print(f"[loraTrue] using task-adapted ESM2-650M embeddings ({PLM_ARR.shape})", flush=True)
        else:
            raise FileNotFoundError("V3_LORA_TRUE=1 but " + lt + " missing - run lora_true_build.py first")
    return lab, folds


def band_targets(age):
    b = np.where(age >= 90, 2, np.where(age > 50, 1, 0))
    y = np.zeros((len(age), 2), dtype=np.float32)  # cumulative targets K-1
    y[:, 0] = (b >= 1).astype(np.float32)
    y[:, 1] = (b == 2).astype(np.float32)
    return b, torch.from_numpy(y)


def dilated_tau(age, band, dil=1.4):
    """band position in [0,1]; L-band span dilated by 1.4x (deceleration prior)."""
    t = np.zeros(len(age), dtype=np.float32)
    t[band == 0] = (age[band == 0] - 10) / 40.0
    t[band == 1] = (age[band == 1] - 51) / 39.0
    span = 39.0 * dil
    t[band == 2] = np.clip((age[band == 2] - 90) / span, 0, 1)
    return t


def ordinal_ce(cum_logits, cum_target, eps=1e-4, smooth=0.05):
    t = cum_target.clamp(eps, 1 - eps)
    t = t * (1 - smooth) + 0.5 * smooth  # P5: real label smoothing (was clamp-only)
    return F.binary_cross_entropy_with_logits(cum_logits, t)


def macro_ordinal_auc(band_logits, b):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(3):
        yk = (b == k).astype(int)
        if yk.min() == yk.max():
            continue
        aucs.append(roc_auc_score(yk, band_logits[:, k]))
    return float(np.mean(aucs)) if aucs else 0.5


def run_fold(lab, folds, fold, seeds, args, registry):
    done = {(r.get("fold"), r.get("seed")) for r in registry}
    seeds = [s for s in seeds if (fold, s) not in done]
    if not seeds:
        print(f"[v3] fold{fold} all seeds done, skip", flush=True)
        return []
    p_all = pd.read_csv(os.path.join(C.FOLD_DIR, "fold_0", "train_expr.csv"), index_col=0)
    genes = list(p_all.columns)  # full-gene fold CSV space == HVG source space
    # fixed pipeline (fold0-train fitted), inherited from v2 protocol for comparability
    cfg_genes = C.select_hvg(p_all, 3000)
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.cluster import KMeans
    X0g = p_all[cfg_genes]
    # FIX (full-cohort): fit scalers on the CURRENT fold's own aligned inner-train
    # features (leakage-free), NOT on the 240-row pool-era X0g reference whose
    # quantile map saturates on full-cohort samples (rho -0.02 collapse root cause).
    qsc = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    zsc = StandardScaler()
    kmcache = os.path.join(DATA3, f"km256_f{fold}.npz")
    if os.path.exists(kmcache):
        assign = {int(k): int(v) for k, v in zip(np.load(kmcache)["g"], np.load(kmcache)["m"])}
    else:
        km = KMeans(n_clusters=256, random_state=42, n_init=10).fit(
            ((X0g - X0g.mean(0)) / (X0g.std(0) + 1e-8)).T.values)
        assign = {i: int(m) for i, m in enumerate(km.labels_)}
        np.savez(kmcache, g=np.arange(len(km.labels_)), m=km.labels_)
    lm22 = pd.read_csv(C.LM22_PATH, sep="\t", index_col=0)
    qc = json.load(open(os.path.join(DATA3, "lm22_qc.json")))
    lm22q = lm22.loc[[g for g in lm22.index if g in p_all.columns], qc["keep_cells"]]

    val_ids = set(folds[str(fold)])
    all_ids = [str(i) for i in lab.index]
    inner_ids = [i for i in all_ids if i not in val_ids]

    def fe(df_expr):
        X = C.align_features(df_expr, cfg_genes)
        return qsc.transform(X), zsc.transform(X)

    def imm_of(df_expr):
        fr = C.lm22_deconvolution(df_expr, lm22q)
        return fr[qc["keep_cells"]].to_numpy(np.float32)

    def plm_of(ids):
        idx = [plm_ids.get(g, -1) for g in cfg_genes]
        out = np.zeros((len(ids), len(cfg_genes), plmz.shape[1]), dtype=np.float32)
        hit = np.zeros(len(cfg_genes), dtype=np.float32)
        for j, k in enumerate(idx):
            if k >= 0:
                out[:, j] = plmz[k]; hit[j] = 1
        return torch.from_numpy(out), torch.from_numpy(hit)

    # expression for inner train / outer val (full-gene CSVs exist only for the
    # 324-pool folds; inner-train extras (non-pool samples) live in the full matrix)
    full = pd.read_csv(os.path.join(C.BASE_DIR, "expression_matrix.tsv.gz"), sep="\t",
                       index_col=0, compression=None).T  # samples x genes
    Xtr_df = full.loc[inner_ids]
    Xva_df = full.loc[sorted(val_ids)]  # REVERT FIX1: train/val must share feature space (same 0-filled columns)
    cfg_genes_eff = cfg_genes
    plm_in = None
    if args.arm in ("plm", "lora"):  # R2 lora proxy reuses the same frozen-slice module averages
        row = {g: i for i, g in enumerate(PLM_IDS)}
        P_mod = np.zeros((256, PLM_ARR.shape[1]), np.float32); cntm = np.zeros(256)
        for j, g in enumerate(cfg_genes_eff):
            if g in PLM_IDSET:
                P_mod[assign[j]] += PLM_ARR[row[g]]; cntm[assign[j]] += 1
        P_mod /= np.maximum(cntm, 1)[:, None]
        plm_in = torch.from_numpy(P_mod[None]).to(DEV)  # (1,256,1280) module-level, sample-independent

    # ---- fe cache (CPU once per fold; NT discipline: CPU relief) ----
    fcache = os.path.join(DATA3, f"fe_cache_f{fold}{os.environ.get('V3_FESUF', '')}.npz")
    if os.path.exists(fcache) and args.fe_cache:
        z = np.load(fcache)
        Xq_tr, Xs_tr, imm_tr = z["Xq_tr"], z["Xs_tr"], z["imm_tr"]
        Xq_va, Xs_va, imm_va = z["Xq_va"], z["Xs_va"], z["imm_va"]
        print(f"[fe-cache] loaded fold{fold}", flush=True)
    else:
        _Xa = C.align_features(Xtr_df, cfg_genes)
        qsc.fit(_Xa); zsc.fit(_Xa)
        Xq_tr, Xs_tr = fe(Xtr_df); Xq_va, Xs_va = fe(Xva_df)
        imm_tr = imm_of(Xtr_df); imm_va = imm_of(Xva_df)
        if args.fe_cache:
            np.savez_compressed(fcache, Xq_tr=Xq_tr, Xs_tr=Xs_tr, imm_tr=imm_tr,
                                Xq_va=Xq_va, Xs_va=Xs_va, imm_va=imm_va)
            print(f"[fe-cache] saved fold{fold}", flush=True)

    lab_tr = lab.loc[[str(i) for i in Xtr_df.index]]; lab_va = lab.loc[[str(i) for i in Xva_df.index]]
    age_tr = lab_tr["age"].to_numpy(np.float32); age_va = lab_va["age"].to_numpy(np.float32)
    b_tr, Ycum_tr = band_targets(age_tr); b_va, _ = band_targets(age_va)
    tau_tr = dilated_tau(age_tr, b_tr, dil=(1.0 if args.arm == "tv1.0" else 1.4))
    w_band_tr = lab_tr["w_band"].to_numpy(np.float32)
    w_ab_tr = lab_tr["w_agebucket"].to_numpy(np.float32)
    sex_tr = C.encode_sex(lab_tr["sex"]); sex_va = C.encode_sex(lab_va["sex"])
    bp_tr = np.eye(3, dtype=np.float32)[b_tr]
    ridge = None
    from sklearn.linear_model import Ridge as _Ridge
    ridge = _Ridge(alpha=1.0).fit(Xq_tr, age_tr)  # P2: anchor on Xq view (diag: 17.3 vs 32.2 MAE)
    # FIX2: denorm caliber = fold0-train (v2-aligned); inner-train median 72 made
    # constant-collapse predictions cost MAE 32 on the two-band val pool
    p0 = C.load_fold(0)[2]
    mu = float(p0["age"].mean()); sd = max(1e-6, float(p0["age"].std()))
    rda_tr = (ridge.predict(Xq_tr) - mu) / sd
    rda_va = (ridge.predict(Xq_va) - mu) / sd
    XVq = torch.from_numpy(Xq_va).float().to(DEV); XVs = torch.from_numpy(Xs_va).float().to(DEV)
    SV = torch.from_numpy(sex_va).to(DEV); IV = torch.from_numpy(imm_va).to(DEV)
    RV = torch.from_numpy(rda_va).float().to(DEV); BV = torch.from_numpy(np.eye(3, dtype=np.float32)[b_va]).to(DEV)

    results = []
    reg_path = os.path.join(RUNS, f"registry_{args.arm}_{args.tag}.json")
    CKPT_DIR = os.path.join(RUNS, "ckpt"); os.makedirs(CKPT_DIR, exist_ok=True)

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        resume_ep, resume_best, resume_bad, resume_best_state = 0, -1e9, 0, None
        ck = os.path.join(CKPT_DIR, f"{args.arm}_{args.tag}_f{fold}_s{seed}_ep.pt")
        use_moe = args.arm in ("moe", "combo")          # R2 combo: gate+moe together
        use_gate = args.arm in ("gate", "combo")
        n_plm = int(os.environ.get("V3_NPLM", "256")) if args.arm in ("plm", "lora") else 0  # V3_NPLM override (lora arm slowdown fix)
        plm_bn = 256 if args.arm == "lora" else 128     # R2 lora = PROXY (peft not installed): frozen slice + widened bottleneck
        model = CraneZV3(n_genes=len(cfg_genes_eff), assignment={i: assign[i] for i in range(len(cfg_genes))},
                         n_modules=256, n_immune=imm_tr.shape[1],
                         use_moe=use_moe, use_gate=use_gate,
                         use_contra=(args.arm == "contra"),
                         n_plm=n_plm, plm_bottleneck=plm_bn).to(DEV)
        if os.environ.get("V3_COMPILE", "0") == "1":  # needs triton; sunflower env lacks it
            model = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        if os.path.exists(ck):  # epoch-level resume (user order: checkpoint resume)
            try:
                d = torch.load(ck, map_location=DEV, weights_only=False)
                model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"])
                sched.load_state_dict(d["sched"])
                resume_ep, resume_best, resume_bad = d["ep"] + 1, d.get("best", -1e9), d.get("bad", 0)
                resume_best_state = d.get("best_state")
                print(f"[ckpt] resumed {os.path.basename(ck)} at ep{resume_ep}", flush=True)
            except Exception as e:
                print("[ckpt] load failed, fresh start:", str(e)[:80], flush=True)
        # GPU-resident: whole dataset lives on device (1651x3000 ~= 20MB); zero H2D per step
        Xt = torch.from_numpy(Xq_tr).float().to(DEV); Xs_ = torch.from_numpy(Xs_tr).float().to(DEV)
        Yc = Ycum_tr.to(DEV); Tb = torch.from_numpy(tau_tr).to(DEV)
        Wb = torch.from_numpy(w_band_tr).to(DEV); Wa = torch.from_numpy(w_ab_tr).to(DEV)
        Rt = torch.from_numpy(rda_tr).float().to(DEV)
        St = torch.from_numpy(sex_tr).to(DEV); It = torch.from_numpy(imm_tr).to(DEV)
        Bp = torch.from_numpy(bp_tr).to(DEV)
        age_z_tr = torch.from_numpy((age_tr - mu) / sd).float().to(DEV)
        NOES = os.environ.get("V3_NOES", "0") == "1"
        n = len(Xt); best, best_state, bad = resume_best, resume_best_state, resume_bad
        BCAP = int(os.environ.get("V3_BCAP", "1024"))  # user order: chase batch (VRAM-guarded by shrink rule)  # back to 512: 4-proc WDDM cap saturates VRAM (~118MiB free @512); 1024 would OOM, not speed up
        ab = AdaptiveBatcher(initial=min(BCAP, n), floor=64, cap=BCAP)  # step-starvation fix (diag E-evidence)
        UW = UtilWindow(); UW.gpu_free_gib = lambda: ab.gpu_free_gib()


        W_REG = float(os.environ.get("V3_WREG", "1"))  # default 1: 100 default blew up delta (MAE 30 vs 17.3, grid audit 05:5x)  # reg loss ~0.011 vs ord ~0.7: 1.6% share starved delta (probe-verified); hoisted for tailft phase-2
        import time as _t
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, device=DEV)
            i = 0
            while i < n:
                bs = max(1, min(ab.bs, n - i))
                idx = perm[i:i + bs]
                t_step = _t.time()
                out = model(Xt[idx], Xs_[idx], St[idx], It[idx], Rt[idx], Bp[idx], plm=plm_in)
                if args.arm == "bktw":
                    loss_reg = (F.smooth_l1_loss(out["age"], age_z_tr[idx], reduction="none")
                                * Wa[idx]).mean()  # R2 bktw: reg loss x 10y age-bucket inverse-freq weight (protocol lock caliber)
                else:
                    loss_reg = F.smooth_l1_loss(out["age"], age_z_tr[idx], reduction="none").mean()  # plain; bucket-w is its own arm
                loss_ord = (ordinal_ce(out["band_cum"], Yc[idx]).mean(-1) * Wb[idx]).mean()
                loss_tau = (F.smooth_l1_loss(out["tau"], Tb[idx],
                                             reduction="none") * Wb[idx]).mean()
                loss = W_REG * loss_reg + 1.0 * loss_ord + 0.5 * loss_tau
                if out["gate_w"] is not None:
                    loss = loss + 1e-3 * out["gate_w"].abs().mean()
                if model.blocks[-1].use_moe and model.blocks[-1].ff.last_usage is not None:
                    wu = model.blocks[-1].ff.last_usage.mean(0)
                    loss = loss + 1e-2 * (wu * torch.log(wu + 1e-9)).sum()
                if args.arm == "contra":
                    loss = loss + 0.1 * out["contra_loss"]  # R2 contra: 3-view supcon (q/s/imm), temp 0.1, weight 0.1
                opt.zero_grad(); loss.backward(); opt.step()
                ab.update(_t.time() - t_step, bs)
                i += bs
            sched.step()
            if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:  # epoch-level ckpt (loss granularity ~2 min)
                tmp = ck + ".tmp"
                torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "best": best, "bad": bad,
                            "best_state": best_state}, tmp)
                os.replace(tmp, ck)
            if ep % 15 == 0:  # wgov: 15ep cadence + 20s window (denser probe, user order reboot+1)
                um = UW.window_mean(20)
                if um is not None:
                    free_g = UW.gpu_free_gib()
                    # user order 00:3x: chase batch cap whenever VRAM allows (no more hold-band)
                    if um < 98 and ab.bs < ab.cap and free_g > 1.5:
                        ab.bs = min(ab.cap, int(ab.bs * 2))
                    elif (um > 98 and free_g < 1.0) or free_g < 0.5:  # shrink only on OOM-risk
                        ab.bs = max(ab.floor, int(ab.bs * 0.67))
                    print(f"[wgov] ep{ep} util_w30s={um:.0f}% free={free_g:.1f}GiB batch={ab.bs} target>=85%", flush=True)
            if ep % 50 == 0:
                print(f"[prog] {args.arm} f{fold} s{seed} ep{ep} score={score if 'score' in dir() else best}", flush=True)
            if ep % 5 == 0 or ep == EPOCHS - 1:  # eval-every-5: per-epoch .cpu() syncs caused the 23-40% util dips (window audit 16:4x)
                model.eval()
                with torch.no_grad():
                    out = model(XVq, XVs, SV, IV, RV, BV, plm=plm_in)
                    mA = macro_ordinal_auc(out["band_logits"].cpu().numpy(), b_va)
                    M_e = out["age"].cpu().numpy() * sd + mu
                    score = mA - float(np.abs(M_e - age_va).mean()) / 50.0  # age-aware early stop (diag P4)
                if NOES or (ep >= 50 and score > best):  # V3_NOES=1: take final epoch
                    best, best_state, bad = mA, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                elif not NOES:
                    bad += 1
                    if args.arm != "tailft" and ep + 1 >= 60 and bad >= 30:  # 30 evals x 5ep = 150ep patience
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        if args.arm == "tailft":
            # R2 tailft phase-2: L-band (age>=90) samples only, 20ep, tail reg target weight x3
            TAIL_EP = int(os.environ.get("V3_TAIL_EP", "20"))
            idxL = torch.from_numpy(np.where(b_tr == 2)[0].astype(np.int64)).to(DEV)
            opt2 = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)  # ft-LR = main LR / 10 (noted in R2 report)
            for ep2 in range(TAIL_EP):
                model.train()
                perm = idxL[torch.randperm(len(idxL), device=DEV)]
                i = 0
                while i < len(perm):
                    bs2 = max(1, min(ab.bs, len(perm) - i))
                    idx = perm[i:i + bs2]
                    out = model(Xt[idx], Xs_[idx], St[idx], It[idx], Rt[idx], Bp[idx], plm=plm_in)
                    l_reg = 3.0 * W_REG * F.smooth_l1_loss(out["age"], age_z_tr[idx], reduction="none").mean()
                    l_ord = (ordinal_ce(out["band_cum"], Yc[idx]).mean(-1) * Wb[idx]).mean()
                    l_tau = (F.smooth_l1_loss(out["tau"], Tb[idx], reduction="none") * Wb[idx]).mean()
                    loss2 = l_reg + 1.0 * l_ord + 0.5 * l_tau
                    opt2.zero_grad(); loss2.backward(); opt2.step()
                    i += bs2
                model.eval()
                with torch.no_grad():
                    out = model(XVq, XVs, SV, IV, RV, BV, plm=plm_in)
                    mA2 = macro_ordinal_auc(out["band_logits"].cpu().numpy(), b_va)
                    M2 = out["age"].cpu().numpy() * sd + mu
                    score2 = mA2 - float(np.abs(M2 - age_va).mean()) / 50.0
                if score2 > best:
                    best, best_state = mA2, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if best_state is not None:
                model.load_state_dict(best_state)  # final ckpt = best across main + tail phases
            print(f"[v3] tailft phase2: {TAIL_EP}ep on L-band n={len(idxL)} (reg weight x3)", flush=True)
        model.eval()
        pass
        with torch.no_grad():
            out = model(XVq, XVs, SV, IV, RV, BV, plm=plm_in)
            M = out["age"].cpu().numpy() * sd + mu
            bl = out["band_logits"].cpu().numpy()
            band_pred = bl.argmax(1)
        res = {"fold": fold, "seed": seed, "arm": args.arm, "best_macroAUC": round(best, 4),
               "MAE": round(float(np.abs(M - age_va).mean()), 3),
               "spearman_age_M": round(float(pd.Series(M).corr(pd.Series(age_va), method="spearman")), 4),
               "band_acc": round(float((band_pred == b_va).mean()), 4)}
        results.append(res)
        print("[v3]", json.dumps(res), flush=True)
        registry.append({"ts": time.strftime("%H:%M:%S"), **res})
        reg_save(reg_path, registry)  # per-seed flush: crash loses <=1 seed, not the job
        if os.path.exists(ck):
            os.remove(ck)
        if os.environ.get("V3_SAVECKPT", "0") == "1" and best_state is not None:
            b3p = os.path.join(CKPT_DIR, f"{args.arm}_{args.tag}_f{fold}_s{seed}_best.pt")
            torch.save({"model": best_state, "seed": seed, "fold": fold,
                        "mae": res.get("MAE"), "mauc": res.get("best_macroAUC")}, b3p + ".tmp")
            os.replace(b3p + ".tmp", b3p)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="stage0",
                    choices=["stage0", "gate", "plm", "moe", "tv1.0",
                             "combo", "contra", "bktw", "tailft", "lora"])
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--tag", default="screen")
    ap.add_argument("--fe_cache", type=int, default=1)
    args = ap.parse_args()
    stop = threading.Event()
    hb = os.path.join(RUNS, f"heartbeat_{args.arm}_{args.tag}.hb")
    threading.Thread(target=heartbeat, args=(hb, stop), daemon=True).start()
    lab, folds = load_all()
    reg_path = os.path.join(RUNS, f"registry_{args.arm}_{args.tag}.json")
    registry = reg_load(reg_path)
    t0 = time.time()
    out = []
    for f in [int(x) for x in args.folds.split(",")]:
        out += run_fold(lab, folds, f, [int(s) for s in args.seeds.split(",")], args, registry)
    reg_save(reg_path, registry)
    stop.set()
    print(json.dumps({"arm": args.arm, "tag": args.tag, "minutes": round((time.time() - t0) / 60, 1),
                      "results": out}, indent=1))


if __name__ == "__main__":
    main()
