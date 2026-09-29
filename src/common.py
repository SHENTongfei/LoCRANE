"""common.py - CRANE-Z v2 shared utilities.

Paths, data loading, HVG selection, LM22 immune deconvolution (CIBERSORT-style
NNLS), gene module construction (correlation clustering), z-score scaling.

All feature engineering is fitted on the TRAIN fold only (no leakage).
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls

# ---------------------------------------------------------------- paths ----
BASE_DIR = os.environ.get("CRANE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
FOLD_DIR = os.path.join(BASE_DIR, "training_data", "balanced_folds")
EXT_DIR = os.path.join(BASE_DIR, "external_test_data")
FULL_EXPR_PATH = os.path.join(BASE_DIR, "expression_matrix.tsv.gz")   # 1715 x Ngenes (plain tsv)
FULL_PHENO_PATH = os.path.join(BASE_DIR, "phenotype.csv")
LM22_PATH = os.path.join(BASE_DIR, "LM22_signature.tsv")
EXT_BLOOD_PATH = os.path.join(EXT_DIR, "gtex_validation_balanced.csv")
EXT_MUSCLE_PATH = os.path.join(EXT_DIR, "gtex_validation_muscle.csv")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
AGE_SPAN = 113.0 - 10.0  # full-cohort age span, used for mae_score


# ------------------------------------------------------------ data load ----
def load_fold(fold):
    """Load train/val expr + pheno for one fold. Returns DataFrames."""
    d = os.path.join(FOLD_DIR, f"fold_{fold}")
    X_tr = pd.read_csv(os.path.join(d, "train_expr.csv"), index_col=0)
    X_va = pd.read_csv(os.path.join(d, "val_expr.csv"), index_col=0)
    p_tr = pd.read_csv(os.path.join(d, "train_pheno.csv"), index_col=0)
    p_va = pd.read_csv(os.path.join(d, "val_pheno.csv"), index_col=0)
    return X_tr, X_va, p_tr, p_va


def load_external():
    """Load GTEx external validation sets (blood + muscle)."""
    blood = pd.read_csv(EXT_BLOOD_PATH, index_col=0)
    muscle = pd.read_csv(EXT_MUSCLE_PATH, index_col=0)
    return blood, muscle


def load_full_cohort():
    """Load the full 1715-sample cohort (unlabeled, for self-supervised
    pre-training). expression_matrix.tsv.gz is a plain TSV despite the
    .gz suffix."""
    expr = pd.read_csv(FULL_EXPR_PATH, sep="\t", index_col=0, compression=None)
    pheno = pd.read_csv(FULL_PHENO_PATH)
    pheno = pheno.set_index("ID")
    return expr, pheno


# --------------------------------------------------------- feature eng. ----
def select_hvg(expr_tr, k=3000):
    """Top-k highly variable genes, fitted on train fold only.
    Returns list of gene names (keeps only genes present in all sets).
    """
    # variance on log1p scale (expression appears to be log-like already)
    v = expr_tr.var(axis=0).sort_values(ascending=False)
    return list(v.index[:k])


def align_features(expr, gene_list):
    """Subset expr (samples x genes) to gene_list, fill missing genes with 0,
    return numpy float64. NOTE: caller must ensure expr is samples x genes
    (full cohort file is genes x samples -- transpose before calling)."""
    expr = expr.reindex(columns=gene_list)
    expr = expr.fillna(0.0)
    return expr.to_numpy(dtype=np.float64)


def lm22_deconvolution(expr_df, lm22_df):
    """CIBERSORT-style immune deconvolution via non-negative least squares.

    expr_df: DataFrame samples x genes (raw/TPM-like values, positive).
    lm22_df: DataFrame genes x 22 cell types (LM22 signature).
    Returns DataFrame samples x 22 cell proportions (rows sum to 1).
    """
    common = [g for g in lm22_df.index if g in expr_df.columns]
    B = lm22_df.loc[common].to_numpy(dtype=np.float64)   # genes x 22
    X = expr_df[common].to_numpy(dtype=np.float64)        # samples x genes
    frac = np.zeros((X.shape[0], B.shape[1]))
    for i in range(X.shape[0]):
        y = np.clip(X[i], 0.0, None)
        sol, _ = nnls(B, y)
        s = sol.sum()
        frac[i] = sol / s if s > 0 else np.full(B.shape[1], 1.0 / B.shape[1])
    return pd.DataFrame(frac, index=expr_df.index, columns=list(lm22_df.columns))


def build_gene_modules(expr_tr, n_modules=256, seed=42):
    """Cluster HVG genes into modules via KMeans on the transposed
    (genes x samples) matrix. Returns {gene: module_id}, module count.
    Fitted on train fold only.
    """
    from sklearn.cluster import KMeans
    G = expr_tr.to_numpy(dtype=np.float64)          # samples x genes
    Gt = (G - G.mean(0)) / (G.std(0) + 1e-8)        # gene x sample (z per gene)
    km = KMeans(n_clusters=n_modules, random_state=seed, n_init=10)
    labels = km.fit_predict(Gt.T)
    return dict(zip(expr_tr.columns, labels)), km


def make_quantile_scaler(expr_tr):
    """QuantileTransformer (rank-gauss) fitted on train fold only.
    Robust to cross-cohort distribution shift (used for the gene
    classification branch)."""
    from sklearn.preprocessing import QuantileTransformer
    n = max(10, min(1000, expr_tr.shape[0]))
    sc = QuantileTransformer(n_quantiles=n, output_distribution="normal",
                             random_state=42)
    sc.fit(expr_tr)
    return sc


def make_std_scaler(expr_tr):
    """StandardScaler (z-score) fitted on train fold only.
    Better for age regression (rank-gauss hurt r2); used by reg branch."""
    sc = StandardScaler()
    sc.fit(expr_tr)
    return sc


def encode_sex(sex_series):
    """Map sex to float. Train folds use 'M'/'F'; GTEx external uses 1/2
    (1=male, 2=female). Female -> 1.0, Male -> 0.0.
    Returns numpy float32 array (n,)."""
    def _m(v):
        v = str(v)
        if v in ("F", "2", "female", "Female"):
            return 1.0
        return 0.0
    return np.array([_m(v) for v in sex_series], dtype=np.float32)


def make_longevity_label(pheno_df):
    """longevity = 1 if group == 'RLL' else 0."""
    return np.array((pheno_df["group"] == "RLL").astype(int), dtype=np.float32)


# ------------------------------------------------------------ OPS (v1 compat)
def _cls_metrics(long_true, pred_label):
    """numpy-based acc/prec/rec/f1 (avoids sklearn multilabel bug on
    numpy 2.x + py3.11)."""
    yt = np.asarray(long_true, dtype=np.int64)
    yp = np.asarray(pred_label, dtype=np.int64)
    tp = float(np.sum((yt == 1) & (yp == 1)))
    fp = float(np.sum((yt == 0) & (yp == 1)))
    fn = float(np.sum((yt == 1) & (yp == 0)))
    n = max(1, len(yt))
    acc = (tp + float(np.sum((yt == 0) & (yp == 0)))) / n
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return acc, prec, rec, f1


def compute_ops_internal(age_true, age_pred, long_true, long_prob, age_span=AGE_SPAN):
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    age_true = np.asarray(age_true); age_pred = np.asarray(age_pred)
    long_true = np.asarray(long_true); long_prob = np.asarray(long_prob)
    mae = mean_absolute_error(age_true, age_pred)
    mae_score = max(0.0, 1.0 - mae / age_span) if age_span > 0 else 0.0
    r2 = float(r2_score(age_true, age_pred))
    r2_clipped = max(0.0, r2)
    auc = roc_auc_score(long_true, long_prob) if len(np.unique(long_true)) >= 2 else 0.5
    thresholds = np.arange(0.0, 1.001, 0.001)
    best_ops = -999.0
    best_metrics = None
    for th in thresholds:
        pred_label = (long_prob >= th).astype(int)
        acc, prec, rec, f1 = _cls_metrics(long_true, pred_label)
        ops = mae_score + r2_clipped + 2.0 * auc + acc + prec + rec + f1
        if ops > best_ops:
            best_ops = ops
            best_metrics = {"mae": float(mae), "mae_score": float(mae_score),
                            "r2": r2, "r2_clipped": float(r2_clipped),
                            "auc": float(auc), "acc": float(acc), "precision": float(prec),
                            "recall": float(rec), "f1": float(f1), "threshold": float(th)}
    return best_ops, best_metrics


def compute_ops_external(long_true, long_prob):
    from sklearn.metrics import roc_auc_score
    long_true = np.asarray(long_true); long_prob = np.asarray(long_prob)
    auc = roc_auc_score(long_true, long_prob) if len(np.unique(long_true)) >= 2 else 0.5
    thresholds = np.arange(0.0, 1.001, 0.001)
    best_ops = -999.0
    best_metrics = None
    for th in thresholds:
        pred_label = (long_prob >= th).astype(int)
        acc, prec, rec, f1 = _cls_metrics(long_true, pred_label)
        ops = 2.0 * auc + acc + prec + rec + f1
        if ops > best_ops:
            best_ops = ops
            best_metrics = {"auc": float(auc), "acc": float(acc), "precision": float(prec),
                            "recall": float(rec), "f1": float(f1), "threshold": float(th)}
    return best_ops, best_metrics
