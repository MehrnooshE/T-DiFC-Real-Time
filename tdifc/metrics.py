"""
Classification metrics and the scan-level cluster bootstrap, copied from the fusion
pipeline's scripts/common.py (used by evaluate_fusion_torch.py).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# Prevalence of peaks among ALL candidates of each split BEFORE the eval sets were balanced
# (split_report.txt: CTC / (CTC + artifact) original windows). For prevalence-adjusted PPV.
POOL_PREVALENCE = {"val": 234 / (234 + 1994), "test": 433 / (433 + 2300)}


def metrics_at(y, p, thr=0.5, prevalence=None):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    out = dict(
        n=len(y), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        ap=average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        acc=(tp + tn) / len(y), sens=sens, spec=spec, ppv=prec,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        bal_acc=np.nanmean([sens, spec]),
    )
    if prevalence is not None:
        q = prevalence
        den = sens * q + (1 - spec) * (1 - q)
        out["ppv_at_pool_prev"] = sens * q / den if den > 0 else np.nan
    return out


def cluster_bootstrap(y, p, groups, stat_fn, n_boot=2000, seed=0):
    """Resample SCANS with replacement (windows from one scan are not independent).
    Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = np.unique(g)
    by = {u: np.where(g == u)[0] for u in uniq}
    point = stat_fn(y, p)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([by[u] for u in pick])
        try:
            v = stat_fn(y[ii], p[ii])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def stat(name, thr=0.5):
    def f(y, p):
        if name == "auc":
            if len(np.unique(y)) < 2:
                raise ValueError
            return roc_auc_score(y, p)
        yhat = p >= thr
        if name == "sens":
            return yhat[y == 1].mean() if (y == 1).any() else np.nan
        if name == "spec":
            return (~yhat[y == 0]).mean() if (y == 0).any() else np.nan
        if name == "bal_acc":
            if len(np.unique(y)) < 2:
                raise ValueError
            return 0.5 * (yhat[y == 1].mean() + (~yhat[y == 0]).mean())
        raise KeyError(name)
    return f
