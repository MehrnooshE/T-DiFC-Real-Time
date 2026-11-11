"""
Evaluate the PyTorch `fusion_lr1e-4_wd_ls` ensemble on the validation and test windows.

Loads models/fusion_lr1e-4_wd_ls_torch.pt exactly as the app does (tdifc.model.FusionEnsemble),
re-scores the val / test windows, checks they match models/predictions_{val,test}.csv from
training, and writes
  models/metrics_at_0.6.csv     AUC, sensitivity, specificity, balanced accuracy at the cut-off
                                with 95% scan-level cluster-bootstrap CIs; PPV (as the pipeline's
                                06_evaluate.py, Table S2)
  models/torch_vs_keras.csv     the same metrics of the original Keras seeds
                                (models/keras_reference/) and the agreement of the two ensembles
                                on the same windows
Usage:  python evaluate_fusion_torch.py
"""
import os

import numpy as np
import pandas as pd

from tdifc import paths, spectral as sp
from tdifc import metrics as mt
from tdifc.model import FusionEnsemble

COL = paths.MODEL_NAME
KERAS_COL = "fusion_lr1e-4_wd_ls"


def table(P, col, thr):
    rows = []
    for s, d in P.items():
        y, p, g = d.label.values, d[col].values, d.scan.values
        r = mt.metrics_at(y, p, thr, prevalence=mt.POOL_PREVALENCE[s])
        out = {"split": s, "n": len(y), "TP": r["tp"], "FP": r["fp"], "TN": r["tn"], "FN": r["fn"]}
        for k in ("auc", "sens", "spec", "bal_acc"):
            pt, lo, hi = mt.cluster_bootstrap(y, p, g, mt.stat(k, thr), n_boot=2000, seed=1)
            out[k] = f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"
        out["PPV (balanced set)"] = round(r["ppv"], 3)
        out["PPV at pre-balancing prevalence"] = round(r["ppv_at_pool_prev"], 3)
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    ens = FusionEnsemble(paths.MODEL_PATH)
    thr = ens.cut_off
    print(f"{paths.MODEL_PATH}: {len(ens.nets)} seeds, best epochs {ens.info.get('best_epochs')}, cut-off {thr}")
    P = {}
    for s in ("val", "test"):
        X, y, M, meta = paths.load_split(s)
        p = ens.predict(X, M, als=sp.als_scaled(X, s, paths.SPLITS_ALS_DIR))
        f = os.path.join(paths.MODELS, f"predictions_{s}.csv")
        d = pd.read_csv(f) if os.path.exists(f) else meta.copy()
        if COL in d:
            print(f"{s}: re-scored vs training predictions, max |diff| {np.abs(d[COL].values - p).max():.1e}")
        d[COL] = p
        P[s] = d
    t = table(P, COL, thr)
    t.to_csv(os.path.join(paths.MODELS, f"metrics_at_{thr:g}.csv"), index=False)
    print("\nPyTorch ensemble")
    print(t.set_index("split").T.to_string())

    # ---- comparison with the original Keras seeds -------------------------------
    K = {s: pd.read_csv(os.path.join(paths.KERAS_REF_DIR, f"predictions_{s}.csv")) for s in P
         if os.path.exists(os.path.join(paths.KERAS_REF_DIR, f"predictions_{s}.csv"))}
    if len(K) != len(P):
        return
    tk = table(K, KERAS_COL, thr)
    rows = []
    for s in P:
        key = ["scan", "orig_idx", "augmentation"]
        m = P[s][key + ["label", COL]].merge(K[s][key + [KERAS_COL]], on=key, validate="one_to_one")
        a, b = m[COL].values, m[KERAS_COL].values
        rows.append({"split": s, "n": len(m), "pearson_r": round(np.corrcoef(a, b)[0, 1], 4),
                     "median_abs_diff": round(float(np.median(np.abs(a - b))), 4),
                     "same_call_at_cut": round(float(np.mean((a >= thr) == (b >= thr))), 4),
                     "torch_only_detected": int(((a >= thr) & (b < thr)).sum()),
                     "keras_only_detected": int(((a < thr) & (b >= thr)).sum())})
    agree = pd.DataFrame(rows)
    both = pd.concat([t.assign(model="pytorch"), tk.assign(model="keras")]).merge(agree, on="split")
    both.to_csv(os.path.join(paths.MODELS, "torch_vs_keras.csv"), index=False)
    print("\nKeras ensemble (original seeds, same windows)")
    print(tk.set_index("split").T.to_string())
    print("\nAgreement PyTorch vs Keras")
    print(agree.set_index("split").T.to_string())


if __name__ == "__main__":
    main()
