"""
Paths of the t-DiFC-Real-Time folder and loading of the labeled train/val/test splits.

The splits are a copy of the fusion pipeline's
data/train-val-test/Joint-Standardization-Augmented/:
  {split}_X.npz   key X, (N, 2, 2001) float32, jointly standardized windows,
                  candidate peak at index 1000, fs = 2000 Hz
  {split}_y.npy   (N,) 1 = peak (CTC), 0 = artifact
  {split}_mask.npy (N, 2001) bool, True = zero-padding / pulsed off-gap
  {split}_meta.csv, scan_assignment.csv, split_report.txt
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
SPLITS_DIR = os.path.join(DATA, "train-val-test", "Joint-Standardization-Augmented")
SPLITS_ALS_DIR = os.path.join(DATA, "train-val-test-als-cache")   # ALS + Ch2-scaled split windows
KERAS_REF_DIR = os.path.join(MODELS, "keras_reference")           # results of the original Keras seeds

MODEL_NAME = "fusion_lr1e-4_wd_ls_torch"
MODEL_PATH = os.path.join(MODELS, f"{MODEL_NAME}.pt")              # the 5-seed ensemble + spectral standardizer
SPLITS = ["train", "val", "test"]


def load_split(split):
    for f in (f"{split}_X.npz", f"{split}_y.npy", f"{split}_mask.npy", f"{split}_meta.csv"):
        if not os.path.exists(os.path.join(SPLITS_DIR, f)):
            raise SystemExit(f"missing {os.path.join(SPLITS_DIR, f)}: copy the splits from the fusion pipeline's "
                             f"data/train-val-test/Joint-Standardization-Augmented/")
    X = np.load(os.path.join(SPLITS_DIR, f"{split}_X.npz"))["X"]
    y = np.load(os.path.join(SPLITS_DIR, f"{split}_y.npy"))
    mask = np.load(os.path.join(SPLITS_DIR, f"{split}_mask.npy"))
    meta = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_meta.csv"))
    assert (meta["label"].values == y).all()
    scans = pd.read_csv(os.path.join(SPLITS_DIR, "scan_assignment.csv"))
    cl = scans.drop_duplicates(["scan", "category"]).set_index(["scan", "category"])["cell_line"]
    meta["cell_line"] = [cl.get((s, c), "unknown") for s, c in zip(meta.scan, meta.category)]
    return X, y, mask, meta
