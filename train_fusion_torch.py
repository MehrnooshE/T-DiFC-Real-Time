"""
Train the time + spectral `fusion_lr1e-4_wd_ls` model in PyTorch (5 seeds).

PyTorch port of the fusion pipeline's scripts/05_train.py (Keras / TensorFlow 2.13):
same architecture (tdifc/model.py), same inputs, same train/val/test splits
(data/train-val-test/, copied from the pipeline), same recipe:
  Adam lr 1e-4 (epsilon 1e-7, as Keras), decoupled weight decay 1e-4 on the conv / dense
  kernels after every step (kernel *= 1 - lr * wd), label smoothing 0.05, batch 64,
  shuffled every epoch, <= 60 epochs;
  ReduceLROnPlateau (x0.5, patience 4, min_delta 1e-4, min lr 1e-6) and early stopping
  (patience 8, best weights restored), both on the PLAIN validation BCE (val_bce),
  with Keras' counting rules;
  seeds 0-4; spectral input standardized with TRAIN statistics (cos-phase channel as is).

Writes models/fusion_lr1e-4_wd_ls_torch.pt   the 5 seeds + spectral standardizer (one file;
                                              what tdifc_app.py loads)
       models/training_history.csv           per-epoch train / val BCE, lr, restored epoch flagged
       models/predictions_{val,test}.csv     meta + per-seed and 5-seed mean probability
Then run evaluate_fusion_torch.py for the metrics.

Usage:  python train_fusion_torch.py [--device auto|cpu|mps] [--threads N] [--seeds 0 1 2 3 4]
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tdifc import paths, parameters, spectral as sp, model as fm

SEEDS = [0, 1, 2, 3, 4]
CFG = dict(lr=1e-4, wd=1e-4, ls=0.05, batch=64, max_epochs=60, adam_eps=1e-7,
           es_patience=8, rlr_factor=0.5, rlr_patience=4, rlr_min_delta=1e-4, min_lr=1e-6)
COL = paths.MODEL_NAME


def to_dev(a, device):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)


@torch.no_grad()
def evaluate(net, xt, xs, y, batch=512):
    """Plain BCE (no label smoothing) and probabilities, inference mode (dropout off, BN moving stats)."""
    net.eval()
    lt = torch.cat([net(xt[a:a + batch], xs[a:a + batch]) for a in range(0, len(xt), batch)])
    return F.binary_cross_entropy_with_logits(lt, y).item(), torch.sigmoid(lt).cpu().numpy()


def train_seed(seed, inp, y, n_bins, device, cfg=CFG):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = fm.FusionNet(n_bins).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])
    kernels = net.kernels()
    xt, xs, yt = inp["train"][0], inp["train"][1], y["train"]
    n = len(yt)

    best, best_epoch, best_state, wait_es = np.inf, -1, None, 0
    rl_best, wait_rl = np.inf, 0
    rows = []
    for epoch in range(cfg["max_epochs"]):
        net.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for a in range(0, n, cfg["batch"]):
            idx = perm[a:a + cfg["batch"]]
            yb = yt[idx]
            lt = net(xt[idx], xs[idx])
            loss = F.binary_cross_entropy_with_logits(lt, yb * (1 - cfg["ls"]) + 0.5 * cfg["ls"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                lr = opt.param_groups[0]["lr"]
                for k in kernels:                       # decoupled weight decay (AdamW-style)
                    k.mul_(1.0 - lr * cfg["wd"])
                tot += F.binary_cross_entropy_with_logits(lt, yb, reduction="sum").item()
        val_bce, _ = evaluate(net, inp["val"][0], inp["val"][1], y["val"])
        lr = opt.param_groups[0]["lr"]
        rows.append({"seed": seed, "epoch": epoch + 1, "train_loss": tot / n, "val_bce": val_bce, "lr": lr})

        # EarlyStopping(monitor="val_bce", patience=8, restore_best_weights=True)
        if val_bce < best:
            best, best_epoch, wait_es = val_bce, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait_es += 1
        stop = wait_es >= cfg["es_patience"] and epoch > 0
        # ReduceLROnPlateau(monitor="val_bce", factor=0.5, patience=4, min_lr=1e-6), Keras rules
        if val_bce < rl_best - cfg["rlr_min_delta"]:
            rl_best, wait_rl = val_bce, 0
        else:
            wait_rl += 1
            if wait_rl >= cfg["rlr_patience"]:
                if lr > cfg["min_lr"]:
                    for g in opt.param_groups:
                        g["lr"] = max(lr * cfg["rlr_factor"], cfg["min_lr"])
                wait_rl = 0
        if stop:
            break
    net.load_state_dict(best_state)
    hist = pd.DataFrame(rows)
    hist["best"] = hist.epoch == best_epoch + 1
    return net.eval(), hist, best_epoch + 1, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--device", default="auto", help="auto (default: Apple-GPU mps if available, else cpu), cpu or mps")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(paths.MODELS, exist_ok=True)

    data = {s: paths.load_split(s) for s in paths.SPLITS}
    spec = {s: sp.centre_spectrum(sp.als_scaled(data[s][0], s, paths.SPLITS_ALS_DIR), data[s][2]) for s in paths.SPLITS}
    zc = sp.Standardizer(spec["train"], skip=(2,))
    inp = {s: (to_dev(data[s][0], device), to_dev(zc(spec[s]), device)) for s in paths.SPLITS}
    y = {s: to_dev(data[s][1], device) for s in paths.SPLITS}
    n_bins = spec["train"].shape[1]
    print(f"device {device}; train {len(data['train'][1])}, val {len(data['val'][1])}, test {len(data['test'][1])} "
          f"windows; spectral input {spec['train'].shape[1:]}", flush=True)

    nets, hists, best_epochs, P = [], [], [], {"val": [], "test": []}
    for seed in args.seeds:
        t0 = time.time()
        net, hist, best_epoch, best = train_seed(seed, inp, y, n_bins, device)
        nets.append(net.cpu())
        hists.append(hist)
        best_epochs.append(int(best_epoch))
        net.to(device)
        for s in P:
            P[s].append(evaluate(net, inp[s][0], inp[s][1], y[s])[1])
        net.cpu()
        print(f"seed {seed}: epochs {len(hist)}, best epoch {best_epoch}, min val BCE {best:.4f}, "
              f"val AUC {roc_auc_score(data['val'][1], P['val'][-1]):.4f} ({time.time() - t0:.0f} s)", flush=True)

    fm.save_ensemble(paths.MODEL_PATH, nets, zc, extra={
        "model_name": COL, "cut_off": float(parameters.cut_off), "seeds": [int(s) for s in args.seeds],
        "best_epochs": best_epochs, "train_cfg": {k: float(v) for k, v in CFG.items()},
        "torch_version": str(torch.__version__), "trained_on": str(device)})
    print(f"saved {paths.MODEL_PATH}")
    h = pd.concat(hists, ignore_index=True)
    h.insert(0, "config", "lr1e-4_wd_ls")
    h.to_csv(os.path.join(paths.MODELS, "training_history.csv"), index=False)

    for s in P:
        d = data[s][3].reset_index(drop=True).copy()
        for k, p in zip(args.seeds, P[s]):
            d[f"{COL}_seed{k}"] = p
        d[COL] = np.mean(P[s], 0)
        d.to_csv(os.path.join(paths.MODELS, f"predictions_{s}.csv"), index=False)
        print(f"{s}: 5-seed mean AUC {roc_auc_score(data[s][1], d[COL]):.4f}")


if __name__ == "__main__":
    main()
