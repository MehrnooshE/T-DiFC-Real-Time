"""
Check the real-time preprocessing against the offline (batch) preprocessing on a recording.

Replays a t-DiFC recording file (.mat with `data` = N x 2) through tdifc.streaming in
10 s chunks, exactly as the app receives it, and compares the candidates with those of the
fusion pipeline's per-file preprocessing (tdifc.scan_preprocessing: one Otsu threshold and
one pulsed / continuous decision for the whole file):
  - candidates found by both (same sample +/- 2), only offline, only real time
  - largest difference of the standardized model windows of the common candidates
  - with --score: 5-seed probabilities of both and the detected peaks (p >= cut-off)

Usage:  python check_streaming.py "<recording.mat>" [--minutes 120] [--start_min 0] [--score]
        [--mode auto|pulsed|continuous]
"""
import argparse
import time

import numpy as np

from tdifc import paths, parameters as var, peak_detection as pp, scan_preprocessing as sc
from tdifc.streaming import StreamProcessor

FS = var.fs


def main():
    ap = argparse.ArgumentParser(description="real-time vs offline preprocessing on one recording")
    ap.add_argument("mat")
    ap.add_argument("--minutes", type=float, default=None, help="only this many minutes (default: whole file)")
    ap.add_argument("--start_min", type=float, default=0.0)
    ap.add_argument("--chunk_s", type=float, default=10.0)
    ap.add_argument("--mode", default="auto", choices=["auto", "pulsed", "continuous"])
    ap.add_argument("--score", action="store_true", help="also score both with the PyTorch ensemble")
    ap.add_argument("--tol", type=int, default=2, help="samples within which two candidates are the same")
    args = ap.parse_args()

    data = pp.load_scan_data(args.mat)
    a = int(args.start_min * 60 * FS)
    b = len(data) if args.minutes is None else min(len(data), a + int(args.minutes * 60 * FS))
    data = data.iloc[a:b].reset_index(drop=True)
    print(f"{args.mat}: {len(data) / FS / 60:.1f} min")

    # offline: whole file at once
    t0 = time.time()
    pulsed = {"auto": None, "pulsed": True, "continuous": False}[args.mode]
    detected, on_frac, n_on = sc.is_pulsed(data["Ch1"].to_numpy())
    pulsed = detected if pulsed is None else pulsed
    s1, s2, pk_b, info = (sc.preprocess_pulsed if pulsed else sc.preprocess_regular)(data.copy())
    raw_b, X_b, M_b = sc.build_windows(s1, s2, pk_b)
    print(f"offline : pulsed={pulsed} (ON fraction {on_frac:.2f}, {n_on} ON transitions), "
          f"{len(pk_b)} candidates, laser-ON {info['on_hours'] * 60:.1f} min ({time.time() - t0:.0f} s)")

    # real time: 10 s chunks
    t0 = time.time()
    st = StreamProcessor(mode=args.mode)
    ch1, ch2 = data["Ch1"].to_numpy(), data["Ch2"].to_numpy()
    step = int(args.chunk_s * FS)
    outs, lag = [], []
    for i in range(0, len(ch1), step):
        st.append(ch1[i:i + step], ch2[i:i + step])
        o = st.process()
        outs.append(o)
        lag += [(st.n - p) / FS for p in o["peaks"]]
    outs.append(st.process(final=True))
    pk_s = np.concatenate([o["peaks"] for o in outs])
    X_s = np.concatenate([o["X"] for o in outs])
    M_s = np.concatenate([o["M"] for o in outs])
    assert np.all(np.diff(pk_s) > 0), "real-time candidates not strictly increasing"
    print(f"realtime: mode {st.mode}, {len(pk_s)} candidates, laser-ON {st.on_samples / FS / 60:.1f} min, "
          f"delay from peak to its report: median {np.median(lag) if lag else np.nan:.1f} s, "
          f"max {np.max(lag) if lag else np.nan:.1f} s ({time.time() - t0:.0f} s)")

    # match
    j = np.searchsorted(pk_s, pk_b)
    best = np.full(len(pk_b), -1)
    for k, p in enumerate(pk_b):
        c = [i for i in (j[k] - 1, j[k]) if 0 <= i < len(pk_s) and abs(pk_s[i] - p) <= args.tol]
        if c:
            best[k] = min(c, key=lambda i: abs(pk_s[i] - p))
    both = best >= 0
    only_s = np.setdiff1d(np.arange(len(pk_s)), best[both])
    print(f"common {both.sum()}, offline only {(~both).sum()}, real-time only {len(only_s)}")
    if both.any():
        d = np.abs(X_b[both] - X_s[best[both]]).max(axis=(1, 2))
        print(f"standardized windows of common candidates: identical (<1e-4) {np.mean(d < 1e-4):.1%}, "
              f"max |diff| median {np.median(d):.2g}, max {d.max():.2g}")
    for name, idx in (("offline only", pk_b[~both][:10]), ("real-time only", pk_s[only_s][:10])):
        if len(idx):
            print(f"  {name} (first 10, min): {np.round(idx / FS / 60, 2).tolist()}")

    if args.score:
        from tdifc.model import FusionEnsemble
        ens = FusionEnsemble(paths.MODEL_PATH)
        p_b, p_s = ens.predict(X_b, M_b), ens.predict(X_s, M_s)
        cut = ens.cut_off
        print(f"detected (p >= {cut}): offline {(p_b >= cut).sum()}, real time {(p_s >= cut).sum()}")
        if both.any():
            dp = np.abs(p_b[both] - p_s[best[both]])
            print(f"common candidates: |p offline - p real time| median {np.median(dp):.2g}, max {dp.max():.2g}, "
                  f"same call {np.mean((p_b[both] >= cut) == (p_s[best[both]] >= cut)):.1%}")
        print(f"detected among offline-only {int((p_b[~both] >= cut).sum())}, "
              f"among real-time-only {int((p_s[only_s] >= cut).sum())}")


if __name__ == "__main__":
    main()
