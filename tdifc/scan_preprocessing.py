"""
Per-file preprocessing and windowing of a recording, copied from the fusion pipeline's
scripts/full_scan_preprocessing.py (same functions, same constants; the caching, zip and
cohort helpers are left out).

  is_pulsed, preprocess_pulsed, preprocess_regular   Otsu laser-ON segmentation, 2 s rolling-
                                                     median background removal, adaptive 4.5 sigma
                                                     peak detection (Methods S1.1-S1.2)
  build_windows                                      2,001-sample windows, invalid mask, joint
                                                     standardization (S1.3)
  edge_distance_ms                                   distance of a candidate to a laser edge

streaming.py runs the same functions on the live data stream; check_streaming.py
compares the two on a recorded scan.
"""
import numpy as np

from . import peak_detection as pp
from . import parameters as var

WIN_LEN = 2001
ON_OVERSHOOT, OFF_OVERSHOOT, OTSU_BINS = 5, 5, 20   # as in 02_preprocess_labeled_scans.py
EDGE_MS = 25          # candidates this close to a laser ON/OFF edge are flagged


# ---------------------------------------------------------- preprocessing ---
def preprocess_pulsed(data):
    """Same steps as 02_preprocess_labeled_scans.preprocess_pulsed_file."""
    fs = var.fs
    otsu = pp.OtsuThreshold(data["Ch1"], bins_num=OTSU_BINS)
    starts, stops = pp.pulse_start_end_index(data, otsu, fs, var.background_sliding_window,
                                             ON_OVERSHOOT, OFF_OVERSHOOT)
    sig1 = np.zeros(len(data))
    sig2 = np.zeros(len(data))
    peaks = []
    prev_stop = 0
    lens = []
    n_short = 0
    for s_idx, e_idx in zip(starts, stops):
        assert s_idx >= prev_stop, "overlapping ON regions"
        if e_idx - s_idx < var.background_sliding_window:
            # region shorter than the background window -> treat as laser-OFF
            n_short += 1
            continue
        bs1, bs2, pk, _ = pp.PreProcessing(data[s_idx:e_idx], fs, var.background_sliding_window,
                                           var.rel_thresh, var.min_peak, var.max_peak,
                                           var.prominence_factor)
        sig1[s_idx:e_idx] = np.asarray(bs1)
        sig2[s_idx:e_idx] = np.asarray(bs2)
        peaks.extend(np.asarray(pk, dtype=int) + s_idx)
        prev_stop = e_idx
        lens.append(e_idx - s_idx)
    if not (np.isfinite(sig1).all() and np.isfinite(sig2).all()):
        raise RuntimeError("non-finite values after preprocessing")
    info = {"otsu_threshold": float(otsu), "n_on_regions": len(lens), "n_short_on_dropped": n_short,
            "on_hours": float(np.sum(lens) / fs / 3600)}
    return sig1, sig2, np.array(sorted(peaks), dtype=int), info


def preprocess_regular(data):
    bs1, bs2, pk, _ = pp.PreProcessing(data, var.fs, var.background_sliding_window, var.rel_thresh,
                                       var.min_peak, var.max_peak, var.prominence_factor)
    return np.asarray(bs1), np.asarray(bs2), np.array(sorted(pk), dtype=int), {
        "on_hours": len(data) / var.fs / 3600}


def is_pulsed(ch1):
    """Laser pulsing gives a bimodal ch1 with ~20% ON time; require many
    ON transitions at the Otsu split to call it pulsed, and ON and OFF samples
    mixed within most 1-min blocks (a decaying continuous scan is bimodal too)."""
    thr = pp.OtsuThreshold(ch1, bins_num=OTSU_BINS)
    on = ch1 > thr
    n_on = int(np.sum(np.diff(on.astype(np.int8)) == 1))
    hours = len(ch1) / var.fs / 3600
    blk = 60 * var.fs
    fr = on[:len(on) // blk * blk].reshape(-1, blk).mean(1)
    mixed = np.mean((fr > 0.02) & (fr < 0.98)) if len(fr) else 1.0
    return (0.05 < on.mean() < 0.6) and n_on > 20 * hours and mixed >= 0.5, on.mean(), n_on


# -------------------- windowing (verbatim logic from functions.py) ----------
def compute_invalid_mask(win, zero_tol=1e-9):
    return (np.abs(win[0]) < zero_tol) & (np.abs(win[1]) < zero_tol)


def standardize_joint(win, invalid_mask=None):
    if invalid_mask is not None and invalid_mask.sum() < win.shape[-1]:
        valid = ~invalid_mask
        flat = np.concatenate([win[0, valid], win[1, valid]])
    else:
        flat = win.reshape(-1)
    mean = flat.mean()
    std = flat.std()
    if std < 1e-12:
        std = 1e-12
    return (win - mean) / std


def extract_window(ch1, ch2, peak_idx, window_len):
    half_left = window_len // 2
    half_right = window_len - half_left
    T = len(ch1)
    start, end = peak_idx - half_left, peak_idx + half_right
    pad_left = max(0, -start)
    a, b = max(0, start), min(T, end)
    win = np.zeros((2, window_len), dtype=np.float64)
    if a < b:
        win[0, pad_left:pad_left + b - a] = ch1[a:b]
        win[1, pad_left:pad_left + b - a] = ch2[a:b]
    return win


def edge_distance_ms(M):
    """Distance from the candidate (window centre) to the nearest invalid
    (OFF-gap / padding) sample on either side, in ms; inf if none in window."""
    c = WIN_LEN // 2
    out = np.full(len(M), np.inf)
    for i, m in enumerate(M):
        left = np.where(m[:c])[0]
        right = np.where(m[c + 1:])[0]
        d = min(c - left[-1] if len(left) else np.inf, right[0] + 1 if len(right) else np.inf)
        out[i] = d / var.fs * 1e3
    return out


def build_windows(sig1, sig2, peaks):
    """raw (bg-subtracted, mV), X (jointly standardized) and M (invalid mask) windows."""
    raw = np.empty((len(peaks), 2, WIN_LEN), dtype=np.float32)
    X = np.empty_like(raw)
    M = np.empty((len(peaks), WIN_LEN), dtype=bool)
    for i, p in enumerate(peaks):
        w = extract_window(sig1, sig2, int(p), WIN_LEN)
        m = compute_invalid_mask(w)
        raw[i], M[i], X[i] = w, m, standardize_joint(w, invalid_mask=m)
    return raw, X, M
