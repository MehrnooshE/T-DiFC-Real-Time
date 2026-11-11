"""
Spectral input of the time + spectral `fusion` model, copied from the fusion pipeline's
scripts/fusion_model.py (the feature half; the Keras architecture is replaced by model.py).

  ALS baseline removal (lam=1e7, p=0.01) and Ch2 scaled to Ch1's background std
  (signal_functions.ALS_BGremoved_Ch2scaled) on the jointly standardized window, then
  the centre 401 samples (+/-100 ms), cosine-tapered at edges and off-gaps:
  log10|Ch1|, log10|Ch2|, cos(phase Ch1 - phase Ch2), 0-300 Hz (61 bins of ~5 Hz).
Only bins <= 300 Hz are used: the SR560 pre-amps low-pass at 300 Hz.
"""
import os
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import parameters as var
from . import signal_functions as labf

FS = var.fs
F_MAX = 300.0
CENTER_HW = 200
RAMP = 50                        # taper length, as compute_fft_masked's default
LOG_EPS = 1e-5


def als_scaled(X, name=None, cache=None, progress=True):
    """signal_functions.ALS_BGremoved_Ch2scaled (~30 ms/window); cached in cache/<name>_als.npy
    when a cache folder is given (training), computed fresh otherwise (real time)."""
    p = os.path.join(cache, f"{name}_als.npy") if cache and name else None
    if p and os.path.exists(p):
        out = np.load(p)
        if out.shape == X.shape:
            return out
    out, n_fail = labf.ALS_BGremoved_Ch2scaled(X.astype(np.float64), progress=progress)
    out = out.astype(np.float32)
    if p:
        os.makedirs(cache, exist_ok=True)
        print(f"  ALS {name}: {len(X)} windows, ch2 scaling failed on {n_fail}")
        np.save(p, out)
    return out


def taper_weights(valid):
    """Same cosine ramp as functions.compute_fft_masked (edges and off-gaps)."""
    d = distance_transform_edt(np.pad(valid, 1))[1:-1]
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(d / RAMP, 0.0, 1.0))


def centre_spectrum(Xals, M):
    """(N, 2, 2001) ALS windows -> (N, n_bins, 3): log|Ch1|, log|Ch2|, cos dphi."""
    c = Xals.shape[-1] // 2
    L = 2 * CENTER_HW + 1
    freqs = np.fft.rfftfreq(L, 1 / FS)
    keep = freqs <= F_MAX
    out = np.empty((len(Xals), keep.sum(), 3), dtype=np.float32)
    for i in range(len(Xals)):
        seg = Xals[i, :, c - CENTER_HW:c + CENTER_HW + 1].astype(np.float64)
        w = taper_weights(~M[i, c - CENTER_HW:c + CENTER_HW + 1])
        Z = np.fft.rfft(seg * w, axis=-1)[:, keep] / max(w.sum(), 1e-8)
        mag = np.abs(Z)
        cross = Z[0] * np.conj(Z[1])
        out[i, :, 0] = np.log10(mag[0] + LOG_EPS)
        out[i, :, 1] = np.log10(mag[1] + LOG_EPS)
        out[i, :, 2] = np.real(cross) / np.maximum(np.abs(cross), 1e-12)
    return out


class Standardizer:
    """Per-(bin, channel) z-score fitted on TRAIN; cos-phase channel left as is.
    Stored inside the model file (mu, sd) so the app needs no training data."""

    def __init__(self, A=None, skip=(), mu=None, sd=None):
        if A is not None:
            self.mu = A.mean(0, keepdims=True)
            self.sd = A.std(0, keepdims=True) + 1e-6
            for k in skip:
                self.mu[..., k], self.sd[..., k] = 0.0, 1.0
        else:
            self.mu, self.sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)

    def __call__(self, A):
        return ((A - self.mu) / self.sd).astype(np.float32)
