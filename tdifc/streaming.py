"""
Real-time (causal) version of the fusion pipeline's per-file preprocessing
(scan_preprocessing.preprocess_pulsed / preprocess_regular + build_windows).

The recording arrives in short chunks (10 s from DiDAQ). StreamProcessor keeps a rolling
buffer of the raw Ch1 / Ch2 signal and, each time process() is called, finalizes the
candidates it can no longer change:

  pulsed laser (~10 s ON / ~40 s OFF)
      Laser-ON segments are found by Otsu thresholding of Ch1 (threshold from the last
      `mode_window_s` of raw Ch1, so it follows slow drifts; offline: one threshold per
      recording file), with the same 5-sample ON/OFF overshoots, the same zeroing of
      pulses shorter than the 2 s background window, and the same per-segment background
      removal and adaptive-threshold peak detection (peak_detection.PreProcessing).
      A segment is processed only once it is complete (laser switched OFF again), so
      every ON segment is processed whole, exactly as offline; candidates therefore
      appear at the end of their laser-ON segment (<= ~10 s after the peak). OFF gaps
      stay zero.
  continuous laser
      The signal from `context_s` before the last finalized sample to the newest sample
      is preprocessed as one segment; candidates up to `margin_s` before the newest
      sample are finalized (the background median and the 1 s window around a peak are
      then complete).
  mode
      'auto' decides pulsed / continuous from the last `mode_window_s` of Ch1 with the
      pipeline's is_pulsed test plus two checks a continuous signal fails: the laser-ON runs
      are long (>= 2 s; a continuous signal chattering around its Otsu split has short ones),
      and the OFF level is dark, far below the ON level (OFF/ON medians < MAX_OFF_ON_RATIO; a
      slowly decaying continuous signal split by Otsu has two bright levels, ratio ~0.4-1).
      The decision is fixed for the rest of the recording once it has been the same for
      `mode_window_s` (continuous only while the laser light is on, Ch1 median > DARK_MV, so a
      recording that starts dark is not fixed as continuous).
      'pulsed' / 'continuous' force the mode (as the pipeline's pulsed=True / False).

Windows are built exactly as offline: 2,001 samples of the background-subtracted signal
(OFF gaps = 0) centred on the candidate, invalid mask, joint standardization.
"""
import logging

import numpy as np
import pandas as pd

from . import parameters as var
from . import peak_detection as pp
from . import scan_preprocessing as sc

FS = var.fs
BGW = var.background_sliding_window
HALF = sc.WIN_LEN // 2
log = logging.getLogger("stream")


def pulse_regions(df, cutoff):
    """peak_detection.pulse_start_end_index with guards for a stream segment (no ON or
    no OFF edge in it). Zeroes pulses shorter than the background window (+/- 1 s) in df
    (in place, as offline) and returns (starts, stops, ends_on, up_edges): the ON regions
    with the 5-sample overshoots; ends_on = the last region is still ON at the end of df."""
    fs, n = FS, len(df)
    on = (df["Ch1"].to_numpy() > cutoff).astype(np.int8)
    up = np.where(np.diff(on) == 1)[0]
    down = np.where(np.diff(on) == -1)[0]
    for s in up:                                   # short pulses with both edges -> OFF
        e = down[down > s]
        if len(e) and e[0] - s < BGW:
            df.iloc[max(0, s - fs):min(n, e[0] + fs)] = 0
    on = (df["Ch1"].to_numpy() > cutoff).astype(np.int8)
    up = np.where(np.diff(on) == 1)[0]
    down = np.where(np.diff(on) == -1)[0]
    starts = up + sc.ON_OVERSHOOT
    stops = down - sc.OFF_OVERSHOOT
    if n and on[0]:                                # segment starts ON (as offline: region from 0)
        starts = np.insert(starts, 0, 0)
    if n and on[-1]:                               # still ON at the end (as offline: region to len)
        stops = np.append(stops, n)
    return starts, stops, bool(n and on[-1]), up


MAX_OFF_ON_RATIO = 0.25      # pulsed: OFF median / ON median (real pulsed data ~0.003, continuous >= 0.39)
DARK_MV = 100                 # Ch1 median below this: no laser light (dark level ~3-5 mV)


def off_on_ratio(ch1, cutoff):
    """Median of the OFF samples (ch1 <= cutoff) / median of the ON samples; 1.0 if one is missing."""
    on = ch1 > cutoff
    if on.all() or not on.any():
        return 1.0
    hi = float(np.median(ch1[on]))
    return float(np.median(ch1[~on])) / hi if hi > 0 else 1.0


def looks_pulsed(ch1, cutoff=None):
    """The laser is pulsed in this stretch of Ch1: the pipeline's is_pulsed test, long ON runs and a
    dark OFF level."""
    cutoff = float(pp.OtsuThreshold(ch1, bins_num=sc.OTSU_BINS)) if cutoff is None else cutoff
    pulsed, _, _ = sc.is_pulsed(ch1)
    return bool(pulsed and long_on_fraction(ch1, cutoff) >= 0.5 and off_on_ratio(ch1, cutoff) < MAX_OFF_ON_RATIO)


def long_on_fraction(ch1, cutoff):
    """Fraction of ON samples (ch1 > cutoff) that lie in ON runs >= the 2 s background window."""
    on = ch1 > cutoff
    if not on.any():
        return 0.0
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    lens = np.where(d == -1)[0] - np.where(d == 1)[0]
    return float(lens[lens >= BGW].sum() / on.sum())


class StreamProcessor:
    def __init__(self, mode="auto", mode_window_s=300, min_start_s=60, context_s=30, margin_s=3,
                 max_on_s=120, keep_s=600):
        assert mode in ("auto", "pulsed", "continuous")
        self.mode_req = mode
        self.mode_win = int(mode_window_s * FS)
        self.min_start = int(min_start_s * FS)
        self.context = int(context_s * FS)
        self.margin = int(margin_s * FS)
        self.max_on = int(max_on_s * FS)
        self.keep = int(keep_s * FS)
        self.raw1 = np.empty(0)
        self.raw2 = np.empty(0)
        self.sig1 = np.empty(0)          # background-subtracted signal (OFF gaps = 0)
        self.sig2 = np.empty(0)
        self.offset = 0                  # absolute sample index of buffer[0]
        self.n = 0                       # samples received
        self.commit = 0                  # candidates before this sample are final
        self.on_samples = 0              # laser-ON samples finalized (rate denominator)
        self.mode = None
        self.otsu = None
        self.mode_fixed = False          # auto: the decision no longer changes
        self._verdict, self._verdict_since = None, 0

    # ------------------------------------------------------------- buffer ---
    def append(self, ch1, ch2):
        ch1 = np.asarray(ch1, np.float64).ravel()
        ch2 = np.asarray(ch2, np.float64).ravel()
        assert len(ch1) == len(ch2)
        self.raw1 = np.concatenate([self.raw1, ch1])
        self.raw2 = np.concatenate([self.raw2, ch2])
        self.sig1 = np.concatenate([self.sig1, np.zeros(len(ch1))])
        self.sig2 = np.concatenate([self.sig2, np.zeros(len(ch1))])
        self.n += len(ch1)

    def seed(self, raw1, raw2, start, commit, on_samples):
        """Continue after the saved data was analysed offline (app started during an experiment).
        raw1 / raw2: the raw signal from absolute sample `start` to the end of the saved data
        (a few minutes: context for the background, the Otsu threshold and the mode decision);
        candidates before `commit` (a laser-OFF point) were already reported; on_samples: laser-ON
        samples analysed so far."""
        self.raw1 = np.asarray(raw1, np.float64).copy()
        self.raw2 = np.asarray(raw2, np.float64).copy()
        self.sig1 = np.zeros(len(self.raw1))
        self.sig2 = np.zeros(len(self.raw1))
        self.offset = int(start)
        self.n = int(start) + len(self.raw1)
        self.commit = int(commit)
        self.on_samples = int(on_samples)

    def _rel(self, a):
        return a - self.offset

    def _trim(self):
        keep_from = min(self.commit - self.context - 2 * HALF, self.n - max(self.keep, self.mode_win))
        cut = self._rel(keep_from)
        if cut > 60 * FS:
            for k in ("raw1", "raw2", "sig1", "sig2"):
                setattr(self, k, getattr(self, k)[cut:])
            self.offset += cut

    # --------------------------------------------------------------- mode ---
    def _decide_mode(self):
        recent = self.raw1[max(0, self._rel(self.n - self.mode_win)):]
        self.otsu = float(pp.OtsuThreshold(recent, bins_num=sc.OTSU_BINS))
        if self.mode_req != "auto":
            return self.mode_req
        if self.mode_fixed:
            return self.mode
        verdict = "pulsed" if looks_pulsed(recent, self.otsu) else "continuous"
        if verdict != self._verdict:
            self._verdict, self._verdict_since = verdict, self.n
        if self.n - self._verdict_since >= self.mode_win and (
                verdict == "pulsed" or float(np.median(recent)) > DARK_MV):
            self.mode_fixed = True
            log.info("laser mode fixed for the rest of the recording: %s (the same for %d min)",
                     verdict, self.mode_win // FS // 60)
        return verdict

    # ------------------------------------------------------------ process ---
    def process(self, final=False):
        """Finalize what can be finalized; returns dict(peaks (absolute sample indices), raw, X, M,
        edge, mode, otsu, start, stop (the newly finalized range), on_samples)."""
        empty = dict(peaks=np.empty(0, int), raw=np.empty((0, 2, sc.WIN_LEN), np.float32),
                     X=np.empty((0, 2, sc.WIN_LEN), np.float32), M=np.empty((0, sc.WIN_LEN), bool),
                     edge=np.empty(0, bool), mode=self.mode, otsu=self.otsu, start=self.commit,
                     stop=self.commit, on_samples=self.on_samples)
        if self.n < self.min_start and not final:
            return empty
        mode = self._decide_mode()
        if mode != self.mode:
            log.info("laser mode: %s (Otsu threshold %.1f)", mode, self.otsu)
            self.mode = mode
        start = self.commit
        peaks = self._pulsed(final) if mode == "pulsed" else self._continuous(final)
        if peaks is None:
            return {**empty, "mode": mode, "otsu": self.otsu}
        raw, X, M = sc.build_windows(self.sig1, self.sig2, np.asarray(peaks, int) - self.offset) \
            if len(peaks) else (empty["raw"], empty["X"], empty["M"])
        edge = sc.edge_distance_ms(M) <= sc.EDGE_MS if len(peaks) else empty["edge"]
        out = dict(peaks=np.asarray(peaks, int), raw=raw, X=X, M=M, edge=edge, mode=mode, otsu=self.otsu,
                   start=start, stop=self.commit, on_samples=self.on_samples)
        self._trim()
        return out

    def _segment(self, a, b):
        ra, rb = self._rel(a), self._rel(b)
        return pd.DataFrame({"Ch1": self.raw1[ra:rb].copy(), "Ch2": self.raw2[ra:rb].copy()})

    def _preprocess(self, df):
        return pp.PreProcessing(df, FS, BGW, var.rel_thresh, var.min_peak, var.max_peak, var.prominence_factor)

    def _pulsed(self, final):
        a, b = self.commit, self.n
        if b - a < BGW and not final:
            return None
        df = self._segment(a, b)
        starts, stops, ends_on, up = pulse_regions(df, self.otsu)
        n = len(df)
        regions = list(zip(starts, stops))
        incomplete, forced_commit = None, None
        if ends_on and not final and regions:
            incomplete = regions.pop()
            if n - incomplete[0] > self.max_on:        # laser ON far too long for pulsing:
                regions.append((incomplete[0], n - self.margin))   # finalize up to the margin
                incomplete, forced_commit = None, b - self.margin
        peaks = []
        last_end = None
        for s, e in regions:
            s, e = int(s), int(e)
            if e - s < BGW:                            # shorter than the background window -> OFF
                continue
            bs1, bs2, pk, _ = self._preprocess(df[s:e])
            self.sig1[self._rel(a + s):self._rel(a + e)] = np.asarray(bs1)
            self.sig2[self._rel(a + s):self._rel(a + e)] = np.asarray(bs2)
            peaks.extend(int(p) + a + s for p in np.asarray(pk, int))
            self.on_samples += e - s
            last_end = e
        if final:
            new_commit = b
        elif forced_commit is not None:
            new_commit = forced_commit
        elif incomplete is not None:
            # stop just before the ongoing ON region (an OFF sample); it is processed whole later
            first_on = [u for u in up if u + sc.ON_OVERSHOOT == incomplete[0]]
            new_commit = a + (first_on[0] if first_on else int(incomplete[0]))
        else:
            # past the last processed region (first OFF sample), or up to 2 s before the newest
            # sample when only OFF signal followed (a pulse starting there is caught next time)
            done = a + last_end + sc.OFF_OVERSHOOT + 1 if last_end is not None else a
            new_commit = max(done, b - 2 * FS)
        self.commit = max(self.commit, int(new_commit))
        return sorted(peaks)

    def _continuous(self, final):
        if not final and self.n - self.commit < self.margin + FS:
            return None
        a0 = max(self.offset, self.commit - self.context)
        b = self.n
        if b - a0 < BGW:
            return None
        df = self._segment(a0, b)
        bs1, bs2, pk, _ = self._preprocess(df)
        new_commit = b if final else b - self.margin
        lo, hi = self.commit - a0, new_commit - a0
        pk = np.asarray(pk, int)
        peaks = sorted(int(p) + a0 for p in pk[(pk >= lo) & (pk < hi)])
        # the windows of these peaks (+/- 1000 samples) are taken from THIS segment's signal
        left, right = max(0, lo - HALF), min(b, new_commit + HALF + 1) - a0
        self.sig1[self._rel(a0 + left):self._rel(a0 + right)] = np.asarray(bs1)[left:right]
        self.sig2[self._rel(a0 + left):self._rel(a0 + right)] = np.asarray(bs2)[left:right]
        self.on_samples += new_commit - self.commit
        self.commit = new_commit
        return peaks
