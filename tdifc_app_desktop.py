"""
tdifc_app_desktop.py
--------------------
Desktop version of tdifc_app.py (same analysis and window), made to be packaged as a macOS app
(build_mac.sh -> "dist/t-DiFC Live.app"). Differences to tdifc_app.py:
  - started without --save_dir / --exp_name (double-click on the app), a start-up window asks for
    the experiment folder, the analysis mode, the results folder and the advanced options
    (remembered for the next start);
  - results go to ~/Documents/t-DiFC results/<exp_name>/ by default (not into the code folder);
  - the log is also written to tdifc_app.log in the results folder (an app has no terminal).
With --save_dir and --exp_name it runs exactly as tdifc_app.py from the terminal.

Real-time t-DiFC peak (CTC) detection with the time + spectral `fusion_lr1e-4_wd_ls`
classifier of the t-DiFC clearance-cohort pipeline, trained in PyTorch
(models/fusion_lr1e-4_wd_ls_torch.pt, train_fusion_torch.py). No TensorFlow.

Architecture:
  - Worker thread  : reads the DiDAQ 10 s chunk files in order as they appear, runs the
                     streaming preprocessing (tdifc/streaming.py: the pipeline's laser-ON
                     segmentation, background removal and adaptive-threshold peak detection)
                     and scores every candidate with the 5-seed ensemble; writes every
                     candidate to results/<exp_name>/candidates.csv
  - GUI main thread: drains a queue every 500 ms and updates the plots

Input (as DiDAQ writes it):  <save_dir>/<exp_name>/F<k>/<exp_name>_F<k>_<n>.mat, n = 1, 2, ...
  each with `data` (samples x channels, mV) and `time` (s). Ch1 / Ch2 are the first two
  columns of F1, F2, ... joined in fiber order: both channels in F1, or Ch1 in F1 and Ch2 in F2.
  Or any folder of .mat files (--folder): read in the order of their names (numbers sorted as
  numbers), e.g. exp_1.mat, exp_2.mat ... / exp_1_F1.mat ... / one long file; F1/ and F2/ folders
  with any file names give Ch1 / Ch2.

SETUP (once, in this folder; see README.md):
    /path/to/python3.11 -m venv t-difc-rt-env
    t-difc-rt-env/bin/python -m pip install -r requirements.txt

RUN:
    cd t-DiFC-Real-Time
    source t-difc-rt-env/bin/activate
    python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc MM" --exp_name "250723-BCMM1_MM-CFSE"
    python tdifc_app.py --save_dir "/Volumes/Public/Jane/ScidBg MM" --exp_name "260521-ScBgMM25-MM-tdT_Pulsed_D41"
    python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc MM" --exp_name "250723-BCMM1_MM-CFSE"
    python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc MM" --exp_name "250723-BCMM2_MM-CFSE"
    python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc L1210A Raw data" --exp_name "250809-BCL1210A2_L1210A-CFSE"

  --analysis (what happens to the data already saved when the app starts):
    real-time               (default) everything with the real-time algorithm, chunk by chunk from the first
                            chunk (a file longer than 10 s, e.g. chunks stitched together, in 10-s slices)
    offline-then-real-time  saved data: offline pipeline in 2-h blocks; new data: real time
    offline                 only the saved data, offline pipeline; no new data is read (finished experiment)
  e.g.  python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc MM" --exp_name "250723-BCMM1_MM-CFSE" --analysis offline

  Test without the DAQ: replay a recorded scan into a folder, in a second terminal:
    python simulate_acquisition.py --mat "<recording>.mat" --save_dir sim --exp_name TEST --speed 10
    python tdifc_app.py --save_dir sim --exp_name TEST

NAS:
  smb://129.10.224.25/Public  or  smb://NIEDRE-NAS.neu.edu/Public
  (macOS: Finder -> Go -> Connect to Server; mounted at /Volumes/Public)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, re, sys, csv, json, html, time, bisect, signal, argparse
import multiprocessing
if __name__ == "__main__":
    # In the packaged app, Python starts helper processes (e.g. multiprocessing's resource tracker)
    # by running the app's own program again. freeze_support() lets such a helper do its job and
    # exit here, before anything else is imported - otherwise each one opened the start-up window.
    multiprocessing.freeze_support()
import logging, threading, queue, traceback

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.optimize import curve_fit
import scipy.io as sio
import torch

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# ── This folder ───────────────────────────────────────────────────────────────
from tdifc import paths, parameters as var
from tdifc.model import FusionEnsemble
from tdifc.streaming import StreamProcessor, looks_pulsed
from tdifc import peak_detection as pkd, scan_preprocessing as scp

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w"))
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(name)-8s]  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log_scan   = logging.getLogger("scan")
log_worker = logging.getLogger("worker")
log_ml     = logging.getLogger("ml")
log_app    = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS  - edit these to match your experiment
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE        = var.fs   # 2000 Hz
WINDOW_SECS          = 60       # width of the live signal view
HIGHLIGHT_W          = 0.2      # seconds - width of highlight rectangle on trace
TRACE_Y_MIN          = 0        # default y range of the signal trace (mV); --y_min / --y_max
TRACE_Y_MAX          = 2500
Y_AUTO_SECS          = 30       # if the signal in the first Y_AUTO_SECS goes above TRACE_Y_MAX, the
Y_AUTO_MARGIN        = 200      # trace's y limit becomes that maximum + Y_AUTO_MARGIN (mV)
DISPLAY_BIN          = 1        # 1 = every sample (full resolution); N > 1 = min/max per N samples
DISPLAY_OFF_AS_ZERO  = True     # pulsed laser: laser-OFF gaps drawn as 0, sparsely (~4x less data)
DISPLAY_OFF_STEP_MS  = 20       # one zero point per 20 ms in laser-OFF gaps (keeps pyqtgraph's thinning local)
DISPLAY_EDGE_MS      = 20       # laser-ON stretches are kept this much wider, so the edges show
DISPLAY_HISTORY_SECS = 3600     # rolling signal buffer: last 1 hour shown in plot
TRIM_STEP_SECS       = 600      # the oldest display data is dropped in 10-min steps (one copy per step)
HIGHLIGHT_MAX_SPAN   = 600      # highlights + labels are drawn when the trace shows <= 10 min
MAGNIFIER_SECS       = 2        # width of the movable magnifier window on the trace
MAGNIFIER_STRETCH    = (5, 4)   # width share trace : magnifier while the magnifier is on
SIDEBAR_WIDTH        = 450      # px of the review sidebar (drag the divider to change it)
CLEARANCE_SHARE      = 0.5      # the sidebar widens to this share of the window on the Clearance tab
WINDOW_SIZE          = (1480, 900)   # start size (px); capped to the screen
COLOR_PEAK           = "#dc3c3c"     # CTC / detected peak: barcode, trace highlights, magnifier, review
                                     # (default; changed with the colour swatch beside "Barcode", remembered)
COLOR_ARTIFACT       = "#8c8c8c"     # artifact / rejected candidate: trace highlights, magnifier, review
PULSE_ZOOM_MARGIN    = 2        # s shown on each side of a pulse clicked in a zoomed-out trace
TRACE_PEN_WIDTH      = 1        # px; a 2-px antialiased pen makes every repaint ~40x slower
POLL_SECS            = 1.0      # how often the worker looks for the next chunk file
RT_SLICE_SECS        = 10       # real time: a longer file (e.g. chunks stitched together) is fed in 10-s slices
TIME_GAP_SECS        = 0.05     # a step of more than this in a file's `time` vector is a gap (pause, lost data)
GAP_TIMEOUT_SECS     = 60       # a missing chunk is skipped after this long if later ones exist
TORCH_THREADS        = 2        # keep the GUI responsive
ANALYSIS_MODES       = ("real-time", "offline-then-real-time", "offline")   # --analysis
DEFAULT_ANALYSIS     = "real-time"
OFFLINE_BLOCK_SECS   = 7200     # chunks already saved at start-up: offline pipeline in 2-h blocks (its file length)
OFFLINE_MIN_SECS     = 120      # less saved data than this: straight to real time
SAFE_CUT_SEARCH_SECS = 120      # a block ends in a laser-OFF gap within its last 2 min ...
SAFE_CUT_OFF_SECS    = 2        # ... of at least 2 s (no laser pulse is cut in two)

CLEARANCE_BIN_MIN    = 5        # default bin size of the clearance plot (min); editable in its window
CLEARANCE_SKIP_MIN   = 0        # default: first X min left out of the exponential fit (still plotted)

MODEL_PATH           = paths.MODEL_PATH
APP_NAME             = "t-DiFC Live"
RESULTS_HOME         = os.path.join(os.path.expanduser("~"), "Documents", "t-DiFC results")   # desktop default
PROB_THRESHOLD       = var.cut_off   # replaced by the cut-off stored in the model file

# ─────────────────────────────────────────────────────────────────────────────
# SHARED QUEUE  - worker puts plot-ready dicts, GUI timer reads them
# ─────────────────────────────────────────────────────────────────────────────
PLOT_QUEUE: queue.Queue = queue.Queue()


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY DECIMATION
# ─────────────────────────────────────────────────────────────────────────────
def minmax_decimate(t: np.ndarray, d: np.ndarray, bin_len: int) -> tuple:
    """Two points (min and max, in time order) per bin of bin_len samples and channel, so
    that 1-5 ms peaks remain visible in the trace at any zoom. t (N,), d (N, C)."""
    if bin_len <= 2:                       # min/max of 1-2 samples is the signal itself
        return t.copy(), np.asarray(d, np.float32).copy()
    n = len(t) // bin_len * bin_len
    if n == 0:
        return t.copy(), d.copy()
    C = d.shape[1]
    db = d[:n].reshape(-1, bin_len, C)
    i_min, i_max = db.argmin(1), db.argmax(1)                  # (B, C)
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    rows = np.arange(len(db))[:, None]
    cols = np.arange(C)[None, :]
    out = np.empty((2 * len(db), C))
    out[0::2] = db[rows, first, cols]
    out[1::2] = db[rows, second, cols]
    tb = t[:n].reshape(-1, bin_len)
    t_out = np.empty(2 * len(db))
    t_out[0::2] = tb[:, bin_len // 4]
    t_out[1::2] = tb[:, (3 * bin_len) // 4]
    return np.concatenate([t_out, t[n:]]), np.vstack([out, d[n:]])


def display_chunk(t: np.ndarray, d: np.ndarray, mode, otsu) -> tuple:
    """Points of one chunk for the trace. Pulsed laser (with DISPLAY_OFF_AS_ZERO): every sample
    of the laser-ON stretches (Ch1 > the stream's Otsu threshold, widened by DISPLAY_EDGE_MS), and
    the laser-OFF stretches as zeros every DISPLAY_OFF_STEP_MS for both channels. (Two points per gap
    would be enough to draw it, but pyqtgraph thins a curve by groups of consecutive points drawn at
    the group's first time; a group spanning a whole gap would draw a false slope across it.) Continuous laser, or before the
    mode is known (first minute): every sample. DISPLAY_BIN > 2: min/max per bin instead."""
    if DISPLAY_BIN > 2:
        return minmax_decimate(t, d, DISPLAY_BIN)
    d = np.asarray(d, np.float32)
    if not (DISPLAY_OFF_AS_ZERO and mode == "pulsed" and otsu is not None):
        return t.copy(), d.copy()
    w = int(DISPLAY_EDGE_MS / 1000 * SAMPLING_RATE)
    on = binary_dilation(d[:, 0] > otsu, structure=np.ones(2 * w + 1, bool))
    if on.all():
        return t.copy(), d.copy()
    off = ~on
    step = max(1, int(DISPLAY_OFF_STEP_MS / 1000 * SAMPLING_RATE))
    keep = on | (off & (np.arange(len(t)) % step == 0))
    edges = np.diff(np.concatenate([[0], off.astype(np.int8), [0]]))
    keep[np.where(edges == 1)[0]] = True               # first and last sample of each OFF stretch
    keep[np.where(edges == -1)[0] - 1] = True
    d = d.copy()
    d[off] = 0.0
    return t[keep], d[keep]


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK FILES  (DiDAQ: <save_dir>/<exp_name>/F<k>/<exp_name>_F<k>_<n>.mat)
# ─────────────────────────────────────────────────────────────────────────────
class ChunkReadError(RuntimeError):
    """A chunk file that cannot be used (wrong format or contents); the reason is for the user."""


def load_mat_stable(path: str, tries: int = 25, wait: float = 0.2):
    """Load a chunk that may still be being written: its size must be stable and
    loadmat must succeed. Returns (data (N, C), time (N,) or None).
    A complete file that cannot be used raises ChunkReadError with the reason at once."""
    last, err = -1, None
    for _ in range(tries):
        try:
            size = os.path.getsize(path)
            settled = time.time() - os.path.getmtime(path) > 2.0      # an older file is complete
            if size > 0 and (size == last or settled):
                try:
                    mat = sio.loadmat(path, squeeze_me=False)
                except NotImplementedError:                   # scipy: MATLAB v7.3 (HDF5) file
                    raise ChunkReadError("saved as a MATLAB v7.3 file, which cannot be read - re-save it in "
                                         "MATLAB with save(..., '-v7')")
                except Exception as e:
                    if settled:
                        raise ChunkReadError(f"not a readable .mat file ({e})")
                    raise
                if "data" not in mat:
                    names = ", ".join(k for k in mat if not k.startswith("__")) or "none"
                    raise ChunkReadError(f"no variable 'data' in the file (variables: {names}); it must hold "
                                         f"the signal as samples x channels, plus 'time' in seconds")
                raw = np.asarray(mat["data"], dtype=np.float64)
                if raw.ndim == 1:
                    raw = raw.reshape(-1, 1)
                if raw.shape[0] < raw.shape[1]:
                    raw = raw.T
                t = np.asarray(mat["time"]).ravel().astype(np.float64) if "time" in mat else None
                return raw, t
            last = size
        except ChunkReadError:
            raise
        except Exception as e:
            last, err = -1, e
        time.sleep(wait)
    raise ChunkReadError(f"could not be read{f' ({err})' if err else ''}")


class ChunkSource:
    def __init__(self, save_dir, exp_name, n_fibers=None, root=None):
        self.root = root or os.path.join(save_dir, exp_name)     # root: the folder with F1, F2, ... (any name)
        self.exp_name = exp_name
        self.n_fibers = n_fibers           # None: decided from the first chunk
        self.pattern = re.compile(rf"^{re.escape(exp_name)}_F1_(\d+)\.mat$")

    def path(self, fiber, idx):
        return os.path.join(self.root, f"F{fiber}", f"{self.exp_name}_F{fiber}_{idx}.mat")

    def first_index(self):
        f1 = os.path.join(self.root, "F1")
        if not os.path.isdir(f1):
            return None
        idx = [int(m.group(1)) for fn in os.listdir(f1) if (m := self.pattern.match(fn))]
        return min(idx) if idx else None

    def ready(self, idx):
        fibers = range(1, (self.n_fibers or 1) + 1)
        return all(os.path.exists(self.path(f, idx)) for f in fibers)

    def load(self, idx):
        """(ch1, ch2, time vector (or None), file names) of chunk idx; columns of F1..Fn joined in order."""
        cols, names, t = [], [], None
        f = 1
        while True:
            p = self.path(f, idx)
            if self.n_fibers is None and not os.path.exists(p):
                # layout not known yet: DiDAQ writes F1, F2, ... of a chunk in turn -> wait for it
                t_wait = time.time()
                while not os.path.exists(p) and time.time() - t_wait < 10 and os.path.isdir(os.path.dirname(p)):
                    time.sleep(0.2)
                if not os.path.exists(p):
                    break
            try:
                d, tf = load_mat_stable(p)
            except ChunkReadError as e:
                raise ChunkReadError(f"{os.path.basename(p)}: {e}") from None
            cols.append(d)
            names.append(os.path.basename(p))
            t = tf if t is None else t
            if self.n_fibers is None and sum(c.shape[1] for c in cols) >= 2:
                break
            if self.n_fibers is not None and f >= self.n_fibers:
                break
            f += 1
        n = min(c.shape[0] for c in cols)
        if self.n_fibers is not None and self.n_fibers >= 2 and len(cols) >= 2:
            cols = [c[:, :1] for c in cols[:2]]                  # Ch1 = F1 (1st column), Ch2 = F2 (1st column)
        data = np.hstack([c[:n] for c in cols])
        if data.shape[1] < 2:
            raise ChunkReadError(f"{names[0]}: only {data.shape[1]} channel in {', '.join(names)}; Ch2 is expected "
                                 f"in F1 (2nd column) or in F2 (--n_fibers 2)")
        if self.n_fibers is None:
            self.n_fibers = len(cols)
            log_scan.info("channels: Ch1, Ch2 = first two columns of %s",
                          " + ".join(f"F{k}" for k in range(1, self.n_fibers + 1)))
            if self.n_fibers == 1 and os.path.exists(self.path(2, idx)):
                log_scan.warning("F1 has two columns, so Ch2 is its 2nd column although there is an F2 folder; "
                                 "for Ch2 from F2 choose Fiber folders = 2 (--n_fibers 2)")
        if t is not None and len(t) < n:
            t = t[:1] if len(t) else None                     # no full time vector: only its start is used
        return data[:, 0], data[:, 1], (t[:n] if t is not None and len(t) >= n else t), names


# ─── a folder of .mat files with any names ───────────────────────────────────
_FIBER_TAG = re.compile(r"(?:^|(?<=[_\-. ]))F(\d+)(?=$|[_\-. ])")      # "F1" as its own part of a name
_TRAILING = re.compile(r"[_\-. ]*\d+$")


def natural_key(s):
    """Sort key of a name with its numbers as numbers: x_2 before x_10."""
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in re.split(r"(\d+)", s) if p)


def mat_stem(fname):
    """A .mat file's name without its F<k> tag and its number: exp_12.mat, exp_12_F1.mat, exp_F1_12.mat -> exp."""
    s = _FIBER_TAG.sub("", os.path.splitext(fname)[0], count=1).rstrip("_-. ")
    return _TRAILING.sub("", s, count=1).rstrip("_-. ")


def scan_mat_folder(folder):
    """The .mat files of a folder grouped as chunks, in the order of their names.
    Returns (chunks, stem, ignored): chunks is [{fiber: file name}] (a file without an F<k> tag is F1);
    only the files with the most common name stem are used, the others are listed in ignored."""
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".mat") and not f.startswith(".")
                 and os.path.isfile(os.path.join(folder, f))]
    except OSError:
        return [], "", []
    stems = {}
    for f in files:
        stems.setdefault(mat_stem(f), []).append(f)
    if not stems:
        return [], "", []
    stem = max(stems, key=lambda k: (len(stems[k]), k))
    groups = {}
    for f in stems[stem]:
        base = os.path.splitext(f)[0]
        m = _FIBER_TAG.search(base)
        fiber = int(m.group(1)) if m else 1
        key = _FIBER_TAG.sub("", base) if m else base
        groups.setdefault(key, {})[fiber] = f
    order = sorted(groups, key=natural_key)
    ignored = sorted((f for k, v in stems.items() if k != stem for f in v), key=natural_key)
    return [groups[k] for k in order], stem, ignored


class FolderSource:
    """Chunks = the .mat files of one folder in the order of their names (numbers sorted as numbers):
    exp_1.mat, exp_2.mat ... / exp_1_F1.mat ... / one long file without a number. Files with an F2 tag
    next to the F1 ones give Ch2 as in the DiDAQ layout. Files that appear later (a recording) are
    added at the end. Chunk numbers are 1, 2, ... in that order."""

    def __init__(self, folder, exp_name, n_fibers=None, f2_folder=None):
        self.root = folder
        self.exp_name = exp_name
        self.n_fibers = n_fibers
        self.f2_folder = f2_folder        # F1/ and F2/ folders with any file names: F2's files give Ch2
        self._chunks, self._keys, self._scanned = [], set(), 0.0

    def _rescan(self, force=False):
        if not force and time.time() - self._scanned < 1.0:
            return
        self._scanned = time.time()
        chunks, stem, ignored = scan_mat_folder(self.root)
        new = [c for c in chunks if c.get(1, next(iter(c.values()))) not in self._keys]
        if new and self._chunks and natural_key(next(iter(new[0].values()))) < natural_key(
                next(iter(self._chunks[-1].values()))):
            log_scan.warning("%s sorts before files already analysed - analysed after them",
                             next(iter(new[0].values())))
        if not self._chunks and ignored:
            log_scan.info("not used (other file names than %s_*.mat): %s", stem,
                          ", ".join(ignored[:10]) + (" ..." if len(ignored) > 10 else ""))
        for c in new:
            self._keys.add(c.get(1, next(iter(c.values()))))
            self._chunks.append(c)
        known = {k: c for c in chunks for k in [c.get(1, next(iter(c.values())))]}
        for c in self._chunks:                                    # an F2 file written after its F1 file
            c.update(known.get(c.get(1, next(iter(c.values()))), {}))
        if self.f2_folder:                                        # Ch2 files in the F2 folder
            other = scan_mat_folder(self.f2_folder)[0]
            key = lambda c: _FIBER_TAG.sub("", os.path.splitext(next(iter(c.values())))[0])
            by_key = {key(c): c for c in other}
            same = sum(key(c) in by_key for c in self._chunks)
            for i, c in enumerate(self._chunks):                  # same name (without F1/F2), else same place
                o = by_key.get(key(c)) if same else (other[i] if i < len(other) else None)
                if o and 2 not in c:
                    c[2] = os.path.join(self.f2_folder, next(iter(o.values())))

    def first_index(self):
        self._rescan(force=True)
        return 1 if self._chunks else None

    def ready(self, idx):
        if idx > len(self._chunks):
            self._rescan()
        if not 1 <= idx <= len(self._chunks):
            return False
        c = self._chunks[idx - 1]
        return all(f in c for f in range(1, (self.n_fibers or 1) + 1))

    def load(self, idx):
        """(ch1, ch2, time vector (or None), file names) of chunk idx; columns of F1..Fn joined in order."""
        c = self._chunks[idx - 1]
        cols, names, t = [], [], None
        for f in range(1, 1 + max(c)):
            if f not in c and self.n_fibers is None and sum(x.shape[1] for x in cols) < 2:
                t_wait = time.time()                              # Ch2 in an F2 file not written yet?
                while f not in c and time.time() - t_wait < 10:
                    time.sleep(0.5)
                    self._rescan(force=True)
            if f not in c:
                break
            p = os.path.join(self.root, c[f])
            try:
                d, tf = load_mat_stable(p)
            except ChunkReadError as e:
                raise ChunkReadError(f"{os.path.basename(c[f])}: {e}") from None
            cols.append(d)
            names.append(os.path.basename(c[f]))
            t = tf if t is None else t
            if (self.n_fibers is None and sum(x.shape[1] for x in cols) >= 2) or \
                    (self.n_fibers is not None and f >= self.n_fibers):
                break
        n = min(x.shape[0] for x in cols)
        if self.n_fibers is not None and self.n_fibers >= 2 and len(cols) >= 2:
            cols = [x[:, :1] for x in cols[:2]]                  # Ch1 = F1 (1st column), Ch2 = F2 (1st column)
        data = np.hstack([x[:n] for x in cols])
        if data.shape[1] < 2:
            raise ChunkReadError(f"{names[0]}: only {data.shape[1]} channel; Ch2 is expected as the 2nd column "
                                 f"of 'data' or in a matching F2 file")
        if self.n_fibers is None:
            self.n_fibers = len(cols)
            log_scan.info("channels: Ch1, Ch2 = first two columns of %s (files in name order in %s)",
                          " + ".join(f"F{k}" for k in range(1, self.n_fibers + 1)), self.root)
            if self.n_fibers == 1 and 2 in c:
                log_scan.warning("F1 has two columns, so Ch2 is its 2nd column although there are F2 files; "
                                 "for Ch2 from F2 choose Fiber folders = 2 (--n_fibers 2)")
        if t is not None and len(t) < n:
            t = t[:1] if len(t) else None
        return data[:, 0], data[:, 1], (t[:n] if t is not None and len(t) >= n else t), names


def didaq_layout(folder, exp_name=None):
    """(name, chunk count) if the folder holds DiDAQ chunks F1/<name>_F1_<n>.mat, else (None, 0).
    exp_name is used if F1 has its chunks, else the most common name in F1."""
    f1 = os.path.join(folder, "F1")
    try:
        files = os.listdir(f1) if os.path.isdir(f1) else []
    except OSError:
        files = []
    names = {}
    for f in files:
        if m := re.match(r"^(.+)_F1_\d+\.mat$", f):
            names[m.group(1)] = names.get(m.group(1), 0) + 1
    if not names:
        return None, 0
    if exp_name in names:
        return exp_name, names[exp_name]
    return max(names.items(), key=lambda kv: kv[1])


class AutoSource:
    """The DiDAQ layout (F1/, F2/ ...) if the folder has it, else its own .mat files in name order.
    Decided when the first data is there, so the app can be started before the recording."""

    def __init__(self, folder, exp_name, n_fibers=None):
        self.root, self.exp_name, self._n = folder, exp_name, n_fibers
        self.impl = None

    def __getattr__(self, k):                                     # n_fibers etc. of the chosen source
        if k == "impl":
            raise AttributeError(k)
        if self.impl is None:
            if k == "n_fibers":
                return self._n
            raise AttributeError(k)
        return getattr(self.impl, k)

    def first_index(self):
        if self.impl is None:
            name, _ = didaq_layout(self.root, self.exp_name)
            if name:
                self.impl = ChunkSource(os.path.dirname(self.root), name, self._n, root=self.root)
                log_scan.info("DiDAQ layout: %s/F1/%s_F1_<n>.mat", self.root, name)
            else:
                for d in (self.root, os.path.join(self.root, "F1")):  # its own files (or those in F1)
                    if scan_mat_folder(d)[0]:
                        f2 = os.path.join(self.root, "F2")
                        f2 = f2 if d != self.root and os.path.isdir(f2) else None
                        self.impl = FolderSource(d, self.exp_name, self._n, f2_folder=f2)
                        if f2:
                            log_scan.info("Ch2 can come from the files in %s (same names, else same order)", f2)
                        log_scan.info("a folder of .mat files (%s): read in the order of their names", d)
                        break
                else:
                    return None
        return self.impl.first_index()

    def ready(self, idx):
        return self.impl is not None and self.impl.ready(idx)

    def load(self, idx):
        return self.impl.load(idx)


def time_segments(t, n):
    """[(offset, length, time0)] of the stretches of a file's n samples without a gap in its `time`
    vector (a step of more than TIME_GAP_SECS, or backwards). A 10-s DiDAQ chunk has one; a file with
    chunks stitched together has one more per pause / lost data in it. time0 None: no time vector."""
    if t is None or len(t) == 0:
        return [(0, n, None)]
    if len(t) < n:
        return [(0, n, float(t[0]))]
    dt = np.diff(t[:n])
    cuts = np.flatnonzero((dt > 1.0 / SAMPLING_RATE + TIME_GAP_SECS) | (dt < -TIME_GAP_SECS)) + 1
    edges = [0, *cuts.tolist(), n]
    return [(a, b - a, float(t[a])) for a, b in zip(edges[:-1], edges[1:])]


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD  - never touches Qt
# ─────────────────────────────────────────────────────────────────────────────
def safe_cut(ch1, lo, hi, thr, min_off):
    """Middle of the last laser-OFF run (ch1 <= thr, >= min_off samples) inside [lo, hi), or None."""
    lo, hi = max(0, int(lo)), min(len(ch1), int(hi))
    if hi - lo < min_off:
        return None
    off = (ch1[lo:hi] <= thr).astype(np.int8)
    d = np.diff(np.concatenate([[0], off, [0]]))
    runs = [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_off]
    return lo + (runs[-1][0] + runs[-1][1]) // 2 if runs else None


class Worker(threading.Thread):
    CSV_COLS = ["sample", "time_s", "time_h", "chunk", "source_file", "laser_mode", "at_laser_edge",
                f"p_{paths.MODEL_NAME}", "detected", "analysed_as"]

    def __init__(self, source: ChunkSource, out_dir: str, mode: str, model_path: str, cut_off=None,
                 analysis=DEFAULT_ANALYSIS):
        """analysis: 'offline-then-real-time' (data saved before the start: offline pipeline, then new
        data in real time), 'real-time' (everything with the real-time algorithm, from the first chunk)
        or 'offline' (only the saved data, offline pipeline, to the end; no new data is read)."""
        super().__init__(daemon=True)
        assert analysis in ANALYSIS_MODES, analysis
        self.analysis = analysis
        self.finished = False
        self.src = source
        self.out_dir = out_dir
        self.stream = StreamProcessor(mode=mode)
        self.model_path = model_path
        self.cut_off = cut_off
        self.model = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()   # set: no new chunks are read (the DAQ keeps writing them)
        self.chunks = []            # (start sample, end sample, chunk index, time0, file name)
        self._chunk_starts = []     # start samples of self.chunks (bisect in _time_of)
        self.next_idx = None
        self.counts = dict(candidates=0, detected=0, detected_not_at_laser_edge=0)
        self.backlog = None         # what the offline analysis of the saved data did (summary.json)
        self._y = dict(n=0, max=-np.inf, seen=set(), done=False)   # signal maximum at the start
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "candidates.csv")
        with open(self.csv_path, "w") as f:
            f.write(",".join(self.CSV_COLS) + "\n")

    def run(self):
        torch.set_num_threads(TORCH_THREADS)
        self.model = FusionEnsemble(self.model_path)
        if self.cut_off is None:
            self.cut_off = self.model.cut_off
        log_worker.info("Model loaded from %s (%d seeds, cut-off %.2f)",
                        self.model_path, len(self.model.nets), self.cut_off)
        PLOT_QUEUE.put({"kind": "config", "cut_off": self.cut_off})
        waiting_since = None
        last_report = 0.0
        caught_up = False
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                if time.time() - last_report > 2.0:          # tell the GUI how much is waiting
                    PLOT_QUEUE.put({"kind": "paused", "waiting_s": self._chunks_waiting() * self._chunk_secs()})
                    last_report = time.time()
                time.sleep(0.2)
                continue
            last_report = 0.0
            try:
                if self.next_idx is None:
                    self.next_idx = self.src.first_index()
                    if self.next_idx is None:
                        if self.analysis == "offline":         # offline: only what is saved - nothing
                            self._finish_offline(none_found=True)
                            return
                        time.sleep(POLL_SECS)
                        continue
                    log_scan.info("first chunk: %d (%s)", self.next_idx, self.src.root)
                if not caught_up:                           # data saved before the app started
                    caught_up = True
                    if self.analysis == "real-time":        # real-time algorithm from the first chunk:
                        self._rt_saved = (self.next_idx, self._count_ready_from(self.next_idx))   # (progress)
                        self._rt_ticks = 0
                        continue
                    self._catch_up_offline()                # offline pipeline (then real time)
                    if self.analysis == "offline":
                        self._finish_offline()
                        return                              # no new data is read
                    continue
                if self.src.ready(self.next_idx):
                    self._handle_chunk(self.next_idx)
                    self.next_idx += 1
                    waiting_since = None
                    continue
                later = any(self.src.ready(self.next_idx + k) for k in range(1, 6))
                if later:
                    waiting_since = waiting_since or time.time()
                    if time.time() - waiting_since > GAP_TIMEOUT_SECS:
                        log_scan.warning("chunk %d missing for %d s while later chunks exist - skipped",
                                         self.next_idx, GAP_TIMEOUT_SECS)
                        self.next_idx += 1
                        waiting_since = None
                        continue
                time.sleep(POLL_SECS)
            except ChunkReadError as e:
                self._read_error(self.next_idx, e)
                self.next_idx = (self.next_idx or 0) + 1
            except Exception:
                log_worker.error("chunk %s - error:\n%s", self.next_idx, traceback.format_exc())
                PLOT_QUEUE.put({"kind": "progress", "text": None})
                self.next_idx = (self.next_idx or 0) + 1
                time.sleep(POLL_SECS)

    # ------------------------------------------- saved data: offline pipeline ---
    def _progress(self, text, frac=None):
        """Status line text (None: done) and the progress bar of the current block (0-1, None: no bar)."""
        PLOT_QUEUE.put({"kind": "progress", "text": text, "frac": frac})

    @staticmethod
    def _hm(sec):
        m = int(max(0.0, sec) // 60)
        return f"{m // 60}:{m % 60:02d}"

    def _block_progress(self, t0, t1, stage, frac):
        self._progress(f"Analysing the saved data (offline mode): {self._hm(t0)} - {self._hm(t1)} h "
                       f"(block {self._block_no + 1} of {self._n_blocks}) - {stage}", frac)

    def _chunk_secs(self):
        return (self.chunks[-1][1] - self.chunks[-1][0]) / SAMPLING_RATE if self.chunks else 10.0

    def _finish_offline(self, none_found=False):
        """--analysis offline: everything saved is analysed; no new data is read."""
        self.finished = True
        if none_found:
            text = f"Offline analysis: no saved data in {self.src.root}"
        else:
            t1 = self._time_of(self.chunks[-1][1])[0] if self.chunks else 0.0
            text = (f"Offline analysis finished: {self._hm(self._t_first)} - {self._hm(t1)} h, "
                    f"{self.counts['detected']} detected peaks of {self.counts['candidates']} candidates "
                    f"(no new data is read)")
        log_worker.info(text)
        self._progress(None)
        PLOT_QUEUE.put({"kind": "finished", "text": text})

    def _wait_if_paused(self):
        """Pause pressed during the offline analysis of the saved data: wait here (between chunk reads,
        before a block's preprocessing, between its scoring batches), telling the GUI what is waiting."""
        last = 0.0
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if time.time() - last > 2.0:
                PLOT_QUEUE.put({"kind": "paused", "waiting_s": self._waiting_offline() * self._chunk_secs()})
                last = time.time()
            time.sleep(0.2)

    def _waiting_offline(self):
        """Chunks not analysed yet during the offline part: read but not yet in a finished block, not
        read yet, and new ones saved after the list was made."""
        idxs = getattr(self, "_co_idxs", None)
        if not idxs:
            return 0
        pend_start = self._pend[2] if getattr(self, "_pend", None) else 0
        read_not_done = len(self.chunks) - bisect.bisect_left(self._chunk_starts, pend_start)
        return read_not_done + (len(idxs) - getattr(self, "_co_read", 0) - 1) + self._count_ready_from(idxs[-1] + 1)

    def _list_saved(self):
        """Indices of the complete chunks on disk from next_idx on (a missing one is skipped, as live)."""
        idxs, i = [], self.next_idx
        while not self.stop_event.is_set():
            if self.src.ready(i):
                idxs.append(i)
            elif not any(self.src.ready(i + k) for k in range(1, 6)):
                break
            i += 1
        return idxs

    def _add_chunk(self, idx, n, tvec, fname):
        """Chunk table entries of n new samples of file idx: one per stretch without a gap in its time
        vector, each with its own start time, so every peak keeps its recording time. Returns the start
        sample and the recording time of every sample (n,)."""
        fs = SAMPLING_RATE
        start = self.chunks[-1][1] if self.chunks else 0
        times = np.empty(n)
        segs = time_segments(tvec, n)
        back = 0
        for off, ln, t0 in segs:
            prev_end = (self.chunks[-1][3] + (self.chunks[-1][1] - self.chunks[-1][0]) / fs) if self.chunks else None
            if t0 is None:                                     # no time vector: continues the previous one
                t0 = prev_end if prev_end is not None else (idx - 1) * n / fs
            elif prev_end is not None and t0 < prev_end - TIME_GAP_SECS:
                back += 1                                      # time runs backwards: keep the clock going
                t0 = prev_end
            self.chunks.append((start + off, start + off + ln, idx, t0, fname))
            self._chunk_starts.append(start + off)
            times[off:off + ln] = t0 + np.arange(ln) / fs
        if len(segs) > 1:
            jumps = [segs[k][2] - (segs[k - 1][2] + segs[k - 1][1] / fs) for k in range(1, len(segs))]
            log_scan.warning("%s: %d gap(s) in its time vector (largest %.1f s) - times follow the time vector",
                             fname, len(segs) - 1, max(jumps))
        if back:
            log_scan.warning("%s: its time vector runs backwards %d time(s) - the clock is continued there",
                             fname, back)
        return start, times

    def _catch_up_offline(self):
        """The chunks already saved when the app starts are analysed as by the offline pipeline:
        in OFFLINE_BLOCK_SECS blocks (its recording-file length), each with its own pulsed /
        continuous decision and Otsu threshold (scan_preprocessing.preprocess_pulsed / _regular,
        build_windows), scored in one batch. Unlike the offline files, a block ends in a laser-OFF
        gap, so no pulse is cut in two. The real-time analysis then continues from the last such
        gap before the end of the saved data (StreamProcessor.seed), so nothing is counted twice."""
        t_start = time.time()
        while True:                                           # the first readable file (fixes the channel layout)
            first = self.next_idx
            try:
                ch1, ch2, tvec, names = self.src.load(first)
                self._y_check(first, ch1, ch2)
                break
            except ChunkReadError as e:
                self._read_error(first, e)
                self.next_idx += 1
                if not self.src.ready(self.next_idx):
                    return
        time0 = float(tvec[0]) if tvec is not None and len(tvec) else None
        self._progress("Analysing the saved data (offline mode): looking for the saved data…")
        idxs = self._list_saved()
        if self.analysis != "offline" and len(idxs) * len(ch1) < OFFLINE_MIN_SECS * SAMPLING_RATE:
            self._progress(None)
            return                                            # little saved data: real time from the start
        fs, block = SAMPLING_RATE, int(OFFLINE_BLOCK_SECS * SAMPLING_RATE)
        keep = max(self.stream.keep, self.stream.mode_win) + self.stream.context
        n_total_est = len(idxs) * len(ch1)
        self._t_first = time0 if time0 is not None else (first - 1) * len(ch1) / fs   # recording time of the data
        self._n_total_est = n_total_est
        self._disp_from = n_total_est - int((DISPLAY_HISTORY_SECS + TRIM_STEP_SECS) * fs)
        self._n_blocks = max(1, int(np.ceil(n_total_est / block)))
        self._block_no, self._on_off = 0, 0
        self._pend = ([], [], 0)                              # pending raw Ch1, Ch2 pieces, abs start
        self._ctx = (np.empty(0), np.empty(0))                # raw just before the pending data
        log_worker.info("saved data: %d chunks (%.2f h) - offline pipeline in %d block(s), then real time",
                        len(idxs), n_total_est / fs / 3600, self._n_blocks)
        self._co_idxs = idxs
        for k, idx in enumerate(idxs):
            self._co_read = k                                 # saved chunks read so far
            self._wait_if_paused()
            if self.stop_event.is_set():
                return
            if idx != first:
                try:
                    ch1, ch2, tvec, names = self.src.load(idx)
                    self._y_check(idx, ch1, ch2)
                except ChunkReadError as e:
                    self._read_error(idx, e)
                    continue                                  # skipped, as a missing chunk
            self._add_chunk(idx, len(ch1), tvec, names[0])
            p1, p2, ps = self._pend
            p1.append(ch1); p2.append(ch2)
            if k % 30 == 0:                                   # reading: first 40% of the block's bar
                pend_n = sum(len(x) for x in p1)
                blk_n = min(block, max(1, n_total_est - ps))
                t0 = self._t_first + ps / fs
                self._block_progress(t0, t0 + blk_n / fs, "reading the data", 0.4 * min(1.0, pend_n / blk_n))
            # (while: a long file, e.g. chunks stitched together, gives several 2-h blocks)
            while sum(len(x) for x in self._pend[0]) >= block + SAFE_CUT_SEARCH_SECS * fs:
                self._offline_block(final=False)
                if self.stop_event.is_set():
                    return
        self._offline_block(final=True)
        self.next_idx = idxs[-1] + 1
        self.backlog = {"chunks": len(idxs), "first_chunk": idxs[0], "last_chunk": idxs[-1],
                        "blocks": self._block_no, "hours": round(self.chunks[-1][1] / fs / 3600, 4),
                        "real_time_from_s": (round(self._time_of(self.stream.commit)[0], 3)
                                             if self.analysis != "offline" else None),
                        "seconds_taken": round(time.time() - t_start, 1)}
        log_worker.info("saved data analysed offline in %.0f s (%d candidates, %d detected)%s",
                        time.time() - t_start, self.counts["candidates"], self.counts["detected"],
                        "" if self.analysis == "offline" else
                        f"; real time from t = {self.backlog['real_time_from_s']:.1f} s")
        self._write_summary(idxs[-1], *self._last_block)
        self._progress(None)

    def _offline_block(self, final):
        fs = SAMPLING_RATE
        p1, p2, start = self._pend
        d1, d2 = np.concatenate(p1), np.concatenate(p2)
        mode_req = self.stream.mode_req
        otsu = float(pkd.OtsuThreshold(d1, bins_num=scp.OTSU_BINS))
        pulsed = looks_pulsed(d1, otsu) if mode_req == "auto" else mode_req == "pulsed"
        search, min_off = int(SAFE_CUT_SEARCH_SECS * fs), int(SAFE_CUT_OFF_SECS * fs)
        if final and self.analysis == "offline":   # offline only: to the very end, as an offline file
            H = len(d1)
        elif final:    # hand over to real time in the last laser-OFF gap (continuous: 30 s before the end)
            H = safe_cut(d1, len(d1) - search, len(d1), otsu, min_off) if pulsed else None
            H = H if H is not None else max(0, len(d1) - 30 * fs)
        else:          # block of ~2 h, ending in a laser-OFF gap
            nominal = int(OFFLINE_BLOCK_SECS * fs)
            H = safe_cut(d1, nominal - search, nominal + search, otsu, min_off) if pulsed else None
            H = H if H is not None else nominal
        mode = "pulsed" if pulsed else "continuous"
        self._last_block = (mode, otsu)
        cands = []
        t0, t1 = self._t_first + start / fs, self._t_first + (start + H) / fs
        if H >= scp.WIN_LEN:
            self._wait_if_paused()
            self._block_progress(t0, t1, f"preprocessing ({mode} laser)", 0.45)
            df = pd.DataFrame({"Ch1": d1[:H], "Ch2": d2[:H]})
            s1, s2, pk, info = (scp.preprocess_pulsed if pulsed else scp.preprocess_regular)(df)
            raw, X, M = scp.build_windows(s1, s2, pk)
            edge = scp.edge_distance_ms(M) <= scp.EDGE_MS if len(pk) else np.empty(0, bool)
            probs = []
            for a in range(0, len(pk), 200):                  # batch scoring: the last half of the bar
                self._wait_if_paused()
                self._block_progress(t0, t1, f"scoring the candidates ({a} of {len(pk)})", 0.5 + 0.5 * a / len(pk))
                probs.append(self.model.predict(X[a:a + 200], M[a:a + 200]))
            probs = np.concatenate(probs) if probs else np.empty(0)
            self._block_progress(t0, t1, f"done ({len(pk)} candidates)", 1.0)
            cands = self._record(start + pk, probs, raw, edge, mode, "offline")
            self._on_off += int(round(info["on_hours"] * 3600 * fs))
            log_worker.info("offline block %d: %.2f h %s, %d candidates, %d detected", self._block_no + 1,
                            H / fs / 3600, mode, len(pk), int((probs >= self.cut_off).sum()) if len(pk) else 0)
        self._block_no += 1                                   # blocks finished (labels show the next one)
        # display: the part of this block inside the trace history (and, at the end, the rest too)
        upto = len(d1) if final else H
        t_plot, d_plot = self._display_range(start, start + upto, d1, d2, start, mode, otsu)
        c1, c2 = self._ctx
        c1, c2 = np.concatenate([c1, d1[:H]]), np.concatenate([c2, d2[:H]])
        keep = max(self.stream.keep, self.stream.mode_win) + self.stream.context
        self._ctx = (c1[-keep:], c2[-keep:])
        self._pend = ([d1[H:]], [d2[H:]], start + H)
        if final and self.analysis != "offline":   # real time continues at start + H (last minutes as context)
            r1 = np.concatenate([self._ctx[0], d1[H:]])
            r2 = np.concatenate([self._ctx[1], d2[H:]])
            self.stream.seed(r1, r2, start + H - len(self._ctx[0]), start + H, self._on_off)
        summary = self._write_summary(self.chunks[-1][2], mode, otsu,
                                      n_total=start + upto, on_samples=self._on_off)
        PLOT_QUEUE.put({"kind": "chunk", "chunk": self.chunks[-1][2], "t_plot": t_plot, "d_plot": d_plot,
                        "cands": cands, "final_t": self._time_of(start + H)[0], "summary": summary})

    def _display_range(self, a, b, d1, d2, d_start, mode, otsu):
        """Trace points for absolute samples [a, b) (only inside the trace history), chunk by chunk
        (each chunk's own DAQ time)."""
        a = max(a, getattr(self, "_disp_from", a))
        ts, ds = [], []
        k = max(0, bisect.bisect_right(self._chunk_starts, a) - 1)
        while k < len(self.chunks) and self.chunks[k][0] < b:
            s0, e0, _, t0, _ = self.chunks[k]
            lo, hi = max(s0, a), min(e0, b)
            if hi > lo:
                t = t0 + (np.arange(lo, hi) - s0) / SAMPLING_RATE
                d = np.stack([d1[lo - d_start:hi - d_start], d2[lo - d_start:hi - d_start]], 1)
                tp, dp = display_chunk(t, d, mode, otsu)
                ts.append(tp); ds.append(dp)
            k += 1
        if not ts:
            return np.empty(0), np.empty((0, 2), np.float32)
        return np.concatenate(ts), np.vstack(ds)

    # ------------------------------------------------- candidates and summary ---
    def _record(self, peaks, probs, raw, edge, mode, analysed_as):
        """Candidate dicts for the GUI and rows in candidates.csv."""
        cands, rows = [], []
        for k, p in enumerate(peaks):
            t_s, chunk, fname = self._time_of(int(p))
            det = bool(probs[k] >= self.cut_off)
            cands.append({"t": float(t_s), "prob": float(probs[k]), "edge": bool(edge[k]),
                          "sample": int(round(t_s * SAMPLING_RATE)), "file": fname,
                          "clip_d": raw[k].T.astype(np.float32)})
            rows.append([int(round(t_s * SAMPLING_RATE)), f"{t_s:.4f}", f"{t_s / 3600:.6f}", chunk, fname,
                         mode, bool(edge[k]), f"{probs[k]:.6f}", det, analysed_as])
            self.counts["candidates"] += 1
            self.counts["detected"] += det
            self.counts["detected_not_at_laser_edge"] += det and not edge[k]
        if rows:
            with open(self.csv_path, "a") as f:
                for r in rows:
                    f.write(",".join(map(str, r)) + "\n")
        return cands

    def _write_summary(self, last_chunk, mode, otsu, n_total=None, on_samples=None):
        n_total = self.stream.n if n_total is None else n_total
        on_samples = self.stream.on_samples if on_samples is None else on_samples
        on_h = on_samples / SAMPLING_RATE / 3600
        summary = {"exp_name": self.src.exp_name, "chunks": len(self.chunks), "last_chunk": last_chunk,
                   "hours_recorded": round(n_total / SAMPLING_RATE / 3600, 4),
                   "laser_mode": mode, "otsu_threshold": otsu,
                   "laser_on_hours": round(on_h, 4), **self.counts,
                   "peaks_per_laser_on_hour": round(self.counts["detected"] / on_h, 2) if on_h else None,
                   "cut_off": self.cut_off, "model": os.path.basename(self.model_path),
                   "analysis": self.analysis, "saved_data_analysed_offline": self.backlog,
                   "finished": self.finished,
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = os.path.join(self.out_dir, ".summary.json.tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1)
        os.replace(tmp, os.path.join(self.out_dir, "summary.json"))
        return summary

    def _chunks_waiting(self):
        return 0 if self.next_idx is None else self._count_ready_from(self.next_idx)

    def _count_ready_from(self, i):
        """Complete chunks on disk from chunk i on (exponential + binary search: a few file checks
        even after hours of pause). Chunks are numbered consecutively by DiDAQ."""
        if not self.src.ready(i):
            return 0
        hi = 1
        while self.src.ready(i + hi):
            hi *= 2
        lo = hi // 2                                          # ready(i + lo), not ready(i + hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if self.src.ready(i + mid) else (lo, mid)
        return lo + 1

    # -------------------------------------------------------------------------
    def _time_of(self, sample):
        """Recording time (s) of a stream sample: DiDAQ time of its chunk + offset in the chunk."""
        k = bisect.bisect_right(self._chunk_starts, sample) - 1
        s0, _, _, t0, fname = self.chunks[max(k, 0)]
        return t0 + (sample - s0) / SAMPLING_RATE, self.chunks[max(k, 0)][2], fname

    def _read_error(self, idx, err):
        """A file that cannot be used: logged and shown in the window; it is skipped as a missing chunk."""
        self.read_errors = getattr(self, "read_errors", 0) + 1
        log_scan.error("chunk %s skipped - %s", idx, err)
        PLOT_QUEUE.put({"kind": "error", "text": str(err), "count": self.read_errors})

    def _y_check(self, idx, ch1, ch2):
        """Maximum of Ch1 / Ch2 in the first Y_AUTO_SECS of the recording -> the GUI (y limit)."""
        y = self._y
        if y["done"] or idx in y["seen"]:
            return
        y["seen"].add(idx)
        need = int(Y_AUTO_SECS * SAMPLING_RATE) - y["n"]
        a1, a2 = np.asarray(ch1[:need]), np.asarray(ch2[:need])
        if len(a1):
            y["max"] = max(y["max"], float(np.nanmax(a1)), float(np.nanmax(a2)))
        y["n"] += len(a1)
        if y["n"] >= int(Y_AUTO_SECS * SAMPLING_RATE):
            y["done"] = True
            PLOT_QUEUE.put({"kind": "y_max", "value": y["max"]})

    def _handle_chunk(self, idx):
        """One new file in real time. A file longer than RT_SLICE_SECS (e.g. 10-s chunks stitched
        together) is fed in RT_SLICE_SECS slices, as if DiDAQ had written it chunk by chunk."""
        try:
            ch1, ch2, tvec, names = self.src.load(idx)
        except ChunkReadError as e:
            self._read_error(idx, e)
            return
        self._y_check(idx, ch1, ch2)
        n, step = len(ch1), int(RT_SLICE_SECS * SAMPLING_RATE)
        n_sl = int(np.ceil(n / step)) if n > 1.5 * step else 1
        if n_sl > 1:
            log_scan.info("%s: %.2f h in one file - analysed in %d slices of %d s", names[0],
                          n / SAMPLING_RATE / 3600, n_sl, RT_SLICE_SECS)
        for k in range(n_sl):
            a, b = (k * step, min(n, (k + 1) * step)) if n_sl > 1 else (0, n)
            if k:
                self._wait_slice(n - a)
                if self.stop_event.is_set():
                    return
            tv = tvec[a:b] if tvec is not None and len(tvec) >= n else (tvec if k == 0 else None)
            self._rt_piece(idx, ch1[a:b], ch2[a:b], tv, names, (k + 1) / n_sl, n)

    def _wait_slice(self, left):
        """Pause pressed while a long file is fed in slices: wait between two slices."""
        last = 0.0
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if time.time() - last > 2.0:
                waiting = left / SAMPLING_RATE + self._count_ready_from(self.next_idx + 1) * self._chunk_secs()
                PLOT_QUEUE.put({"kind": "paused", "waiting_s": waiting})
                last = time.time()
            time.sleep(0.2)

    def _rt_piece(self, idx, ch1, ch2, tvec, names, file_frac, file_n):
        """Real-time analysis of one 10-s piece (a whole DiDAQ chunk, or a slice of a longer file)."""
        n0, t_disp = self._add_chunk(idx, len(ch1), tvec, names[0])
        assert n0 == self.stream.n, "chunk table and stream out of step"
        time0 = float(t_disp[0])
        self.stream.append(ch1, ch2)
        res = self.stream.process()
        t_plot, d_plot = display_chunk(t_disp, np.stack([ch1, ch2], 1), self.stream.mode, self.stream.otsu)
        cands = []
        if len(res["peaks"]):
            t_start = time.time()
            probs = self.model.predict(res["X"], res["M"])
            log_ml.info("chunk %d: %d candidates scored in %.2f s (%d detected)", idx, len(res["peaks"]),
                        time.time() - t_start, int((probs >= self.cut_off).sum()))
            cands = self._record(res["peaks"], probs, res["raw"], res["edge"], res["mode"], "real-time")
        summary = self._write_summary(idx, res["mode"], res["otsu"])
        rt = getattr(self, "_rt_saved", None)              # --analysis real-time: the saved part's progress
        if rt and rt[1]:
            first, n_saved = rt
            if getattr(self, "_t_first", None) is None:     # start of the saved data
                self._t_first = time0
            done = idx - first + file_frac                     # files done (a sliced file: in part)
            if done >= n_saved:
                self._progress(None)
                self._rt_saved = None
            elif self._rt_ticks % 30 == 0:
                t_end = self._t_first + n_saved * file_n / SAMPLING_RATE
                self._progress(f"Analysing the saved data (real-time algorithm): at {self._hm(time0 + len(ch1) / SAMPLING_RATE)}"
                               f" of {self._hm(self._t_first)} - {self._hm(t_end)} h", done / n_saved)
            self._rt_ticks += 1
        PLOT_QUEUE.put({
            "kind"      : "chunk",
            "chunk"     : idx,
            "t_plot"    : t_plot,
            "d_plot"    : d_plot,
            "cands"     : cands,
            "final_t"   : self._time_of(res["stop"])[0] if res["stop"] > 0 else time0,
            "summary"   : summary,
        })


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE TIME AXIS
# ─────────────────────────────────────────────────────────────────────────────
class TimeAxis(pg.AxisItem):
    """
    Time axis (values in seconds) that shows s / min / h, chosen from the VISIBLE span:
      <= s_max → seconds (s_max: 1 min for the live trace, 5 min elsewhere),
      up to 1 hour → minutes,  > 1 hour → hours.
    The tick numbers and the axis label always use the same unit, and the ticks fall on
    round values of that unit (e.g. 880, 890, 900 min; 14.5, 15.0 h).
    clock=True: recording-clock labels h:mm:ss at any zoom (live trace, magnifier).
    """
    _unit = None                             # class default: AxisItem.__init__ already calls setRange
    CLOCK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200,
                   10800, 21600, 43200)

    def __init__(self, *args, whole_hours=False, s_max=300, clock=False, **kwargs):
        self._whole_hours = whole_hours      # hours as integers only (barcode): 0, 1, 2, ...
        self._s_max = s_max                  # largest visible span (s) still shown in seconds
        self._clock = clock
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)       # the unit is ours, not an SI prefix

    def _scale(self):
        span = abs(self.range[1] - self.range[0])
        return (3600.0, "h") if span > 3600 else (60.0, "min") if span > self._s_max + 1e-6 else (1.0, "s")

    def setRange(self, mn, mx):
        super().setRange(mn, mx)
        if self._scale() != self._unit:      # unit changed with the zoom → relabel
            self._unit = self._scale()
            self._updateLabel()

    def tickSpacing(self, minVal, maxVal, size):
        if self._clock:                      # one label per ~110 px, on clock-friendly steps
            target = abs(maxVal - minVal) / max(2.0, size / 110.0)
            return [(next((st for st in self.CLOCK_STEPS if st >= target), self.CLOCK_STEPS[-1]), 0.0)]
        u, unit = self._scale()              # round spacings in the display unit
        if self._whole_hours and unit == "h":
            span_h = abs(maxVal - minVal) / 3600
            step = 1 if span_h <= 24 else next((k for k in (2, 3, 6, 12, 24) if span_h / k <= 24), 48)
            return [(step * 3600.0, 0.0)]
        return [(sp * u, off * u) for sp, off in super().tickSpacing(minVal / u, maxVal / u, size)]

    @staticmethod
    def clock_string(v, dec=0):
        v = round(float(v), dec)
        sign, v = ("-" if v < 0 else ""), abs(v)
        h, m, sec = int(v // 3600), int(v % 3600 // 60), v % 60
        sec = f"{sec:0{3 + dec}.{dec}f}" if dec else f"{int(round(sec)):02d}"
        return f"{sign}{h}:{m:02d}:{sec}"

    def tickStrings(self, values, scale, spacing):
        if self._clock:
            dec = 0 if spacing >= 1 else 1
            return [self.clock_string(v, dec) for v in values]
        u, _ = self._scale()
        step = spacing / u
        dec = 0
        while dec < 6 and abs(step * 10 ** dec - round(step * 10 ** dec)) > 1e-6:
            dec += 1
        return [f"{v / u:.{dec}f}" for v in values]

    def labelString(self):
        unit = "h:mm:ss" if self._clock else self._scale()[1]
        return f"<span style='color:#444444'>Time ({unit})</span>"


# ─────────────────────────────────────────────────────────────────────────────
# HOVER REGION - highlight rectangle that shows tooltip on hover
# ─────────────────────────────────────────────────────────────────────────────
class HoverRegion(pg.LinearRegionItem):
    """A highlight rectangle on the signal plot that shows the candidate's
    sample and probability when hovered, and opens its clip when clicked."""
    def __init__(self, loc, prob, edge, tooltip_item, on_click=None, **kwargs):
        super().__init__(**kwargs)
        self._loc          = loc
        self._prob         = prob
        self._edge         = edge
        self._tooltip_item = tooltip_item
        self._on_click     = on_click
        self.setAcceptHoverEvents(True)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self._on_click:
            self._on_click(self._loc)
            ev.accept()

    def hoverEnterEvent(self, ev):
        sample = int(round(self._loc * SAMPLING_RATE))
        self._tooltip_item.setText(
            f"sample = {sample}\nprob = {self._prob:.3f}" + ("\nat laser edge" if self._edge else ""))
        self._tooltip_item.setPos(self._loc, TRACE_Y_MIN)
        self._tooltip_item.setVisible(True)
        ev.accept()

    def hoverLeaveEvent(self, ev):
        self._tooltip_item.setVisible(False)
        ev.accept()


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY BUFFER - preallocated, so an update never copies the whole hour
# ─────────────────────────────────────────────────────────────────────────────
class DisplayBuffer:
    """Time and per-channel arrays of the trace. New points are written in place
    (capacity doubles when needed); the oldest data is dropped in steps of
    TRIM_STEP_SECS, i.e. one copy every 10 min instead of one per update."""
    def __init__(self, n_ch=2, capacity=1 << 16):
        self._t = np.empty(capacity)                         # float64: times late in a 20-h run need it
        self._d = np.empty((n_ch, capacity), np.float32)     # mV values: float32 is plenty
        self.n = 0

    def append(self, t, d):                    # t (k,), d (k, n_ch)
        k = len(t)
        if self.n + k > len(self._t):
            cap = max(2 * len(self._t), self.n + k)
            t2, d2 = np.empty(cap), np.empty((self._d.shape[0], cap), np.float32)
            t2[:self.n], d2[:, :self.n] = self._t[:self.n], self._d[:, :self.n]
            self._t, self._d = t2, d2
        self._t[self.n:self.n + k] = t
        self._d[:, self.n:self.n + k] = d.T
        self.n += k

    def trim(self, history, step):
        if self.n and self._t[0] < self._t[self.n - 1] - history - step:
            i = int(np.searchsorted(self._t[:self.n], self._t[self.n - 1] - history))
            m = self.n - i
            self._t[:m] = self._t[i:self.n]            # numpy handles the overlapping copy
            self._d[:, :m] = self._d[:, i:self.n]
            self.n = m

    @property
    def t(self):
        return self._t[:self.n]

    def ch(self, k):
        return self._d[k, :self.n]


# ─────────────────────────────────────────────────────────────────────────────
# GUI  - main thread only, zero ML work
# ─────────────────────────────────────────────────────────────────────────────
class TDiFCWindow:
    CH_COLORS = ["#3fb950", "#e06c00", "#58a6ff", "#d29922"]

    # light "card" look: grey window, white titled cards, one button style
    APP_CSS = (
        "QWidget#root { background:#f2f3f5; }"
        " QFrame#card { background:#ffffff; border:1px solid #dcdfe4; border-radius:8px; }"
        " QFrame#vline { background:#dcdfe4; }"
        " QFrame#infobox { background:#f6f8fa; border:1px solid #e4e7eb; border-radius:6px; }"
        " QWidget#sidepage { background:#ffffff; }"
        " QLabel { color:#24292f; font-size:12px; background:transparent; }"
        " QLabel#cardtitle { font-size:13px; font-weight:600; }"
        " QLabel#muted { color:#6e7781; font-size:11px; }"
        " QLabel#section { color:#57606a; font-size:11px; font-weight:600; }"
        " QComboBox, QSpinBox { font-size:12px; color:#24292f; }"
        " QTabWidget::pane { border:1px solid #dcdfe4; border-radius:8px; background:#ffffff; top:-1px; }"
        " QTabBar::tab { background:#e9ebee; color:#57606a; border:1px solid #dcdfe4; border-bottom:none;"
        "  padding:6px 16px; margin-right:2px; border-top-left-radius:6px; border-top-right-radius:6px;"
        "  font-size:12px; }"
        " QTabBar::tab:selected { background:#ffffff; color:#24292f; font-weight:600; }"
        " QTabBar::tab:hover:!selected { background:#f3f4f6; }"
        " QScrollArea { background:transparent; border:none; }"
        " QSplitter::handle { background:transparent; }")
    BTN_STYLE = (
        "QPushButton { background:#ffffff; color:#24292f; border:1px solid #c9ced6; border-radius:5px;"
        " font-size:12px; padding:0 10px; }"
        " QPushButton:hover { background:#f3f4f6; border-color:#afb5bd; }"
        " QPushButton:pressed { background:#e9ebee; }"
        " QPushButton:checked { background:#dbe9fb; border:1px solid #5a8fd8; color:#0b3d91; }"
        " QPushButton:disabled { background:#f7f7f8; color:#a8adb4; border-color:#e1e4e8; }")

    def __init__(self, exp_name, out_dir, worker):
        self.exp_name = exp_name
        self.out_dir = out_dir
        self.worker = worker
        self.cut_off = PROB_THRESHOLD
        self._settings = QtCore.QSettings("t-DiFC", "tdifc_app")          # remembered between runs
        self._peak_color = QtGui.QColor(str(self._settings.value(
            "peak_color", self._settings.value("barcode_color", COLOR_PEAK))))
        if not self._peak_color.isValid():
            self._peak_color = QtGui.QColor(COLOR_PEAK)

        # ---- state
        self._cand_t: list = []        # every candidate, in time order (for barcode / highlight clicks)
        self._cands:  list = []
        self._summary = None
        self._final_t = 0.0            # recording time (s) analysed so far
        self._clr = None               # clearance tab widgets
        self._clr_key = None
        self._clr_warn_text = ""
        # operator review of the model's calls: results/<exp_name>/review.csv (append-only, never
        # wiped; verdicts are matched back to candidates by sample number after a restart)
        self._review_path = os.path.join(out_dir, "review.csv")
        self._reviews = self._load_reviews()     # sample -> 1 (peak) / 0 (artifact)
        self._current = None                      # candidate shown in the review tab
        self._curves = []
        self._clip_curves = []
        pts_per_s = (2 * SAMPLING_RATE / DISPLAY_BIN if DISPLAY_BIN > 2
                     else 0.3 * SAMPLING_RATE if DISPLAY_OFF_AS_ZERO else SAMPLING_RATE)   # pulsed: ~25% ON
        self._buf = DisplayBuffer(capacity=int((DISPLAY_HISTORY_SECS + TRIM_STEP_SECS + 60) * pts_per_s))
        self._highlights = {}          # candidate time -> (region, label); only the visible ones
        self._user_panned = False
        self._paused_since = None
        self._waiting = 0
        self._det_t = []               # times of detected peaks, sorted
        self._barcode_dirty = False
        self._t_max = 0.0
        self._barcode_span = 0.0       # barcode x-axis end (s), grows in steps - see _barcode_span_for

        header = self._build_header()
        signal_card = self._build_signal_card()
        barcode_card = self._build_barcode_card()
        self._build_magnifier()

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)
        lv.addWidget(signal_card, 1)
        lv.addWidget(barcode_card, 0)

        self._side_tabs = QtWidgets.QTabWidget()
        self._side_tabs.addTab(self._build_review_tab(), "Review calls")
        self._side_tabs.addTab(self._build_clearance_tab(), "Clearance")
        self._side_prev, self._side_w = 0, {}
        self._side_tabs.currentChanged.connect(self._side_tab_changed)

        self._split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._split.setHandleWidth(10)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(left)
        self._split.addWidget(self._side_tabs)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 0)

        container = QtWidgets.QWidget()
        container.setObjectName("root")
        container.setStyleSheet(self.APP_CSS)
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(12, 10, 12, 12)
        vlay.setSpacing(10)
        vlay.addWidget(header)
        vlay.addWidget(self._split, 1)
        _scr = QtWidgets.QApplication.primaryScreen()
        if _scr is None:
            _w, _h = WINDOW_SIZE
        else:
            g = _scr.availableGeometry()
            _w, _h = min(WINDOW_SIZE[0], g.width() - 40), min(WINDOW_SIZE[1], g.height() - 40)
        container.resize(_w, _h)
        QtCore.QTimer.singleShot(0, lambda: self._split.setSizes(            # once the window has its size
            [max(self._split.width() - SIDEBAR_WIDTH, 400), SIDEBAR_WIDTH]))
        container.setWindowTitle(f"t-DiFC Live - {exp_name}  [{getattr(worker, 'analysis', DEFAULT_ANALYSIS)}]")
        container.show()
        self._container = container
        for key, fn in (("Right", lambda: self._step(+1)), ("Left", lambda: self._step(-1)),
                        ("Home", lambda: self._go_end(first=True)), ("End", lambda: self._go_end(first=False)),
                        ("Y", lambda: self._review(1)), ("N", lambda: self._review(0))):
            QtGui.QShortcut(QtGui.QKeySequence(key), container).activated.connect(fn)
        self._update_review_panel()

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._drain)
        self._timer.start(500)

    # -------------------------------------------------------------- layout ---
    def _button(self, text, slot=None, tip="", checkable=False, extra="", height=26):
        b = QtWidgets.QPushButton(text)
        b.setFixedHeight(height)
        b.setStyleSheet(self.BTN_STYLE + extra)
        b.setCheckable(checkable)
        if tip:
            b.setToolTip(tip)
        if slot is not None:
            (b.toggled if checkable else b.clicked).connect(slot)
        return b

    @staticmethod
    def _card(title, *widgets, hint=""):
        """White titled card: title + muted hint on the left, widgets (buttons) on the right."""
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(6)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        t = QtWidgets.QLabel(title)
        t.setObjectName("cardtitle")
        head.addWidget(t)
        sub = QtWidgets.QLabel(hint)
        sub.setObjectName("muted")
        head.addWidget(sub)
        head.addStretch()
        for w in widgets:
            head.addWidget(w)
        lay.addLayout(head)
        return card, lay, sub

    @staticmethod
    def _section(title, hint="", *widgets):
        """Small upper-case section heading inside the sidebar."""
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 6, 0, 0)
        h.setSpacing(6)
        t = QtWidgets.QLabel(title.upper())
        t.setObjectName("section")
        h.addWidget(t)
        if hint:
            m = QtWidgets.QLabel(hint)
            m.setObjectName("muted")
            h.addWidget(m)
        h.addStretch()
        for x in widgets:
            h.addWidget(x)
        return w

    def _build_header(self):
        head = QtWidgets.QFrame()
        head.setObjectName("card")
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(14, 8, 14, 8)
        hl.setSpacing(16)
        ttl = QtWidgets.QLabel(f"<span style='font-size:15px; font-weight:600'>t-DiFC Live</span><br>"
                               f"<span style='color:#6e7781'>{self.exp_name} · "
                               f"{getattr(self.worker, 'analysis', DEFAULT_ANALYSIS)}</span>")
        ttl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        hl.addWidget(ttl)
        sep = QtWidgets.QFrame()
        sep.setObjectName("vline")
        sep.setFixedWidth(1)
        hl.addWidget(sep)
        mid = QtWidgets.QVBoxLayout()
        mid.setSpacing(4)
        self._status_lbl = QtWidgets.QLabel("<b>Waiting - loading model and looking for data…</b>")
        self._status_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        # progress bar of the analysis of the saved data (offline: one 2-h block at a time)
        self._status_bar = QtWidgets.QProgressBar()
        self._status_bar.setRange(0, 1000)
        self._status_bar.setTextVisible(False)
        self._status_bar.setFixedHeight(6)
        self._status_bar.setMaximumWidth(520)
        self._status_bar.setStyleSheet("QProgressBar { border:none; border-radius:3px; background:#e9ebee; }"
                                       " QProgressBar::chunk { background:#5a8fd8; border-radius:3px; }")
        self._status_bar.setVisible(False)
        mid.addWidget(self._status_lbl)
        mid.addWidget(self._status_bar)
        hl.addLayout(mid)
        hl.addStretch(1)
        self._pause_btn = self._button(
            "⏸  Pause", self._toggle_pause, checkable=True, height=30,
            tip="Stop reading new data (the DAQ keeps saving it); Resume catches up, nothing is lost",
            extra="QPushButton:checked { background:#fde2c8; border:1px solid #d9822b; color:#111111; }")
        self._pause_btn.setFixedWidth(110)
        hl.addWidget(self._pause_btn)
        return head

    def _build_signal_card(self):
        self._mag_btn = self._button("🔍  Magnifier", self._toggle_magnifier, checkable=True,
                                     tip="A 2-s window that follows the cursor, shown large beside the trace")
        self._mag_fixed_y = QtWidgets.QCheckBox("fixed y per pulse")
        self._mag_fixed_y.setToolTip("Magnifier y-axes: off = auto-scale to the window; on = fixed to the min / max "
                                     "of the laser-ON pulse under the window")
        self._mag_fixed_y.setEnabled(False)                    # only while the magnifier is on
        self._mag_fixed_y.toggled.connect(lambda *_: self._update_magnifier())
        self._reset_btn = self._button("↺  Reset view", self._reset_view,
                                       tip="Back to the newest data, and follow it live again")
        card, lay, _ = self._card("Signal", self._mag_btn, self._mag_fixed_y, self._reset_btn,
                                  hint=f"last {WINDOW_SECS} s live · drag to pan · scroll to zoom · "
                                       f"click a highlight to review it")
        self.win = pg.GraphicsLayoutWidget()
        self.win.setBackground("w")
        self.win.ci.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.win, 1)

        # signal trace (raw Ch1 / Ch2), recording clock h:mm:ss; magnifier beside it when on
        self._trace_row = self.win.addLayout(row=0, col=0)
        self.trace = self._trace_row.addPlot(row=0, col=0, axisItems={"bottom": TimeAxis(orientation="bottom", clock=True)})
        self.trace.setLabel("left", "Signal", units="mV")
        self.trace.setLabel("bottom", "Time")
        self.trace.getAxis("left").enableAutoSIPrefix(False)
        self.trace.showGrid(x=False, y=False)
        self.trace.setYRange(TRACE_Y_MIN, TRACE_Y_MAX, padding=0)
        # lock vertical axis: mouse pan/zoom only moves along time
        self.trace.setMouseEnabled(x=True, y=False)
        self.trace.enableAutoRange(axis="y", enable=False)
        self.trace.setLimits(yMin=TRACE_Y_MIN, yMax=TRACE_Y_MAX,
                             minYRange=TRACE_Y_MAX - TRACE_Y_MIN, minXRange=0.5)
        self.trace.setDownsampling(auto=True, mode="peak")
        self.trace.setClipToView(True)
        self.trace.addLegend(offset=(10, 10))
        self.trace.sigRangeChangedManually.connect(self._on_manual_pan)
        self.trace.autoBtn.clicked.connect(self._on_manual_pan)   # 'A' (show all) also stops auto-follow
        # highlights follow the view; re-synced 80 ms after the last pan / zoom step
        self._hl_timer = QtCore.QTimer()
        self._hl_timer.setSingleShot(True)
        self._hl_timer.timeout.connect(self._sync_highlights)
        self.trace.sigXRangeChanged.connect(lambda *_: self._hl_timer.start(80))
        self._tooltip = pg.TextItem(text="", color="#ffffff", fill=pg.mkBrush(30, 30, 30, 200),
                                    anchor=(0.0, 1.0))
        self._tooltip.setVisible(False)
        self._tooltip.setZValue(100)
        # the newest data not analysed yet (continuous: the last ~3 s; pulsed: the laser-ON pulse
        # still going on): its candidates and highlights appear when the analysis reaches it
        self._pending = pg.LinearRegionItem(values=(0, 0), orientation="vertical", movable=False,
                                            brush=pg.mkBrush(120, 120, 120, 28), pen=pg.mkPen(None))
        self._pending.setAcceptHoverEvents(False)
        self._pending.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._pending.setZValue(-20)
        self._pending.setVisible(False)
        self._pending_txt = pg.TextItem(text="analysing…", color="#6e7781", anchor=(1.0, 0.0))
        self._pending_txt.setVisible(False)
        self.trace.addItem(self._pending, ignoreBounds=True)
        self.trace.addItem(self._pending_txt, ignoreBounds=True)
        self.trace.addItem(self._tooltip)
        self.win.scene().sigMouseMoved.connect(self._trace_mouse_moved)
        self.win.scene().sigMouseClicked.connect(self._trace_clicked)   # zoomed out: zoom to a pulse
        return card

    def _build_barcode_card(self):
        self._barcode_color_btn = QtWidgets.QPushButton()
        self._barcode_color_btn.setFixedSize(26, 26)
        self._barcode_color_btn.setToolTip("CTC colour: barcode lines, highlights on the signal trace and the "
                                           "review calls panel")
        self._barcode_color_btn.clicked.connect(self._pick_barcode_color)
        self._style_color_btn()
        self._save_barcode_btn = self._button("💾  Save PNG", self._save_barcode, tip="Save the barcode as PNG")
        colour_lbl = QtWidgets.QLabel("CTC colour")
        colour_lbl.setObjectName("muted")
        card, lay, self._bc_hint = self._card("Detected CTCs", colour_lbl, self._barcode_color_btn,
                                              self._save_barcode_btn, hint="full experiment · click a line to review it")
        self.bc_win = pg.GraphicsLayoutWidget()
        self.bc_win.setBackground("w")
        self.bc_win.ci.setContentsMargins(6, 0, 14, 0)       # room for the end labels (0, last hour)
        self.bc_win.setFixedHeight(140)
        lay.addWidget(self.bc_win)
        self.barcode = self.bc_win.addPlot(row=0, col=0,
                                           axisItems={"bottom": TimeAxis(orientation="bottom", whole_hours=True)})
        self.barcode.setLabel("bottom", "Time")          # TimeAxis adds the unit: Time (h)
        self.barcode.showGrid(x=False, y=False)
        self.barcode.getAxis("left").hide()
        # the axis ends on a whole hour (see _barcode_span_for): let the end labels (0, last hour)
        # flow up to 20 px past the axis ends instead of being hidden
        self.barcode.getAxis("bottom").setStyle(hideOverlappingLabels=20)
        self.barcode.setYRange(0, 1, padding=0)
        self.barcode.setMouseEnabled(x=False, y=False)
        self.barcode.setMenuEnabled(False)
        self.barcode.hideButtons()
        # all detected peaks as ONE item (vertical segments), not one object per peak
        # not antialiased: a smoothed 2-px pen takes Qt's slow path (~0.5 s per repaint at 2,000 CTCs,
        # ~5 s at 20,000, freezing the window); plain vertical lines look the same and take ms
        self._barcode_curve = pg.PlotCurveItem(pen=pg.mkPen(self._peak_color, width=2), connect="pairs",
                                               antialias=False)
        self.barcode.addItem(self._barcode_curve)
        self._barcode_hover = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen(self._peak_color.lighter(125), width=3))
        self._barcode_hover.setVisible(False)
        self.barcode.addItem(self._barcode_hover)
        self.bc_win.scene().sigMouseMoved.connect(self._barcode_mouse_moved)
        self.bc_win.scene().sigMouseClicked.connect(self._barcode_clicked)
        return card

    def _side_tab_changed(self, i):
        """Each sidebar tab keeps its own width: Review narrow, Clearance wide (drag to change)."""
        sizes = self._split.sizes()
        total = sum(sizes)
        self._side_w[self._side_prev] = sizes[1]
        w = self._side_w.get(i) or (SIDEBAR_WIDTH if i == 0 else int(total * CLEARANCE_SHARE))
        self._split.setSizes([total - w, w])
        self._side_prev = i
        if i == 1:
            self._clr_key = None
            self._update_clearance()

    # ------------------------------------------------------------- helpers ---
    @staticmethod
    def _fmt_hms(t: float) -> str:
        t = int(t)
        return f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}"

    def _set_status(self, html: str):
        self._status_lbl.setText(html)

    def _nearest_candidate(self, t, max_dt=0.5, detected_only=False):
        if not self._cand_t:
            return None
        i = bisect.bisect_left(self._cand_t, t)
        best = None
        for j in range(max(0, i - 3), min(len(self._cand_t), i + 3)):
            c = self._cands[j]
            if detected_only and c["prob"] < self.cut_off:
                continue
            if abs(c["t"] - t) <= max_dt and (best is None or abs(c["t"] - t) < abs(best["t"] - t)):
                best = c
        return best

    def _barcode_peak_at(self, scene_pos, tol_px=4):
        """Detected peak under the mouse (within tol_px), or None."""
        vb = self.barcode.getViewBox()
        if not self._det_t or not vb.sceneBoundingRect().contains(scene_pos):
            return None
        x = vb.mapSceneToView(scene_pos).x()
        tol = tol_px * vb.viewPixelSize()[0]
        i = bisect.bisect_left(self._det_t, x)
        near = [self._det_t[j] for j in (i - 1, i) if 0 <= j < len(self._det_t)]
        best = min(near, key=lambda t: abs(t - x)) if near else None
        return best if best is not None and abs(best - x) <= tol else None

    def _trace_mouse_moved(self, scene_pos):
        dragging = (bool(getattr(self.win.scene(), "clickEvents", None))       # button down (pyqtgraph)
                    or QtWidgets.QApplication.mouseButtons() != QtCore.Qt.MouseButton.NoButton)
        vb = self.trace.getViewBox()
        on_trace = vb.sceneBoundingRect().contains(scene_pos)
        if self._mag_region is not None and not self._zoomed_out() and not dragging and on_trace:
            # (while dragging, the trace pans with the cursor: the window stays on the same
            # signal under the cursor by itself; moving it here would use the not-yet-panned view)
            self._place_magnifier(vb.mapSceneToView(scene_pos).x())   # magnifier follows the cursor
        if self._zoomed_out() and on_trace:
            self.win.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)     # a pulse can be clicked
        else:
            self.win.unsetCursor()

    def _barcode_mouse_moved(self, scene_pos):
        t = self._barcode_peak_at(scene_pos)
        self._barcode_hover.setVisible(t is not None)
        if t is not None:
            self._barcode_hover.setValue(t)
            self.bc_win.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.bc_win.unsetCursor()

    def _zoomed_out(self):
        (x0, x1), _ = self.trace.getViewBox().viewRange()
        return x1 - x0 > HIGHLIGHT_MAX_SPAN

    def _trace_clicked(self, ev):
        """Zoomed-out trace (no highlights drawn): a click on a laser-ON pulse zooms to that pulse
        (+/- PULSE_ZOOM_MARGIN s), where its candidates are highlighted. Continuous laser: +/- 30 s."""
        vb = self.trace.getViewBox()
        if ev.isAccepted() or ev.button() != QtCore.Qt.MouseButton.LeftButton \
                or not vb.sceneBoundingRect().contains(ev.scenePos()) or self._buf.n == 0:
            return                                             # e.g. a highlight took the click
        x = vb.mapSceneToView(ev.scenePos()).x()
        if not self._zoomed_out():
            if self._mag_region is not None:                   # magnifier: the trace stops scrolling here
                self._user_panned = True                       # (Reset view: live again); the window keeps
                self._place_magnifier(x)                       # following the cursor on the stopped trace
                ev.accept()
            return
        s = self._summary or {}
        view = (x - WINDOW_SECS / 2.0, x + WINDOW_SECS / 2.0)
        if s.get("laser_mode") == "pulsed" and s.get("otsu_threshold") is not None:
            pulse = self._pulse_at(x, max(6 * vb.viewPixelSize()[0], 1.0))   # a few pixels around the click
            if pulse is None:
                return                                         # clicked a laser-OFF gap
            i0, i1 = pulse
            view = (self._buf.t[i0] - PULSE_ZOOM_MARGIN, self._buf.t[i1] + PULSE_ZOOM_MARGIN)
        self._user_panned = True
        self.trace.setXRange(*view, padding=0)
        ev.accept()

    def _pulse_at(self, x, tol):
        """Display-buffer indices (first, last) of the laser-ON pulse nearest to time x within
        +/- tol s, or None. A pulse = Ch1 above the stream's Otsu threshold for >= 0.5 s, with
        dips shorter than 50 ms (e.g. on the rising edge) bridged. Pulsed laser only."""
        s = self._summary or {}
        thr = s.get("otsu_threshold")
        if s.get("laser_mode") != "pulsed" or thr is None or self._buf.n == 0:
            return None
        tb = self._buf.t
        lo, hi = np.searchsorted(tb, [x - tol - 60, x + tol + 60])    # a pulse lies within +/- 60 s
        if hi - lo < 2:
            return None
        seg_t = tb[lo:hi]
        on = (self._buf.ch(0)[lo:hi] > thr).astype(np.int8)
        d = np.diff(np.concatenate([[0], on, [0]]))
        runs = []
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
            if runs and seg_t[a] - seg_t[runs[-1][1]] < 0.05:
                runs[-1][1] = b                                 # bridge a short dip
            else:
                runs.append([a, b])
        best, best_d = None, None
        for a, b in runs:
            if seg_t[b] - seg_t[a] < 0.5:
                continue                                        # a blip, not a pulse
            dist = 0.0 if seg_t[a] <= x <= seg_t[b] else min(abs(x - seg_t[a]), abs(x - seg_t[b]))
            if dist <= tol and (best_d is None or dist < best_d):
                best, best_d = (int(lo + a), int(lo + b)), dist
        return best

    def _barcode_clicked(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        t = self._barcode_peak_at(ev.scenePos())
        if t is not None:
            self._jump_to_time(t)
            ev.accept()

    # -------------------------------------------------------------- actions ---
    def _jump_to_time(self, t: float):
        """Barcode line or highlight clicked: pan the signal plot there and show the
        candidate's 1 s window (background-subtracted, full resolution) below."""
        c = self._nearest_candidate(t)
        if c is None:
            self._user_panned = True
            self.trace.setXRange(t - WINDOW_SECS / 2.0, t + WINDOW_SECS / 2.0, padding=0)
            log_app.warning("No candidate found at t=%.2f s", t)
            return
        self._show_candidate(c)

    def _show_candidate(self, c):
        """Pan the signal plot to candidate c and show its 1 s window and details."""
        self._user_panned = True
        self.trace.setXRange(c["t"] - WINDOW_SECS / 2.0, c["t"] + WINDOW_SECS / 2.0, padding=0)
        self._current = c
        if self._mag_region is not None:                      # magnifier on the candidate too
            self._place_magnifier(c["t"])
        clip_d = c["clip_d"]
        clip_t = c["t"] + (np.arange(len(clip_d)) - len(clip_d) // 2) / SAMPLING_RATE
        n_ch = clip_d.shape[1]
        if len(self._clip_curves) != n_ch:
            for crv in self._clip_curves:
                self.clip_plot.removeItem(crv)
            self._clip_curves = [None] * n_ch
            for ch in range(n_ch - 1, -1, -1):     # Ch2 first (bottom), Ch1 on top
                self._clip_curves[ch] = self.clip_plot.plot(
                    pen=pg.mkPen(self.CH_COLORS[ch % len(self.CH_COLORS)], width=1), name=f"Ch{ch + 1}")
        for ch, crv in enumerate(self._clip_curves):
            crv.setData(clip_t, clip_d[:, ch])
        self.clip_plot.setXRange(float(clip_t[0]), float(clip_t[-1]), padding=0.0)
        self.clip_plot.enableAutoRange(axis="y", enable=True)
        self.clip_plot.getAxis("bottom").setTicks([[(c["t"], f"{c['sample']}")], []])
        self._update_clip_info()
        self._clip_stack.setCurrentWidget(self._clip_win)
        self._clip_info_box.setVisible(True)
        self._save_clip_btn.setEnabled(True)
        if self._side_tabs.currentIndex() != 0:
            self._side_tabs.setCurrentIndex(0)                # show it in the Review tab
        self._update_review_panel()

    def _update_clip_info(self):
        c = self._current
        if c is None:
            return
        pk = self._peak_color.name()
        verdict = (f"<span style='color:{pk}'>detected peak</span>" if c["prob"] >= self.cut_off
                   else f"<span style='color:{COLOR_ARTIFACT}'>rejected (artifact)</span>")
        op = {1: f"<span style='color:{pk}'><b>CTC ✓</b></span>",
              0: f"<span style='color:{COLOR_ARTIFACT}'><b>artifact ✗</b></span>"}.get(self._reviews.get(c["sample"]), "-")
        rows = (("time", self._fmt_hms(c["t"])), ("sample", c["sample"]),
                ("prob", f"{c['prob']:.3f} - {verdict}"), ("operator", op),
                ("laser edge", "yes" if c["edge"] else "no"),
                ("file", f"<span style='font-size:11px'>{c['file']}</span>"))
        self._clip_info_label.setText(
            "<table cellspacing='0' cellpadding='2'>"
            + "".join(f"<tr><td style='color:#57606a'>{k}</td><td>&nbsp;&nbsp;{v}</td></tr>" for k, v in rows)
            + "</table>")

    # ------------------------------------------------------------ magnifier ---
    def _build_magnifier(self):
        """Ch1 and Ch2 of the magnifier window, beside the trace (same row) while the magnifier is on.
        Each y-axis is fixed to the min / max of the laser-ON pulse under the window."""
        self._mag_region = None
        self._mag_marks = []
        self._mag_ylim_key = None
        self._mag_layout = pg.GraphicsLayout()
        self._mag_layout.setMinimumWidth(300)
        self._mag_plots, self._mag_curves = [], []
        for k in range(2):
            p = self._mag_layout.addPlot(row=k, col=0, axisItems={"bottom": TimeAxis(orientation="bottom", clock=True)})
            p.setLabel("left", f"Ch{k + 1}", units="mV")
            p.getAxis("left").setWidth(52)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            if k == 0:
                p.hideAxis("bottom")                            # one time axis, under Ch2
            else:
                p.setXLink(self._mag_plots[0])
            self._mag_plots.append(p)
            self._mag_curves.append(p.plot(pen=pg.mkPen(self.CH_COLORS[k], width=1), antialias=False))

    def _toggle_magnifier(self, on):
        if on:
            vb = self.trace.getViewBox()
            (x0, x1), _ = vb.viewRange()
            c = (x0 + x1) / 2
            cur = self.win.mapToScene(self.win.viewport().mapFromGlobal(QtGui.QCursor.pos()))
            if vb.sceneBoundingRect().contains(cur):           # start where the cursor is on the signal
                c = vb.mapSceneToView(cur).x()
            if x1 - x0 > 5 * WINDOW_SECS:                     # zoomed out: show 60 s so the window is grabbable
                self._user_panned = True
                self.trace.setXRange(c - WINDOW_SECS / 2, c + WINDOW_SECS / 2, padding=0)
            h = MAGNIFIER_SECS / 2
            # not movable: it follows the cursor, and a press-and-drag on it must pan the trace
            # (a movable region would take the drag, so the trace would not move)
            self._mag_region = pg.LinearRegionItem(values=(c - h, c + h), orientation="vertical", movable=False,
                                                   brush=pg.mkBrush(88, 166, 255, 45),
                                                   pen=pg.mkPen((60, 120, 220), width=1))
            self._mag_region.setZValue(-1)                    # below curves and candidate highlights
            self.trace.addItem(self._mag_region)
            self._mag_region.sigRegionChanged.connect(self._update_magnifier)
            self._trace_row.addItem(self._mag_layout, row=0, col=1)   # beside the trace; trace narrows
            self._trace_row.layout.setColumnStretchFactor(0, MAGNIFIER_STRETCH[0])
            self._trace_row.layout.setColumnStretchFactor(1, MAGNIFIER_STRETCH[1])
            self._mag_fixed_y.setEnabled(True)
            self._mag_ylim_key = None
            self._style_magnifier()
            self._update_magnifier()
        else:
            if self._mag_region is not None:
                self.trace.removeItem(self._mag_region)
                self._mag_region = None
                self._trace_row.removeItem(self._mag_layout)
                self._trace_row.layout.setColumnStretchFactor(1, 0)
            self._mag_fixed_y.setEnabled(False)

    def _place_magnifier(self, x):
        h = MAGNIFIER_SECS / 2
        self._mag_region.setRegion((x - h, x + h))

    def _style_magnifier(self):
        for ln in self._mag_region.lines:
            ln.setPen(pg.mkPen((60, 120, 220), width=1.5))
        self._mag_btn.setToolTip("Magnifier on: the 2-s window follows the cursor on the trace. Click the trace "
                                 "to stop its live scrolling (Reset view: live again). Press this button to close.")

    def _update_magnifier(self, *_):
        """Ch1 / Ch2 (mV) inside the magnifier window, full resolution from the display buffer,
        with the candidates in it marked (red: detected, grey: rejected)."""
        if self._mag_region is None:
            return
        lo, hi = self._mag_region.getRegion()
        tb = self._buf.t
        a, b = np.searchsorted(tb, [lo, hi])
        a, b = max(a - 1, 0), min(b + 1, len(tb))
        for k, crv in enumerate(self._mag_curves):
            crv.setData(tb[a:b], self._buf.ch(k)[a:b], skipFiniteCheck=True)
        for p in self._mag_plots:
            p.setXRange(lo, hi, padding=0)
        if self._mag_fixed_y.isChecked():
            self._set_magnifier_ylim(lo, hi)                    # fixed to the pulse under the window
        else:
            for p in self._mag_plots:                           # auto-scale to the window
                p.enableAutoRange(axis="y", enable=True)
            self._mag_ylim_key = None
        for p, item in self._mag_marks:
            p.removeItem(item)
        self._mag_marks = []
        i0, i1 = bisect.bisect_left(self._cand_t, lo), bisect.bisect_right(self._cand_t, hi)
        for c in self._cands[i0:i1]:
            det = c["prob"] >= self.cut_off
            col = self._peak_color if det else QtGui.QColor(COLOR_ARTIFACT)
            for k, p in enumerate(self._mag_plots):
                ln = pg.InfiniteLine(pos=c["t"], angle=90, movable=False,
                                     pen=pg.mkPen(col, width=1, style=QtCore.Qt.PenStyle.DashLine),
                                     label=f"{c['prob']:.2f}" if k == 0 else None,
                                     labelOpts={"position": 0.92, "color": col, "movable": False})
                p.addItem(ln, ignoreBounds=True)
                self._mag_marks.append((p, ln))

    def _set_magnifier_ylim(self, lo, hi):
        """y-limits of Ch1 / Ch2 = min / max of that channel over the laser-ON pulse under the window
        (only samples with Ch1 above the laser threshold, + 5% margin), so they stay fixed while the
        window moves inside one pulse. Continuous laser: the 60-s block of the window centre.
        No pulse under the window (laser-OFF gap): automatic."""
        c = (lo + hi) / 2
        tb = self._buf.t
        pulse = self._pulse_at(c, (hi - lo) / 2)
        if pulse is not None:
            key = ("pulse", float(tb[pulse[0]]))
            # the pulse without its first / last 25 ms (laser switching transients; as the pipeline's
            # laser-edge flag), only samples above the laser threshold
            a, b = np.searchsorted(tb, [tb[pulse[0]] + 0.025, tb[pulse[1]] - 0.025])
            if b - a < 10:
                a, b = pulse[0], pulse[1] + 1
            sel = slice(a, b)
            on = self._buf.ch(0)[sel] > self._summary["otsu_threshold"]
        elif (self._summary or {}).get("laser_mode") == "continuous" and self._buf.n:
            b0 = np.floor(c / 60) * 60
            key = ("block", b0)
            a, b = np.searchsorted(tb, [b0, b0 + 60])
            sel, on = slice(a, b), slice(None)
        else:
            if self._mag_ylim_key is not None:
                for p in self._mag_plots:
                    p.enableAutoRange(axis="y", enable=True)
                self._mag_ylim_key = None
            return
        if key == self._mag_ylim_key:
            return                                             # same pulse: limits unchanged
        for k, p in enumerate(self._mag_plots):
            v = self._buf.ch(k)[sel][on]
            if len(v) == 0:
                continue
            y0, y1 = float(v.min()), float(v.max())
            m = max(0.05 * (y1 - y0), 1.0)
            p.setYRange(y0 - m, y1 + m, padding=0)              # also switches y auto-range off
        self._mag_ylim_key = key

    # --------------------------------------------------------------- review ---
    REVIEW_SETS = ["Detected peaks", "Near misses (0.5 ≤ p < cut-off)", "All rejected", "All candidates",
                   "Reviewed (change a verdict)"]
    REVIEWED_SET = 4

    def _build_review_tab(self):
        page = QtWidgets.QWidget()
        page.setObjectName("sidepage")
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 8, 12, 12)
        lay.setSpacing(7)

        # which calls
        lay.addWidget(self._section("Calls to review", "keys: ← → Home End, Y / N"))
        self._review_set = QtWidgets.QComboBox()
        self._review_set.addItems(self.REVIEW_SETS)
        self._review_set.currentIndexChanged.connect(lambda *_: self._update_review_panel())
        lay.addWidget(self._review_set)
        self._skip_reviewed = QtWidgets.QCheckBox("skip reviewed")
        self._skip_reviewed.setChecked(True)
        self._skip_reviewed.toggled.connect(lambda *_: self._update_review_panel())
        self._auto_next = QtWidgets.QCheckBox("next after verdict")
        self._auto_next.setChecked(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._skip_reviewed)
        row.addWidget(self._auto_next)
        row.addStretch()
        lay.addLayout(row)
        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(4)
        for text, fn in (("⏮ First", lambda: self._go_end(first=True)), ("◀ Prev", lambda: self._step(-1)),
                         ("Next ▶", lambda: self._step(+1)), ("Latest ⏭", lambda: self._go_end(first=False))):
            nav.addWidget(self._button(text, lambda _=False, fn=fn: fn()))
        lay.addLayout(nav)
        self._review_pos = QtWidgets.QLabel("")
        lay.addWidget(self._review_pos)

        # the candidate: 1-s window + details
        self._save_clip_btn = self._button("💾  Save clip", self._save_clip, height=22,
                                           tip="Save the 1-s window as PNG")
        self._save_clip_btn.setEnabled(False)
        lay.addWidget(self._section("Candidate", "1-s window, background removed", self._save_clip_btn))
        self._clip_stack = QtWidgets.QStackedWidget()
        self._clip_stack.setFixedHeight(230)
        ph = QtWidgets.QLabel("Click a highlight on the signal or a line on the barcode,\n"
                              "or press Next ▶ to step through the calls.")
        ph.setObjectName("muted")
        ph.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("QLabel { border:1px dashed #c9ced6; border-radius:6px; }")
        self._clip_stack.addWidget(ph)
        self._clip_win = pg.GraphicsLayoutWidget()
        self._clip_win.setBackground("w")
        self._clip_win.ci.setContentsMargins(0, 0, 0, 0)
        self.clip_plot = self._clip_win.addPlot(row=0, col=0, axisItems={"bottom": TimeAxis(orientation="bottom")})
        self.clip_plot.setLabel("left", "Signal - background (mV)")
        self.clip_plot.getAxis("left").enableAutoSIPrefix(False)
        self.clip_plot.showGrid(x=False, y=False)
        self.clip_plot.addLegend(offset=(5, 5))
        self._clip_stack.addWidget(self._clip_win)
        lay.addWidget(self._clip_stack)
        info = QtWidgets.QFrame()
        info.setObjectName("infobox")
        il = QtWidgets.QVBoxLayout(info)
        il.setContentsMargins(8, 4, 8, 4)
        self._clip_info_label = QtWidgets.QLabel("")
        self._clip_info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._clip_info_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        il.addWidget(self._clip_info_label)
        self._clip_info_box = info
        info.setVisible(False)
        lay.addWidget(info)

        # verdict
        lay.addWidget(self._section("Verdict"))
        ver = QtWidgets.QHBoxLayout()
        self._btn_peak = QtWidgets.QPushButton("✓  CTC  (Y)")
        self._btn_art = QtWidgets.QPushButton("✗  Artifact  (N)")
        for b, lab in ((self._btn_peak, 1), (self._btn_art, 0)):
            b.setFixedHeight(32)
            b.clicked.connect(lambda _=False, lab=lab: self._review(lab))
            ver.addWidget(b)
        lay.addLayout(ver)
        hint = QtWidgets.QLabel("To change a verdict, press the other button. The same button again clears it.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # progress
        lay.addWidget(self._section("Review progress"))
        self._review_stats = QtWidgets.QLabel("")
        lay.addWidget(self._review_stats)
        lay.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _load_reviews(self):
        """Last verdict per sample from review.csv (empty label = verdict cleared)."""
        out = {}
        if os.path.exists(self._review_path):
            try:
                with open(self._review_path, newline="") as f:
                    for r in csv.DictReader(f):
                        lab = {"peak": 1, "artifact": 0}.get(r.get("operator_label", ""))
                        if lab is None:
                            out.pop(int(r["sample"]), None)
                        else:
                            out[int(r["sample"])] = lab
                log_app.info("review: %d earlier verdicts loaded from %s", len(out), self._review_path)
            except Exception as e:
                log_app.error("could not read %s: %s", self._review_path, e)
        return out

    def _set_members(self):
        """All candidates of the selected set, in time order."""
        k = self._review_set.currentIndex()
        cut = self.cut_off
        keep = [lambda c: c["prob"] >= cut, lambda c: 0.5 <= c["prob"] < cut, lambda c: c["prob"] < cut,
                lambda c: True, lambda c: c["sample"] in self._reviews][k]
        return [c for c in self._cands if keep(c)]

    def _review_items(self):
        """What Prev / Next / First / Latest move through: the selected set, without the reviewed
        candidates if 'skip reviewed' (except the one on screen; not for the 'Reviewed' set)."""
        skip = self._skip_reviewed.isChecked() and self._review_set.currentIndex() != self.REVIEWED_SET
        return [c for c in self._set_members()
                if not (skip and c["sample"] in self._reviews and c is not self._current)]

    def _go_end(self, first):
        skip = self._skip_reviewed.isChecked() and self._review_set.currentIndex() != self.REVIEWED_SET
        items = [c for c in self._set_members() if not (skip and c["sample"] in self._reviews)]
        if not items:
            self._review_pos.setText("nothing to show in this set")
            return
        self._show_candidate(items[0] if first else items[-1])

    def _step(self, d):
        items = self._review_items()
        if not items:
            self._review_pos.setText("nothing to show in this set")
            return
        if self._current is None:
            c = items[0] if d > 0 else items[-1]
        else:
            times = [x["t"] for x in items]
            i = (bisect.bisect_right(times, self._current["t"]) if d > 0
                 else bisect.bisect_left(times, self._current["t"]) - 1)
            if not 0 <= i < len(items):
                self._update_review_panel()
                self._review_pos.setText(self._review_pos.text() + ("   (last one)" if d > 0 else "   (first one)"))
                return
            c = items[i]
        self._show_candidate(c)

    def _review(self, label):
        """Record the operator's verdict for the candidate on screen (same verdict again = clear)."""
        c = self._current
        if c is None:
            return
        new = None if self._reviews.get(c["sample"]) == label else label
        if new is None:
            self._reviews.pop(c["sample"], None)
        else:
            self._reviews[c["sample"]] = new
        cols = ["sample", "time_s", "time_h", "source_file", f"p_{paths.MODEL_NAME}", "model_call",
                "operator_label", "reviewed_at"]
        try:
            first = not os.path.exists(self._review_path)
            with open(self._review_path, "a", newline="") as f:
                w = csv.writer(f)
                if first:
                    w.writerow(cols)
                w.writerow([c["sample"], f"{c['t']:.4f}", f"{c['t'] / 3600:.6f}", c["file"], f"{c['prob']:.6f}",
                            "peak" if c["prob"] >= self.cut_off else "artifact",
                            {1: "peak", 0: "artifact", None: ""}[new], time.strftime("%Y-%m-%d %H:%M:%S")])
        except Exception as e:
            log_app.error("could not write %s: %s", self._review_path, e)
        self._update_clip_info()
        self._update_review_panel()
        if new is not None and self._auto_next.isChecked():
            self._step(+1)

    def _update_review_panel(self):
        members = self._set_members()
        cur = self._current
        pos = next((i for i, c in enumerate(members) if c is cur), None)
        left = sum(c["sample"] not in self._reviews for c in members)
        where = f"<b>{pos + 1}</b> / {len(members)}" if pos is not None else f"- / {len(members)}"
        tail = "" if self._review_set.currentIndex() == self.REVIEWED_SET else f"  ·  {left} left to review"
        self._review_pos.setText(f"{where} in this set{tail}")
        lab = self._reviews.get(cur["sample"]) if cur is not None else None
        on = "QPushButton { background:%s; color:#ffffff; border:1px solid %s; border-radius:5px; font-size:12px; }"
        off = self.BTN_STYLE
        self._btn_peak.setStyleSheet(on % ((self._peak_color.name(),) * 2) if lab == 1 else off)
        self._btn_art.setStyleSheet(on % (COLOR_ARTIFACT, COLOR_ARTIFACT) if lab == 0 else off)
        self._btn_peak.setEnabled(cur is not None)
        self._btn_art.setEnabled(cur is not None)
        cut = self.cut_off
        groups = (("Detected", lambda c: c["prob"] >= cut), ("Near misses", lambda c: 0.5 <= c["prob"] < cut),
                  ("Other rejected", lambda c: c["prob"] < 0.5))
        lines = []
        for name, keep in groups:
            g = [c for c in self._cands if keep(c)]
            r = [self._reviews[c["sample"]] for c in g if c["sample"] in self._reviews]
            if name == "Detected":
                prec = f"<br>&nbsp;&nbsp;confirmed as peaks: {sum(r)} ({100 * sum(r) / len(r):.0f}%)" if r else ""
                lines.append(f"<b>{name}</b>: {len(r)} of {len(g)} reviewed{prec}")
            else:
                lines.append(f"<b>{name}</b>: {len(r)} of {len(g)} reviewed"
                             + (f"<br>&nbsp;&nbsp;judged to be peaks: {sum(r)}" if r else ""))
        self._review_stats.setText("<br>".join(lines))

    def _style_color_btn(self):
        c = self._peak_color.name()
        self._barcode_color_btn.setStyleSheet(
            f"QPushButton {{ background:{c}; border:1px solid #888888; border-radius:4px; }}"
            f" QPushButton:hover {{ border:2px solid #444444; }}")

    def _pick_barcode_color(self):
        """One CTC colour for the barcode, the detected-peak highlights + labels on the trace, the
        magnifier marks and the review calls panel / clip info."""
        col = QtWidgets.QColorDialog.getColor(self._peak_color, self._container, "CTC colour")
        if not col.isValid():                       # dialog cancelled
            return
        self._peak_color = col
        self._settings.setValue("peak_color", col.name())
        self._barcode_curve.setPen(pg.mkPen(col, width=2))
        self._barcode_hover.setPen(pg.mkPen(col.lighter(125), width=3))
        self._style_color_btn()
        for rgn, txt in self._highlights.values():  # redraw the visible highlights in the new colour
            self.trace.removeItem(rgn)
            self.trace.removeItem(txt)
        self._highlights.clear()
        self._sync_highlights()
        self._update_magnifier()
        self._update_review_panel()
        self._update_clip_info()

    def _save_barcode(self):
        """Save barcode as PNG by rendering it into a dedicated offscreen window."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save barcode", os.path.join(self.out_dir, "barcode.png"), "PNG (*.png)")
        if not path:
            return
        try:
            tmp = pg.GraphicsLayoutWidget()
            tmp.setBackground("w")
            tmp.resize(1600, 120)
            p = tmp.addPlot(row=0, col=0, axisItems={"bottom": TimeAxis(orientation="bottom", whole_hours=True)})
            p.getAxis("left").hide()
            p.getAxis("bottom").setStyle(hideOverlappingLabels=20)
            p.setYRange(0, 1, padding=0)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            x, y = self._barcode_xy()
            p.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen(self._peak_color, width=1.5), connect="pairs",
                                       antialias=False))
            if self._t_max > 0:
                p.setXRange(0, self._barcode_span_for(self._t_max), padding=0)
            p.setLabel("bottom", "Time (h)" if self._t_max > 3600 else "Time (min)" if self._t_max > 300
                       else "Time (s)")
            tmp.show()
            tmp.raise_()
            QtWidgets.QApplication.processEvents()
            tmp.grab().save(path)
            tmp.close()
            log_app.info("Barcode saved to %s", path)
        except Exception as e:
            log_app.error("Save barcode failed: %s", e)

    # ----------------------------------------------------------- clearance ---
    def _build_clearance_tab(self):
        page = QtWidgets.QWidget()
        page.setObjectName("sidepage")
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        ctl = QtWidgets.QHBoxLayout()
        ctl.setSpacing(6)
        self._clr_bin = QtWidgets.QSpinBox()
        self._clr_bin.setRange(1, 240)
        self._clr_bin.setValue(CLEARANCE_BIN_MIN)
        self._clr_bin.setSuffix(" min")
        self._clr_bin.setToolTip("Detected CTCs are counted in bins of this length")
        self._clr_skip = QtWidgets.QSpinBox()
        self._clr_skip.setRange(0, 100000)
        self._clr_skip.setValue(CLEARANCE_SKIP_MIN)
        self._clr_skip.setSuffix(" min")
        self._clr_skip.setToolTip("Bins in the first X min are plotted but left out of the exponential fit")
        for sb in (self._clr_bin, self._clr_skip):
            sb.setFixedWidth(90)
            sb.valueChanged.connect(lambda *_: self._update_clearance())
        ctl.addWidget(QtWidgets.QLabel("Bin size"))
        ctl.addWidget(self._clr_bin)
        ctl.addSpacing(10)
        ctl.addWidget(QtWidgets.QLabel("Skip first"))
        ctl.addWidget(self._clr_skip)
        ctl.addStretch()
        ctl.addWidget(self._button("💾  Save PNG", self._save_clearance, tip="Save the plot and the fit results as PNG"))
        lay.addLayout(ctl)
        note = QtWidgets.QLabel("Skipped bins are still plotted (hollow); the fit starts after them.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        self._clr = self._make_clearance_fig()
        lay.addWidget(self._clr["fig"], 1)
        return page

    def _make_clearance_fig(self, title=None):
        """Plot + fit results + warnings, for the Clearance tab and (a fresh copy) for the saved PNG."""
        fig = QtWidgets.QWidget()
        fig.setObjectName("sidepage")
        fl = QtWidgets.QVBoxLayout(fig)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        pw = pg.PlotWidget(background="w")
        p = pw.getPlotItem()
        if title:
            p.setTitle(f"<span style='color:#444444'>{title}</span>")
        p.setLabel("bottom", "Time (h)")
        p.getAxis("left").enableAutoSIPrefix(False)
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.showGrid(x=False, y=False)
        p.setMouseEnabled(x=False, y=False)
        p.setMenuEnabled(False)
        p.hideButtons()
        p.addLegend(offset=(-10, 10))
        h = {"fig": fig, "plot": p,
             "skip": pg.ScatterPlotItem(size=7, pen=pg.mkPen("#9aa4ae", width=1.2), brush=pg.mkBrush("#ffffff"),
                                        name="not fitted (skipped)"),
             "pts": pg.ScatterPlotItem(size=7, pen=pg.mkPen("#1f5fbf"), brush=pg.mkBrush("#5a8fd8"), name="fitted bins"),
             "fit": pg.PlotCurveItem(pen=pg.mkPen((220, 60, 60), width=2), name="single-phase exponential fit"),
             "vline": pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen("#888888", width=1, style=QtCore.Qt.PenStyle.DashLine))}
        for k in ("skip", "pts", "fit", "vline"):
            p.addItem(h[k])
        fl.addWidget(pw, 1)
        h["info"] = QtWidgets.QLabel("")
        h["info"].setTextFormat(QtCore.Qt.TextFormat.RichText)
        h["info"].setWordWrap(True)
        h["info"].setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        h["info"].setStyleSheet("QLabel { background:#f6f8fa; color:#24292f; font-size:12px; border:1px solid #e4e7eb;"
                                " border-radius:6px; padding:8px; }")
        fl.addWidget(h["info"])
        h["warn"] = QtWidgets.QLabel("")
        h["warn"].setTextFormat(QtCore.Qt.TextFormat.RichText)
        h["warn"].setWordWrap(True)
        h["warn"].setStyleSheet("QLabel { background:#fff4e5; color:#7a3e00; font-size:12px; border:1px solid #f0b46a;"
                                " border-radius:6px; padding:8px; }")
        h["warn"].setVisible(False)
        fl.addWidget(h["warn"])
        return h

    @staticmethod
    def _fit_single_exp(u, y):
        """Least-squares fit of y = Y0 * exp(-k * u) (u = time since the fit start, min).
        Returns Y0, k, their standard errors and R^2."""
        pos = y > 0
        if pos.sum() >= 2 and np.ptp(u[pos]) > 0:          # start from a log-linear fit
            slope, icpt = np.polyfit(u[pos], np.log(y[pos]), 1)
            k0, y00 = -slope, float(np.exp(icpt))
        else:
            k0, y00 = 1.0 / max(np.ptp(u), 1.0), float(y.max())
        popt, pcov = curve_fit(lambda u, y0, k: y0 * np.exp(-k * u), u, y, p0=[y00, k0], maxfev=20000)
        y0, k = (float(v) for v in popt)
        if not np.all(np.isfinite(popt)):
            raise RuntimeError("the fit did not converge")
        se = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan])
        res = y - y0 * np.exp(-k * u)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        return y0, k, float(se[0]), float(se[1]), r2

    def _clearance_data(self):
        """Bins, fit and texts of the clearance plot for the current bin size / skip (x in hours)."""
        bin_min, skip_min = self._clr_bin.value(), self._clr_skip.value()
        done = self._final_t
        n_bins = int(done // (bin_min * 60))                   # complete bins only
        d = {"bin": bin_min, "skip": skip_min, "skip_xy": ([], []), "fit_xy": ([], []), "curve": ([], []),
             "vline": None, "xr": None, "yr": None, "info": "", "warns": []}
        warns = d["warns"]
        if n_bins < 1:
            warns.append(f"Cannot plot yet: no complete {bin_min}-min bin (analysed so far: "
                         f"{self._fmt_hms(done)}). The plot appears as soon as the first bin is complete.")
            return d
        edges = np.arange(n_bins + 1, dtype=float) * bin_min           # min
        counts, _ = np.histogram(np.asarray(self._det_t, float) / 60.0, bins=edges)
        centers = edges[:-1] + bin_min / 2.0
        in_fit = edges[:-1] >= skip_min                        # bins that start at or after X
        d["skip_xy"] = (centers[~in_fit] / 60.0, counts[~in_fit])          # x axis in hours
        d["fit_xy"] = (centers[in_fit] / 60.0, counts[in_fit])
        d["xr"] = (0, edges[-1] / 60.0)
        d["yr"] = (0, max(1, int(counts.max())) * 1.08)
        if skip_min > 0:
            d["vline"] = skip_min / 60.0

        zero = np.flatnonzero(counts == 0)
        if len(zero):
            where = ", ".join(f"{edges[i]:g}-{edges[i + 1]:g}" for i in zero[:6]) + (
                f" and {len(zero) - 6} more" if len(zero) > 6 else "")
            n_zero_fit = int(in_fit[zero].sum())
            warns.append(f"{len(zero)} of {n_bins} bins have 0 detected CTCs ({where} min); "
                         f"{n_zero_fit} of them are in the fit. A gap in the recording, a paused DAQ or a "
                         f"laser-off period gives empty bins; a larger bin size helps if the counts are low.")

        tf, yf = centers[in_fit], counts[in_fit].astype(float)
        fit = None
        if len(tf) < 3:
            warns.append(f"Cannot fit: only {len(tf)} complete bin(s) after the first {skip_min} min "
                         f"(at least 3 needed). Lower 'Skip first' or the bin size.")
        elif not yf.any():
            warns.append(f"Cannot fit: every bin after the first {skip_min} min has 0 detected CTCs.")
        else:
            try:
                fit = self._fit_single_exp(tf - skip_min, yf)
            except Exception as e:
                warns.append(f"The exponential fit failed: {e}")

        n_tot = int(counts.sum())
        fit_rng = f"{edges[in_fit.argmax()]:g}-{edges[-1]:g} min" if in_fit.any() else "-"
        foot = (f"fit: {fit_rng}, {int(in_fit.sum())} bins of {bin_min} min · {n_tot} detected CTCs in "
                f"0-{edges[-1]:g} min (analysed: {self._fmt_hms(done)})")
        if fit is None:
            d["info"] = f"<b>No fit</b><br><span style='color:#555555'>{foot}</span>"
            return d
        y0, k, se_y0, se_k, r2 = fit
        tt = np.linspace(skip_min, edges[-1], 400)
        d["curve"] = (tt / 60.0, y0 * np.exp(-k * (tt - skip_min)))
        pm = lambda v, se: f"{v:.3g} ± {se:.2g}" if np.isfinite(se) else f"{v:.3g}"
        r2s = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
        if k > 0:
            tau, se_tau = 1.0 / k, se_k / k ** 2
            arg = f"-t / {tau:.3g}" if skip_min == 0 else f"-(t - {skip_min}) / {tau:.3g}"
            d["info"] = (f"<b>Single-phase exponential decay</b><br>"
                         f"<span style='font-size:13px'>N(t) = {y0:.3g} · e<sup>{arg}</sup></span>"
                         f" &nbsp; <span style='color:#777777'>(t = time since the recording start, min)</span><br>"
                         f"<b>τ = {pm(tau, se_tau)} min</b> &nbsp;·&nbsp; "
                         f"<b>half-life t½ = τ · ln 2 = {pm(tau * np.log(2), se_tau * np.log(2))} min</b><br>"
                         f"Y0 = {pm(y0, se_y0)} CTCs per {bin_min} min at t = {skip_min} min"
                         f" &nbsp;·&nbsp; R² = {r2s}<br><span style='color:#555555'>{foot}</span>")
        else:
            warns.append(f"No decay: the detected CTC counts do not decrease after the first {skip_min} min "
                         f"(fitted rate {k:.3g} per min), so there is no half-life. Check the 'Skip first' "
                         f"value or wait for more data.")
            d["info"] = (f"<b>Single-phase exponential fit</b> &nbsp; N(t) = {y0:.3g} · e<sup>-{k:.3g} · "
                         f"(t - {skip_min})</sup> &nbsp; R² = {r2s}<br><span style='color:#555555'>{foot}</span>")
        return d

    @staticmethod
    def _fill_clearance(h, d):
        p = h["plot"]
        p.setLabel("left", f"Detected CTCs per {d['bin']} min")
        h["skip"].setData(*d["skip_xy"])
        h["pts"].setData(*d["fit_xy"])
        h["fit"].setData(*d["curve"])
        h["vline"].setVisible(d["vline"] is not None)
        if d["vline"] is not None:
            h["vline"].setValue(d["vline"])
        if d["xr"] is not None:
            p.setXRange(*d["xr"], padding=0.02)
            p.setYRange(*d["yr"], padding=0)
        h["info"].setText(d["info"])
        h["info"].setVisible(bool(d["info"]))
        h["warn"].setText("<br>".join(f"⚠ {w}" for w in d["warns"]))
        h["warn"].setVisible(bool(d["warns"]))

    def _update_clearance(self):
        if self._clr is None:
            return
        key = (self._clr_bin.value(), self._clr_skip.value(),
               int(self._final_t // (self._clr_bin.value() * 60)), len(self._det_t))
        if key == self._clr_key:
            return
        self._clr_key = key
        d = self._clearance_data()
        self._fill_clearance(self._clr, d)
        text = "\n".join(d["warns"])
        if text != self._clr_warn_text:                        # log each new warning once
            for w in d["warns"]:
                log_app.warning("clearance: %s", w)
            self._clr_warn_text = text

    def _save_clearance(self):
        name = f"clearance_bin{self._clr_bin.value()}min_skip{self._clr_skip.value()}min.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save clearance plot", os.path.join(self.out_dir, name), "PNG (*.png)")
        if not path:
            return
        try:                                     # a full-size copy, whatever the width of the sidebar
            h = self._make_clearance_fig(title=f"Clearance - {self.exp_name}")
            self._fill_clearance(h, self._clearance_data())
            fig = h["fig"]
            fig.setStyleSheet("QWidget#sidepage { background:#ffffff; }")
            fig.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            fig.resize(1000, 640)
            fig.show()
            QtWidgets.QApplication.processEvents()
            fig.grab().save(path)
            fig.close()
            fig.deleteLater()
            log_app.info("Clearance plot saved to %s", path)
        except Exception as e:
            log_app.error("Save clearance plot failed: %s", e)

    def _save_clip(self):
        c = self._current
        if c is None:
            log_app.warning("No clip to save - click a peak first")
            return
        name = f"peak_{c['sample']}_{c['prob']:.3f}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save clip", os.path.join(self.out_dir, name), "PNG (*.png)")
        if path:
            try:
                self._clip_win.grab().save(path)
                log_app.info("Clip saved to %s", path)
            except Exception as e:
                log_app.error("Save clip failed: %s", e)

    def _on_manual_pan(self):
        self._user_panned = True

    def _reset_view(self):
        """Snap the signal trace back to the newest data and resume auto-pan."""
        self._user_panned = False
        if self._buf.n == 0:
            return
        t_end = float(self._buf.t[-1])
        self.trace.setXRange(t_end - WINDOW_SECS, t_end, padding=0)
        self.trace.setYRange(TRACE_Y_MIN, TRACE_Y_MAX, padding=0)

    # --------------------------------------------------------------- update ---
    def _drain(self):
        """Apply every queued update (a backlog at start-up is caught up quickly), then redraw once."""
        n, t0 = 0, time.time()
        while time.time() - t0 < 0.1:                  # short: a click never waits long behind a backlog
            try:
                p = PLOT_QUEUE.get_nowait()
            except queue.Empty:
                break
            if p["kind"] == "config":
                self.cut_off = p["cut_off"]
                continue
            if p["kind"] == "progress":
                self._progress_text = p["text"]
                frac = p.get("frac")
                self._status_bar.setVisible(bool(p["text"]) and frac is not None)
                if frac is not None:
                    self._status_bar.setValue(int(1000 * frac))
                self._update_status()
                continue
            if p["kind"] == "y_max":
                self._auto_y_max(p["value"])
                continue
            if p["kind"] == "error":
                self._error_text = (f"{p['count']} file(s) could not be read and were skipped - last: "
                                    f"{p['text']}")
                self._update_status()
                continue
            if p["kind"] == "finished":
                self._finished_text = p["text"]
                self._pause_btn.setEnabled(False)           # nothing left to pause
                self._update_status()
                continue
            if p["kind"] == "paused":
                if self._paused_since is not None:
                    self._waiting = p["waiting_s"]
                    self._update_status()
                continue
            self._apply(p)
            n += 1
        if n:
            self._redraw()

    def _auto_y_max(self, peak):
        """The signal at the start goes above the trace's y limit: raise the limit to it + margin."""
        global TRACE_Y_MAX
        if not np.isfinite(peak) or peak <= TRACE_Y_MAX:
            return
        TRACE_Y_MAX = float(np.ceil(peak) + Y_AUTO_MARGIN)
        log_app.info("signal y limit raised to %.0f mV (maximum in the first %d s: %.0f mV)",
                     TRACE_Y_MAX, Y_AUTO_SECS, peak)
        self.trace.setLimits(yMin=TRACE_Y_MIN, yMax=TRACE_Y_MAX, minYRange=TRACE_Y_MAX - TRACE_Y_MIN)
        self.trace.setYRange(TRACE_Y_MIN, TRACE_Y_MAX, padding=0)

    def _apply(self, p):
        self._buf.append(p["t_plot"], p["d_plot"])
        for c in p["cands"]:
            self._add_candidate(c)
        self._summary = p["summary"]
        self._final_t = p["final_t"]
        self._last = p

    def _signal_at(self, t):
        """Highest Ch1 display value within the highlight around t (label height)."""
        tb = self._buf.t
        if len(tb) == 0:
            return TRACE_Y_MAX * 0.5
        a, b = np.searchsorted(tb, [t - HIGHLIGHT_W / 2, t + HIGHLIGHT_W / 2])
        seg = self._buf.ch(0)[a:max(b, a + 1)]
        return float(seg.max()) if len(seg) else TRACE_Y_MAX * 0.5

    def _add_candidate(self, c):
        i = bisect.bisect(self._cand_t, c["t"])
        self._cand_t.insert(i, c["t"])
        self._cands.insert(i, c)
        if c["prob"] >= self.cut_off:
            bisect.insort(self._det_t, c["t"])
            self._barcode_dirty = True

    @staticmethod
    def _barcode_span_for(t):
        """Barcode x-axis end for data up to t (s), rounded up to a fixed step: 10 min below
        1 h, then whole hours. Between steps the scale is constant, so each peak stays on the
        same pixel column instead of the pattern re-quantising whenever the axis grows."""
        step = 600.0 if t < 3600 else 3600.0
        return max(step, np.ceil(t / step) * step)

    def _barcode_xy(self):
        x = np.repeat(np.asarray(self._det_t, float), 2)
        y = np.tile([0.0, 1.0], len(self._det_t))
        return x, y

    def _make_highlight(self, i):
        """Shaded rectangle + probability label of candidate i (index in self._cands)."""
        c = self._cands[i]
        above = c["prob"] >= self.cut_off
        base = self._peak_color if above else QtGui.QColor(COLOR_ARTIFACT)
        fill = QtGui.QColor(base)
        fill.setAlpha(70 if above else 45)
        half_w = HIGHLIGHT_W / 2.0
        rgn = HoverRegion(loc=c["t"], prob=c["prob"], edge=c["edge"], tooltip_item=self._tooltip,
                          on_click=self._jump_to_time, values=(c["t"] - half_w, c["t"] + half_w),
                          orientation="vertical", movable=False, brush=pg.mkBrush(fill),
                          pen=pg.mkPen(base, width=0.5))
        txt = pg.TextItem(text=f"{c['prob']:.2f}", color=base, anchor=(0.5, 1.0))
        margin = (TRACE_Y_MAX - TRACE_Y_MIN) * 0.04
        near = 0                                   # stack the labels of candidates < 0.6 s apart
        while near < i and c["t"] - self._cand_t[i - near - 1] < 0.6:
            near += 1
        txt.setPos(c["t"], min(self._signal_at(c["t"]) + margin * (1 + 1.6 * near), TRACE_Y_MAX - margin))
        self.trace.addItem(rgn)
        self.trace.addItem(txt)
        return rgn, txt

    def _sync_highlights(self):
        """Keep highlight items only for candidates in (and next to) the visible time range,
        and none when more than HIGHLIGHT_MAX_SPAN is shown: the item count stays small
        however long the recording, and panning stays fast."""
        (x0, x1), _ = self.trace.getViewBox().viewRange()
        span = x1 - x0
        self.trace.getViewBox().setToolTip("Click a laser pulse to zoom in" if span > HIGHLIGHT_MAX_SPAN else "")
        want = set()
        if span <= HIGHLIGHT_MAX_SPAN and self._buf.n:
            lo, hi = max(x0 - span / 2, float(self._buf.t[0])), x1 + span / 2
            a, b = bisect.bisect_left(self._cand_t, lo), bisect.bisect_right(self._cand_t, hi)
            want = set(range(a, b))
        keep_t = {self._cand_t[i] for i in want}
        for t in [t for t in self._highlights if t not in keep_t]:
            rgn, txt = self._highlights.pop(t)
            self.trace.removeItem(rgn)
            self.trace.removeItem(txt)
        for i in sorted(want):
            if self._cand_t[i] not in self._highlights:
                self._highlights[self._cand_t[i]] = self._make_highlight(i)

    def _redraw(self):
        # 1. signal curves - rolling buffer of DISPLAY_HISTORY_SECS (+ up to one trim step)
        self._buf.trim(DISPLAY_HISTORY_SECS, TRIM_STEP_SECS)
        tb = self._buf.t
        if not self._curves:
            self._curves = [self.trace.plot(pen=pg.mkPen(self.CH_COLORS[ch], width=TRACE_PEN_WIDTH), antialias=False,
                                            name=f"Ch{ch + 1}") for ch in range(2)]
        for ch, crv in enumerate(self._curves):
            crv.setData(tb, self._buf.ch(ch), skipFiniteCheck=True)
        if not self._user_panned and len(tb):
            t_end = float(tb[-1])
            self.trace.setXRange(t_end - WINDOW_SECS, t_end, padding=0)
        if len(tb):                                     # not analysed yet: grey band at the end
            t_end, t_fin = float(tb[-1]), float(self._final_t or 0.0)
            show = t_end - t_fin > 0.2 and not getattr(self, "_finished_text", None)
            if show:
                self._pending.setRegion((max(t_fin, float(tb[0])), t_end))
                self._pending_txt.setPos(t_end, TRACE_Y_MAX)
            self._pending.setVisible(show)
            self._pending_txt.setVisible(show)

        # 2. highlights + probability labels of the visible candidates (+ the magnifier)
        self._sync_highlights()
        self._update_magnifier()
        self._update_review_panel()            # set sizes / review counts grow with new candidates

        # 3. barcode: redrawn only when a peak was added; x-axis t = 0 → newest data
        if self._barcode_dirty:
            self._barcode_curve.setData(*self._barcode_xy())
            self._barcode_dirty = False
            self._bc_hint.setText(f"{len(self._det_t)} detected · full experiment · click a line to review it")
        if len(tb) and float(tb[-1]) > self._t_max:
            self._t_max = float(tb[-1])
            span = self._barcode_span_for(self._t_max)
            if span != self._barcode_span:     # rescale only when a step is crossed, so the peak
                self._barcode_span = span      # pattern does not shift with every new chunk
                self.barcode.setXRange(0, span, padding=0)

        # 4. clearance window (if open): refreshed when a bin is completed or a peak is added
        if self._clr is not None and self._clr["fig"].isVisible():
            self._update_clearance()

        # 5. status
        self._update_status()

    def _update_status(self):
        s = self._summary
        line1 = (f"<b>{s['laser_mode'] or 'waiting for 1 min of data'} laser · "
                 f"recorded {self._fmt_hms(s['hours_recorded'] * 3600)} · "
                 f"laser-ON {s['laser_on_hours'] * 60:.1f} min</b>") if s else (
                 {"offline": "<b>Offline mode - the saved data is analysed (no new data is read)</b>",
                  "real-time": "<b>Real-time mode - the saved data is analysed first, chunk by chunk</b>"}.get(
                     getattr(self.worker, "analysis", ""), "<b>Saved data found - it is analysed first, then real time</b>")
                 if getattr(self, "_progress_text", None) else "<b>Waiting for data…</b>")
        if getattr(self, "_finished_text", None):
            line2 = f"<b><span style='color:#1a7f37'>{self._finished_text}</span></b>"
        elif getattr(self, "_progress_text", None) and self._paused_since is None:
            line2 = f"<b><span style='color:#1f5fbf'>{self._progress_text}</span></b>"
        elif self._paused_since is not None:
            if self._waiting is None:
                wait = "pausing after the current step…"
            else:
                wait = (f"{self._waiting / 3600:.1f} h" if self._waiting >= 3600 else f"{self._waiting / 60:.0f} min")
                wait += " of data waiting; press Resume to catch up"
            line2 = (f"<b><span style='color:#cf222e'>PAUSED for {self._fmt_hms(time.time() - self._paused_since)}"
                     f" - {wait}</span></b>")
        elif s:
            new = self._last["cands"]
            n_new_det = sum(c["prob"] >= self.cut_off for c in new)
            line2 = (f"<b>analysed to t = {self._final_t:.0f} s : new candidates {len(new)} ({n_new_det} detected) | "
                     f"total detected peaks: {s['detected']} of {s['candidates']} candidates</b>")
        else:
            line2 = ""
        err = getattr(self, "_error_text", None)
        line3 = f"<span style='color:#cf222e'>⚠ {html.escape(err)}</span>" if err else ""
        self._set_status("<br>".join(x for x in (line1, line2, line3) if x))

    def _toggle_pause(self, paused):
        ev = getattr(self.worker, "pause_event", None)
        if ev is not None:
            ev.set() if paused else ev.clear()
        self._paused_since = time.time() if paused else None
        self._waiting = None                                   # until the worker has stopped and reported
        self._pause_btn.setText("▶  Resume" if paused else "⏸  Pause")
        log_app.info("processing %s", "paused" if paused else "resumed")
        self._update_status()


EXP_NAME_RE = re.compile(r"^\d{6}-")             # experiment names start with the date, e.g. 250723-BCMM2_MM-CFSE


class LaunchDialog(QtWidgets.QDialog):
    """Start-up window of the desktop app: experiment folder, analysis mode, results folder and the
    advanced options; the choices are remembered for the next start (QSettings)."""
    MODES = (("real-time", "Real time",
              "Everything with the real-time algorithm from the first chunk, then new data as it is saved."),
             ("offline-then-real-time", "Offline, then real time",
              "The data already saved with the offline pipeline (2-h blocks, faster), then new data in "
              "real time."),
             ("offline", "Offline only",
              "Only the data already saved (a finished experiment); no new data is read."))
    CSS = (TDiFCWindow.APP_CSS.replace("QWidget#root", "QDialog#launch")
           + " QLineEdit { font-size:12px; color:#24292f; padding:3px 6px; border:1px solid #c9ced6;"
             " border-radius:5px; background:#ffffff; }"
           + " QRadioButton { font-size:12px; color:#24292f; font-weight:600; }")

    def __init__(self, args, settings):
        super().__init__()
        self.setObjectName("launch")
        self.setStyleSheet(self.CSS)
        self.setWindowTitle(APP_NAME)
        self._st = settings
        self._last_path = None         # experiment folder last looked into
        self._results_root = RESULTS_HOME
        self._results_auto = True      # results folder = <root>/<experiment name> until edited by hand
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        head = QtWidgets.QLabel(f"<span style='font-size:18px; font-weight:600'>{APP_NAME}</span><br>"
                                f"<span style='color:#6e7781'>Real-time CTC detection in t-DiFC recordings</span>")
        lay.addWidget(head)

        # data
        card, cl, _ = TDiFCWindow._card("Data", hint="the experiment folder DiDAQ writes into (with F1, F2, …), or "
                                                     "any folder of .mat files (read in the order of their names)")
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit()
        self.folder.setPlaceholderText("/Volumes/Public/Jane/…/<experiment>")
        self.folder.editingFinished.connect(self._inspect)
        row.addWidget(self.folder, 1)
        row.addWidget(self._btn("Browse…", self._browse_data))
        cl.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Experiment name"))
        self.name = QtWidgets.QLineEdit()
        self.name.setToolTip("Used for the results folder. DiDAQ folders: the scan name of the files "
                             "<name>_F1_1.mat, <name>_F1_2.mat, …")
        self.name.textChanged.connect(self._sync_results)
        row.addWidget(self.name, 1)
        cl.addLayout(row)
        self.found = QtWidgets.QLabel("")
        self.found.setWordWrap(True)
        cl.addWidget(self.found)
        lay.addWidget(card)

        # analysis mode
        card, cl, _ = TDiFCWindow._card("Analysis")
        self._mode_btns = {}
        for key, label, text in self.MODES:
            rb = QtWidgets.QRadioButton(label)
            self._mode_btns[key] = rb
            cl.addWidget(rb)
            d = QtWidgets.QLabel(text)
            d.setObjectName("muted")
            d.setWordWrap(True)
            d.setContentsMargins(22, 0, 0, 4)
            cl.addWidget(d)
        lay.addWidget(card)

        # results
        card, cl, _ = TDiFCWindow._card("Results folder")
        row = QtWidgets.QHBoxLayout()
        self.results = QtWidgets.QLineEdit()
        self.results.setToolTip("Follows the experiment name; type here to choose a folder of your own")
        self.results.textEdited.connect(self._results_edited)
        row.addWidget(self.results, 1)
        row.addWidget(self._btn("Browse…", self._browse_results))
        cl.addLayout(row)
        self.results_hint = QtWidgets.QLabel("")
        self.results_hint.setObjectName("muted")
        self.results_hint.setWordWrap(True)
        cl.addWidget(self.results_hint)
        lay.addWidget(card)

        # advanced options (hidden until opened)
        self.adv_btn = QtWidgets.QToolButton()
        self.adv_btn.setText("Advanced options")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.adv_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_btn.setStyleSheet("QToolButton { border:none; color:#57606a; font-size:12px; font-weight:600; }")
        self.adv_btn.toggled.connect(self._toggle_adv)
        lay.addWidget(self.adv_btn)
        self.adv = QtWidgets.QFrame()
        self.adv.setObjectName("card")
        g = QtWidgets.QFormLayout(self.adv)
        g.setContentsMargins(12, 10, 12, 10)
        self.laser = QtWidgets.QComboBox()
        self.laser.addItems(["auto", "pulsed", "continuous"])
        self.laser.setToolTip("auto: decided from the last 5 min of Ch1")
        g.addRow("Laser illumination", self.laser)
        self.fibers = QtWidgets.QComboBox()
        self.fibers.addItems(["auto", "1 (Ch1 and Ch2 in F1)", "2 (Ch1 in F1, Ch2 in F2)"])
        self.fibers.setToolTip("auto: Ch1 and Ch2 from F1 if its files have two columns, else Ch2 from F2.\n"
                               "2: Ch1 = 1st column of the F1 file, Ch2 = 1st column of the F2 file (F2 files "
                               "with the same name, without F1/F2, or else in the same order).")
        g.addRow("Fiber folders", self.fibers)
        self.cut = QtWidgets.QDoubleSpinBox()
        self.cut.setRange(0.0, 1.0)
        self.cut.setSingleStep(0.05)
        self.cut.setDecimals(2)
        self.cut.setSpecialValueText("model default (0.6)")       # shown at 0
        g.addRow("Probability cut-off", self.cut)
        self.ymin, self.ymax = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sb in (self.ymin, self.ymax):
            sb.setRange(-100000, 100000)
            sb.setSuffix(" mV")
        yr = QtWidgets.QHBoxLayout()
        yr.addWidget(self.ymin)
        yr.addWidget(QtWidgets.QLabel("to"))
        yr.addWidget(self.ymax)
        g.addRow("Signal y-range", yr)
        self.adv.setVisible(False)
        lay.addWidget(self.adv)

        lay.addStretch()
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        btns.addWidget(self._btn("Quit", self.reject))
        self.start = self._btn("▶  Start", self._accept, height=32)
        self.start.setDefault(True)
        self.start.setStyleSheet("QPushButton { background:#1f6feb; color:#ffffff; border:1px solid #1a5fd0;"
                                 " border-radius:5px; font-size:13px; font-weight:600; padding:0 18px; }"
                                 " QPushButton:hover { background:#1a5fd0; }")
        btns.addWidget(self.start)
        lay.addLayout(btns)

        # values: command line > last start > defaults
        st = lambda k, d: settings.value(f"launch/{k}", d)
        folder = (os.path.join(args.save_dir, args.exp_name) if args.save_dir and args.exp_name
                  else args.save_dir or st("folder", ""))
        self.folder.setText(str(folder))
        if args.out_dir:
            self._results_root, self._results_auto = os.path.dirname(args.out_dir.rstrip("/")), False
            self.results.setText(args.out_dir)
        else:
            self._results_root = self._root_of(str(st("results_root", "") or os.path.dirname(  # older versions: one experiment's
                str(st("results", "")).rstrip("/")) or RESULTS_HOME), str(st("name", "")))  # results folder
        self._mode_btns.get(args.analysis if args.analysis_given else str(st("analysis", DEFAULT_ANALYSIS)),
                            self._mode_btns[DEFAULT_ANALYSIS]).setChecked(True)
        self.laser.setCurrentText(args.mode if args.mode != "auto" else str(st("laser", "auto")))
        self.fibers.setCurrentIndex(int(args.n_fibers or st("fibers", 0) or 0))
        self.cut.setValue(float(args.cut_off if args.cut_off is not None else st("cut_off", 0.0)))
        self.ymin.setValue(int(float(st("y_min", TRACE_Y_MIN))))
        self.ymax.setValue(int(float(st("y_max", TRACE_Y_MAX))))
        self._inspect()
        if args.exp_name:
            self.name.setText(args.exp_name)
        elif not self.name.text():
            self.name.setText(str(st("name", "")))
        self.results_hint.setText("candidates.csv, summary.json, review.csv and tdifc_app.log are saved here.")
        self._sync_results()
        self.setMinimumWidth(640)
        self.adjustSize()

    def _btn(self, text, slot, height=28):
        b = QtWidgets.QPushButton(text)
        b.setFixedHeight(height)
        b.setStyleSheet(TDiFCWindow.BTN_STYLE)
        b.clicked.connect(slot)
        return b

    def _toggle_adv(self, on):
        self.adv.setVisible(on)
        self.adv_btn.setArrowType(QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow)
        QtCore.QTimer.singleShot(0, self.adjustSize)

    def _browse_data(self):
        start = self.folder.text() or "/Volumes"
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder (with F1, F2, … or .mat files)", start)
        if d:
            self.folder.setText(d)
            self._inspect()

    def _browse_results(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the results (a folder named after the "
                                                       "experiment is made in it)", self._results_root)
        if d:
            name = self.name.text().strip()
            self._results_root = self._root_of(d, name)
            self._results_auto = True
            self._sync_results()

    @staticmethod
    def _root_of(d, name=""):
        """The folder that holds the experiments' results folders. A folder that is itself one experiment's
        results (named after the experiment, or with candidates.csv etc. in it) gives its parent."""
        d = os.path.expanduser(str(d or RESULTS_HOME)).rstrip("/") or RESULTS_HOME
        marks = ("candidates.csv", "summary.json", "review.csv", "tdifc_app.log")
        for _ in range(3):                         # also a results folder made inside another one
            base = os.path.basename(d)
            own = ((name and base.lower() == name.lower()) or EXP_NAME_RE.match(base)
                   or any(os.path.isfile(os.path.join(d, m)) for m in marks))
            if not own or os.path.dirname(d) in ("", "/", d):
                break
            d = os.path.dirname(d)
        return d

    def _results_edited(self, *_):
        self._results_auto = False                 # typed by hand: no longer follows the name

    def _sync_results(self, *_):
        if self._results_auto:
            self.results.setText(os.path.join(self._results_root, self.name.text().strip()))

    def _inspect(self):
        """Look into the data folder: what will be read, and the experiment name it gives."""
        path = self.folder.text().strip().rstrip("/")
        up = os.path.dirname(path)
        if re.fullmatch(r"F\d+", os.path.basename(path)) and (
                didaq_layout(up)[0] or os.path.isdir(os.path.join(up, "F2"))):
            path = up                                          # the F1 folder of an experiment (with F2)
            self.folder.setText(path)
        if not path:
            self.found.setText("")
            return
        ok, warn = "color:#1a7f37", "color:#9a6700"
        name = os.path.basename(path)
        if re.fullmatch(r"F\d+", name) and os.path.dirname(path) not in ("", "/"):
            name = os.path.basename(os.path.dirname(path))     # a plain F1 folder: the experiment's name
        if not os.path.isdir(path):
            self.found.setText(f"<span style='{warn}'>⚠ This folder does not exist (yet).</span>")
        else:
            didaq, n = didaq_layout(path)
            if didaq:
                name = didaq
                f2 = os.path.isdir(os.path.join(path, "F2"))
                dur = ("1 file (a long file is analysed in 10-s slices in real time)" if n == 1
                       else f"{n} chunk files, about {n * 10 / 3600:.1f} h")
                self.found.setText(f"<span style='{ok}'>✓ DiDAQ folder: {dur} in F1{' (+ F2)' if f2 else ''}</span>")
            else:
                d = path
                chunks, stem, ignored = scan_mat_folder(d)
                if not chunks and os.path.isdir(os.path.join(path, "F1")):
                    d = os.path.join(path, "F1")
                    chunks, stem, ignored = scan_mat_folder(d)
                if not chunks:
                    self.found.setText(f"<span style='{warn}'>⚠ No .mat files yet: the app waits for them "
                                       f"(DiDAQ chunks in F1, or .mat files in this folder).</span>")
                else:
                    first, last = (html.escape(next(iter(c.values()))) for c in (chunks[0], chunks[-1]))
                    fib = sorted({f for c in chunks for f in c})
                    size = sum(os.path.getsize(os.path.join(d, f)) for c in chunks for f in c.values()
                               if os.path.exists(os.path.join(d, f)))
                    where = "" if d == path else " in F1"
                    what = (f"1 file{where}: {first}" if len(chunks) == 1 else
                            f"{len(chunks)} files{where}, read in this order: {first} … {last}")
                    extra = f" (+ F{', F'.join(map(str, fib[1:]))} files)" if len(fib) > 1 else ""
                    if d != path and os.path.isdir(os.path.join(path, "F2")) and scan_mat_folder(
                            os.path.join(path, "F2"))[0]:
                        k = len(scan_mat_folder(os.path.join(path, "F2"))[0])
                        extra = f" (+ {k} file{'s' if k != 1 else ''} in F2)"
                    txt = f"<span style='{ok}'>✓ {what}{extra}, {size / 1e9:.2f} GB</span>"
                    if ignored:
                        txt += (f"<br><span style='{warn}'>⚠ not used ({len(ignored)}, other names): "
                                f"{html.escape(', '.join(ignored[:3]))}{' …' if len(ignored) > 3 else ''}</span>")
                    self.found.setText(txt)
                    if stem and (EXP_NAME_RE.match(stem) or (len(chunks) > 1 and len(stem) > 3)):
                        name = stem
        if path != self._last_path:                # a new folder: its experiment name (and results folder)
            if not self._results_auto:             # a typed results path: keep its place, not its name
                self._results_root = self._root_of(os.path.dirname(self.results.text().strip().rstrip("/")))
                self._results_auto = True
            self.name.setText(name)
            self._sync_results()
        self._last_path = path

    def _accept(self):
        folder, name = self.folder.text().strip().rstrip("/"), self.name.text().strip()
        out = self.results.text().strip().rstrip("/")
        if not folder or not name or not out:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Choose the experiment folder and its name first.")
            return
        if self.ymax.value() <= self.ymin.value():
            QtWidgets.QMessageBox.warning(self, APP_NAME, "The signal y-range must go from low to high.")
            return
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"The results folder cannot be made:\n{e}")
            return
        mode = next(k for k, b in self._mode_btns.items() if b.isChecked())
        self.choice = {"save_dir": os.path.dirname(folder), "root": folder, "exp_name": name, "out_dir": out,
                       "analysis": mode, "mode": self.laser.currentText(),
                       "n_fibers": self.fibers.currentIndex() or None,
                       "cut_off": self.cut.value() or None, "y_min": self.ymin.value(), "y_max": self.ymax.value()}
        root = self._results_root if self._results_auto else self._root_of(os.path.dirname(out))
        for k, v in (("folder", folder), ("name", name), ("results_root", root), ("analysis", mode),
                     ("laser", self.laser.currentText()), ("fibers", self.fibers.currentIndex()),
                     ("cut_off", self.cut.value()), ("y_min", self.ymin.value()), ("y_max", self.ymax.value())):
            self._st.setValue(f"launch/{k}", v)
        self.accept()


def light_theme(app):
    """The window is drawn on white. Force a light Fusion palette so that macOS dark mode does not
    turn widget text (check boxes, drop-down) white, i.e. invisible on the white panels."""
    app.setStyle("Fusion")
    R, G = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    pal = QtGui.QPalette()
    for role, col in ((R.Window, "#ffffff"), (R.WindowText, "#111111"), (R.Base, "#ffffff"),
                      (R.AlternateBase, "#f4f4f4"), (R.Text, "#111111"), (R.Button, "#f0f0f0"),
                      (R.ButtonText, "#111111"), (R.ToolTipBase, "#ffffe0"), (R.ToolTipText, "#111111"),
                      (R.PlaceholderText, "#888888"), (R.Highlight, "#2a78d6"), (R.HighlightedText, "#ffffff"),
                      (R.Link, "#2a78d6"), (R.BrightText, "#cf222e")):
        pal.setColor(role, QtGui.QColor(col))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(G.Disabled, role, QtGui.QColor("#999999"))
    app.setPalette(pal)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global TRACE_Y_MIN, TRACE_Y_MAX
    ap = argparse.ArgumentParser(description="t-DiFC real-time peak detection (PyTorch fusion model)")
    ap.add_argument("--save_dir", default=None, help="folder that contains <exp_name>/F1, F2, ... (DiDAQ save dir); "
                                                     "without it (and --exp_name) the start-up window opens")
    ap.add_argument("--exp_name", default=None, help="scan name (DiDAQ scanName)")
    ap.add_argument("--folder", default=None, help="any folder of .mat files (or a DiDAQ experiment folder), "
                                                   "read in the order of the file names; instead of "
                                                   "--save_dir/--exp_name (name: the folder's)")
    ap.add_argument("--n_fibers", type=int, default=None,
                    help="fiber folders to join for Ch1/Ch2 (default: F1, plus F2 if F1 has one channel)")
    ap.add_argument("--mode", default="auto", choices=["auto", "pulsed", "continuous"],
                    help="laser illumination (default: detected from the last 5 min of Ch1)")
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS, choices=ANALYSIS_MODES,
                    help="real-time (default): everything with the real-time algorithm from the first chunk "
                         "(a long file in 10-s slices); offline-then-real-time: data already saved -> offline "
                         "pipeline, new data -> real time; offline: only the saved data, offline pipeline "
                         "(no new data is read)")
    ap.add_argument("--out_dir", default=None,
                    help="results folder (default: ~/Documents/t-DiFC results/<exp_name>/)")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--cut_off", type=float, default=None, help="probability cut-off (default: the model's, 0.6)")
    ap.add_argument("--y_min", type=float, default=TRACE_Y_MIN)
    ap.add_argument("--y_max", type=float, default=TRACE_Y_MAX)
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args, _unknown = ap.parse_known_args()          # (macOS may add its own arguments to an app)
    if args.folder:                                 # any folder: root = it, name = its own
        args.root = os.path.abspath(os.path.expanduser(args.folder.rstrip("/")))
        args.save_dir = os.path.dirname(args.root)
        args.exp_name = args.exp_name or os.path.basename(args.root)
    args.analysis_given = any(a.startswith("--analysis") for a in sys.argv[1:])
    logging.getLogger().setLevel(args.log_level)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
    light_theme(app)
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    if not os.path.exists(args.model):
        QtWidgets.QMessageBox.critical(None, APP_NAME, f"The model file is missing:\n{args.model}")
        raise SystemExit(f"model not found: {args.model} (train it with: python train_fusion_torch.py)")
    if not (args.save_dir and args.exp_name):                  # double-click: ask in the start-up window
        dlg = LaunchDialog(args, QtCore.QSettings("t-DiFC", "tdifc_app"))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        for k, v in dlg.choice.items():
            setattr(args, k, v)
    TRACE_Y_MIN, TRACE_Y_MAX = args.y_min, args.y_max
    out_dir = args.out_dir or os.path.join(RESULTS_HOME, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, "tdifc_app.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  [%(name)-8s]  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    log_app.info("%s - %s, analysis %s", APP_NAME, getattr(args, "root", None)
                 or os.path.join(args.save_dir, args.exp_name), args.analysis)
    root = getattr(args, "root", None) or os.path.join(args.save_dir, args.exp_name)
    if not os.path.isdir(root):
        log_scan.warning("%s does not exist yet - waiting for the acquisition to start", root)

    worker = Worker(AutoSource(root, args.exp_name, args.n_fibers), out_dir, args.mode, args.model,
                    args.cut_off, analysis=args.analysis)
    log_app.info("analysis: %s", args.analysis)
    worker.start()
    log_app.info("results: %s", out_dir)
    window = TDiFCWindow(args.exp_name, out_dir, worker)   # noqa: F841 (keeps the window alive)
    try:
        sys.exit(app.exec())
    finally:
        worker.stop_event.set()


if __name__ == "__main__":
    main()
