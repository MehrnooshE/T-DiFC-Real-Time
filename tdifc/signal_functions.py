"""
Lab signal-processing functions used by the fusion_lr1e-4_wd_ls model, copied verbatim
from the fusion pipeline's scripts/signal_functions.py (t-DiFC-clearance cohort/
fusion_lr1e-4_wd_ls_pipeline), which in turn copies the project's shared functions.py.

  compute_invalid_mask, standardize_joint, extract_window   window extraction
  ALS_BGremoved_Ch2scaled (remove_baseline, build_invalid_mask,
      compute_background_std and their helpers)             spectral input (spectral.py)

Only change: ALS_BGremoved_Ch2scaled takes `progress` (default True, as before) so the
real-time app can switch the tqdm bar off. match_labels (labeling only) is left out.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import grey_opening
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

# --------------------------------------------------------------------------
# Invalid-sample masking (pulse off-gaps + zero-padded scan boundaries)
# --------------------------------------------------------------------------
def compute_invalid_mask(win, zero_tol=1e-9):
    """
    win: (2, L) array, freshly extracted, BEFORE standardization.
    Returns (L,) bool mask, True where the sample is not real signal.

    Both known invalid sources in this pipeline manifest as exact zero in
    BOTH channels simultaneously:
      - extract_window's own zero-padding at true scan boundaries (it
        allocates via np.zeros and only fills the in-bounds segment)
      - pulse off-gaps already embedded in the raw Pulsed-category signals
        (the on-regions were processed separately and concatenated with
        zero-filled gaps, per the acquisition's pulsed on/off structure)

    Must be computed on the RAW window, before standardize_joint -- after
    standardization, invalid samples become some non-zero constant
    (mu-shifted), and this detection no longer works. If you need the mask
    later, save it alongside X/y rather than trying to reconstruct it from
    standardized data.

    Using a small tolerance rather than strict ==0 for float-precision
    safety; genuine sensor noise landing at exactly 0.0 in both channels
    simultaneously is not a realistic concern for continuous analog-like
    data.
    """
    return (np.abs(win[0]) < zero_tol) & (np.abs(win[1]) < zero_tol)


def standardize_joint(win, invalid_mask=None):
    """
    Per-window standardization, jointly across ch1 and ch2: a single
    mean/std computed over both channels together, applied to the whole
    window (including invalid samples, which just map to some constant).

    If invalid_mask is given, mean/std are computed from VALID samples
    ONLY. This matters: an off-gap contributes zero variance, and computing
    mean/std over the whole window (invalid samples included) makes the
    effective standardized scale depend on how much of a given window
    happens to be off-time -- verified this distorts the standardized noise
    level by ~30-35% between a window with no off-gap and one that's ~40%
    off-gap, for otherwise-identical signal. Excluding invalid samples from
    the mean/std calculation removes that dependency.

    Falls back to using all samples if no mask is given, or if the mask
    flags every sample as invalid (fully degenerate window -- e.g. a
    peak_idx entirely out of bounds).
    """
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


# --------------------------------------------------------------------------
# Robust flank baseline/noise estimation
# --------------------------------------------------------------------------
def sigma_clipped_baseline_std(flank, n_iter=3, clip_k=3.0):
    """
    Robust (baseline, noise_std) via iterative sigma-clipping: median
    location + MAD-based scale, iteratively rejecting samples beyond
    clip_k robust-sigma, so a competing peak or other contamination landing
    in a flank region doesn't corrupt the estimate the way plain mean/std
    would. Not yet used in build_splits.py -- included here for the
    engineered_features step, so both scripts pull from one definition.
    """
    mask = np.ones(len(flank), dtype=bool)
    baseline = np.median(flank)
    robust_std = np.median(np.abs(flank - baseline)) * 1.4826
    for _ in range(n_iter):
        vals = flank[mask]
        if len(vals) < 5:
            break
        baseline = np.median(vals)
        mad = np.median(np.abs(vals - baseline))
        robust_std = mad * 1.4826
        if robust_std < 1e-9:
            break
        new_mask = np.abs(flank - baseline) < clip_k * robust_std
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    return baseline, robust_std


# --------------------------------------------------------------------------
# Window extraction (shared between build_splits.py and inference)
# --------------------------------------------------------------------------
def extract_window(ch1, ch2, peak_idx, window_len):
    """
    Extract a (2, window_len) window centered on peak_idx from ch1/ch2,
    zero-padding out-of-bounds regions so the peak always stays at index
    window_len // 2.

    Shared between build_splits.py (training-set construction) and
    inference so both extract windows IDENTICALLY -- this matters
    concretely: an earlier version of the inference-side extractor used
    edge-repeat padding (np.pad mode="edge") while build_splits.py used
    zero-padding. That's a real train/inference mismatch -- a boundary
    candidate at inference time would get a repeated real value the model
    never saw as "padding" during training. Fixed by sharing this one
    implementation instead of maintaining two.
    """
    half_left = window_len // 2
    half_right = window_len - half_left
    T = len(ch1)
    assert len(ch2) == T, "ch1/ch2 length mismatch"

    start = peak_idx - half_left
    end = peak_idx + half_right
    pad_left = max(0, -start)
    pad_right = max(0, end - T)
    actual_start = max(0, start)
    actual_end = min(T, end)

    win = np.zeros((2, window_len), dtype=np.float64)
    if actual_start < actual_end:
        seg_len = actual_end - actual_start
        win[0, pad_left:pad_left + seg_len] = ch1[actual_start:actual_end]
        win[1, pad_left:pad_left + seg_len] = ch2[actual_start:actual_end]
    else:
        print(f"[WARN] peak_idx {peak_idx} yields empty window (T={T}); returning all-zero window")
    return win


"""
Baseline / trend removal for peak-containing time series, without
dampening the peaks themselves.

Two interchangeable methods are provided:

1. morphological_baseline  - grayscale morphological opening
                              (erosion -> dilation) with a flat window
                              wider than the widest expected peak.
                              Simple, fast, and directly parameterized
                              by something you already know (max peak
                              width).

2. als_baseline             - Asymmetric Least Squares smoothing
                              (Eilers & Boelens, 2005). Produces a
                              smoother, less "staircase-y" baseline.
                              Two parameters to tune (lam, p).

Both return an estimated baseline the same length as the input; the
peak-preserving corrected signal is simply `x - baseline`.
"""



def morphological_baseline(x: np.ndarray, window: int) -> np.ndarray:
    """
    Estimate a slowly varying baseline via grayscale morphological
    opening (erosion followed by dilation) with a flat structuring
    element of length `window`.

    Any positive-going feature (peak) narrower than `window` samples
    is excluded from the baseline estimate; only features wider than
    `window` survive. Choose `window` comfortably larger than your
    widest expected peak.
    """
    if window % 2 == 0:
        window += 1  # keep the structuring element centered
    return grey_opening(x, size=window, mode="nearest")


def als_baseline(y: np.ndarray, lam: float = 1e7, p: float = 0.01,
                  niter: int = 10, mask: np.ndarray = None) -> np.ndarray:
    """
    Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    lam   : smoothness penalty (larger -> stiffer/smoother baseline)
    p     : asymmetry (small p, e.g. 0.001-0.02, makes the baseline
            hug the lower envelope and ignore upward peaks)
    niter : number of reweighting iterations
    mask  : optional boolean array, same length as y. True = valid
            sample, participates in the fit. False = invalid/dead
            sample (e.g. "pulse off" padding) that should NEVER
            influence the fitted curve. The smoothness penalty still
            lets the baseline interpolate smoothly across masked
            stretches using only the valid samples on either side.
    """
    # print("1")
    L = len(y)
    # print("2")
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    # print("3")
    D = lam * D.dot(D.transpose())
    # print("4")
    if mask is None:
        mask_f = np.ones(L)
    else:
        mask_f = mask.astype(float)

    # print("5")
    w = mask_f.copy()  # invalid samples start at weight 0 and stay there
    z = y.copy()
    for _ in range(niter):
        # print("6")
        W = diags(w, 0, shape=(L, L))
        Z = W + D
        z = spsolve(Z.tocsc(), w * y)
        # re-apply mask every iteration so invalid samples can never
        # regain weight through the asymmetric reweighting step
        w = mask_f * (p * (y > z) + (1 - p) * (y < z))
    return z


def detect_flat_regions(x: np.ndarray, window: int = 50,
                         std_thresh: float = 1e-2,
                         min_run: int = 200) -> np.ndarray:
    """
    Fallback auto-detector for "pulse off" / dead-time padding when you
    don't already know its location: flags long runs of near-constant
    signal (rolling std below std_thresh, sustained for at least
    min_run samples) as invalid.

    Returns a boolean mask, True = valid (keep), False = flat/invalid.
    Only use this if you don't have the pulse timing directly (e.g.
    from a trigger column or known acquisition-window length) --
    exact timing is always more reliable than a heuristic.
    """
    roll_std = pd.Series(x).rolling(window, center=True, min_periods=1).std()
    flat = (roll_std < std_thresh).to_numpy()

    valid = np.ones_like(flat, dtype=bool)
    i = 0
    n = len(flat)
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            if (j - i) >= min_run:
                valid[i:j] = False
            i = j
        else:
            i += 1
    return valid


def detect_zero_padding(x: np.ndarray, min_run: int = 200,
                         guard: int = 30, tol: float = 0.0) -> np.ndarray:
    """
    Detect manually-zeroed "pulse off" regions exactly, since they're
    forced to precisely 0.0 rather than merely low-variance. Extends
    each detected zero run backward AND forward by `guard` samples, to
    also exclude any edge ringing/overshoot from whatever process did
    the zeroing (that ringing is real-valued, not exactly 0, so a bare
    `x == 0` mask would miss it and let it corrupt the ALS fit).

    min_run : minimum consecutive exact-zero samples to count as a
        genuine off-region (avoids flagging an incidental single
        sample that happens to land on exactly 0.0 during real signal)
    guard   : samples of margin added on both sides of each detected
        run, to swallow edge transients
    tol     : treat |x| <= tol as "zero" (use a small tolerance if
        your zeroing isn't bit-exact)

    Returns a boolean mask, True = valid (keep), False = invalid.
    """
    is_zero = np.abs(x) <= tol
    n = len(x)
    valid = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            if (j - i) >= min_run:
                start = max(0, i - guard)
                end = min(n, j + guard)
                valid[start:end] = False
            i = j
        else:
            i += 1
    return valid


def detect_amplitude_outliers(x: np.ndarray, k: float = 8.0,
                               guard: int = 50) -> np.ndarray:
    """
    Flags samples whose magnitude is an extreme outlier relative to
    the rest of the signal (robust z-score via MAD), regardless of
    sign.

    This matters specifically because ALS's asymmetry (p) assumes
    deviations only go one way -- peaks sit ABOVE the baseline and get
    down-weighted, while anything BELOW the fitted curve is trusted as
    real background (weight ~= 1-p). A large NEGATIVE-going artifact
    (acquisition-edge overshoot, detector saturation recovery, etc.)
    is therefore treated as trustworthy baseline by plain ALS and
    drags the whole smooth curve down toward it -- producing exactly
    the kind of hump/overshoot that spreads far beyond the artifact's
    own location. Masking it out here prevents that.

    k     : threshold in robust standard deviations (MAD-based).
        Start high (8-10) so genuine peaks aren't accidentally
        excluded -- you WANT real peaks to stay in the fit's data
        (they get handled correctly by p already); this is only for
        artifacts far outside anything plausible.
    guard : samples of margin added around each flagged run, since
        the ringing on either side of the extreme point is usually
        still somewhat artifact-shaped even if it doesn't cross k.

    Returns a boolean mask, True = valid (keep), False = outlier.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826  # normal-consistent scale
    if mad == 0:
        return np.ones_like(x, dtype=bool)
    z = np.abs(x - med) / mad
    is_outlier = z > k

    n = len(x)
    valid = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if is_outlier[i]:
            j = i
            while j < n and is_outlier[j]:
                j += 1
            start = max(0, i - guard)
            end = min(n, j + guard)
            valid[start:end] = False
            i = j
        else:
            i += 1
    return valid


def detect_onset(x: np.ndarray, window: int = 50, k: float = 10.0,
                  min_run: int = 150, range_ratio_thresh: float = 0.25) -> int:
    """
    Find the sample index where the signal transitions from a flat
    dead/off segment into real active data, via a jump in rolling
    standard deviation, with a second independent check to guard
    against false positives on channels that have NO dead time.

    Two channels can have very different (or no) dead time even on
    the same trigger, and ordinary noise can have a locally quieter
    stretch purely by chance. To avoid misclassifying that as "off",
    this requires BOTH:
      1. A leading run where local std is much below the signal's
         overall (median) std -- not just below one small reference
         window, which is itself a noisy estimate.
      2. That leading run's absolute peak-to-peak range is genuinely
         small relative to the rest of the signal's range -- real
         dead time is close to flat in absolute terms, not merely
         "quieter than what came after."

    If either check fails, returns 0 (assume active from the start).
    This is deliberately conservative: a missed real dead-time region
    is a smaller problem (crop just does nothing) than a false
    positive (crop deletes real data).

    window  : rolling window for the local std estimate
    k       : local std must be at least this many times BELOW the
        global median std to count as a dead-time candidate
    min_run : the candidate quiet region must sustain for at least
        this many samples
    range_ratio_thresh : candidate region's peak-to-peak range,
        divided by the remainder's peak-to-peak range, must be below
        this to confirm it's genuinely flat (not just quieter)

    Returns the onset index, or 0 if no dead time is detected.
    """
    n = len(x)
    roll_std = pd.Series(x).rolling(window, min_periods=1).std().to_numpy()
    global_std = np.nanmedian(roll_std)
    if not np.isfinite(global_std) or global_std == 0:
        return 0
    thresh = global_std / k

    # treat undefined (NaN, from the very first sample) as compatible
    # with "quiet" rather than breaking a genuine leading run
    is_quiet = np.where(np.isnan(roll_std), True, roll_std < thresh)

    if not is_quiet[0]:
        return 0
    j = 0
    while j < n and is_quiet[j]:
        j += 1
    if j < min_run:
        return 0

    lead_range = np.ptp(x[:j])
    rest_range = np.ptp(x[j:]) if j < n else 0.0
    if rest_range == 0 or (lead_range / rest_range) > range_ratio_thresh:
        return 0  # not actually flat in absolute terms -- don't trust it

    return j


def baseline_correct_active_region(x: np.ndarray, onset: int = None,
                                    method: str = "als",
                                    als_lam: float = 1e7,
                                    als_p: float = 0.01,
                                    als_niter: int = 10,
                                    morph_window: int = None,
                                    exclude_outliers: bool = True,
                                    outlier_k: float = 8.0,
                                    outlier_guard: int = 50):
    """
    Crops out everything before `onset` (auto-detected if None),
    baseline-corrects ONLY the active remainder, then reassembles a
    full-length result with the cropped dead-time region forced to
    exactly 0.

    exclude_outliers: if True, also runs detect_amplitude_outliers on
        the active slice BEFORE fitting. This matters because onset
        detection triggers on any std jump -- including a transient
        artifact that happens to sit right at the on/off boundary --
        so the cropped active region can still start with a spike
        that would otherwise drag the baseline fit around. For ALS,
        flagged samples get zero weight (never influence the curve).
        For morphological opening, which has no weighting mechanism,
        flagged samples are linearly interpolated over before the
        rolling min/max pass, so the erosion/dilation window doesn't
        get contaminated by the spike's extreme value. Either way,
        the returned _corrected signal still uses the true (non
        -interpolated) x, so real dips/spikes still show up in the
        output -- only the baseline *estimate* is protected from them.

    Returns (baseline_full, corrected_full), both length len(x).
    """
    n = len(x)
    if onset is None:
        onset = detect_onset(x)
    active = x[onset:]

    outlier_mask = None
    if exclude_outliers:
        outlier_mask = detect_amplitude_outliers(active, k=outlier_k,
                                                  guard=outlier_guard)

    if method == "als":
        baseline_active = als_baseline(active, lam=als_lam, p=als_p,
                                        niter=als_niter, mask=outlier_mask)
    elif method == "morphological":
        window = morph_window if morph_window is not None else 901
        fit_input = active
        if outlier_mask is not None and not outlier_mask.all():
            despiked = pd.Series(active).copy()
            despiked[~outlier_mask] = np.nan
            fit_input = despiked.interpolate(limit_direction="both").to_numpy()
        baseline_active = morphological_baseline(fit_input, window)
    else:
        raise ValueError("method must be 'als' or 'morphological'")

    baseline_full = np.zeros(n)
    corrected_full = np.zeros(n)
    baseline_full[onset:] = baseline_active
    corrected_full[onset:] = active - baseline_active
    return baseline_full, corrected_full


def remove_baseline(df: pd.DataFrame, channels=("Ch1", "Ch2"),
                     max_peak_width_samples: int = 600,
                     method: str = "morphological",
                     als_lam: float = 1e7, als_p: float = 0.01,
                     smooth_baseline: bool = True,
                     valid_mask: np.ndarray = None,
                     zero_invalid: bool = True,
                     crop_dead_time: bool = False,
                     onset: int = None,
                     exclude_outliers: bool = True,
                     outlier_k: float = 8.0,
                     outlier_guard: int = 50) -> pd.DataFrame:
    """
    Adds `<channel>_baseline` and `<channel>_corrected` columns to a
    copy of df.

    method: 'morphological' or 'als'
    smooth_baseline: if True and method='morphological', lightly
        smooths the (already peak-free) baseline afterwards with a
        moving average to remove blocky artifacts.
    valid_mask: optional boolean array (see detect_zero_padding /
        detect_amplitude_outliers). Ignored if crop_dead_time=True.
    zero_invalid: if True and valid_mask is given, forces _corrected
        to exactly 0 wherever valid_mask is False.
    crop_dead_time: if True, use baseline_correct_active_region
        instead of weight-masking -- crops the dead-time prefix out
        before fitting, and zero-fills it afterward. Preferred when
        the dead-time duration varies per record and/or the active
        region is itself large-amplitude/fast-oscillating (masking in
        place tends to leave a boundary transient in that case).
    onset: optional fixed onset index shared across channels, passed
        through to baseline_correct_active_region. Auto-detected per
        channel if None.
    """
    out = df.copy()
    window = int(max_peak_width_samples * 1.5)  # margin above widest peak

    for ch in channels:
        x = out[ch].to_numpy(dtype=float)

        if crop_dead_time:
            baseline, corrected = baseline_correct_active_region(
                x, onset=onset, method=method,
                als_lam=als_lam, als_p=als_p, morph_window=window,
                exclude_outliers=exclude_outliers,
                outlier_k=outlier_k, outlier_guard=outlier_guard,
            )
        else:
            if method == "morphological":
                baseline = morphological_baseline(x, window)
                if smooth_baseline:
                    baseline = (
                        pd.Series(baseline)
                        .rolling(window // 3, center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )
            elif method == "als":
                baseline = als_baseline(x, lam=als_lam, p=als_p, niter=10,
                                         mask=valid_mask)
            else:
                raise ValueError("method must be 'morphological' or 'als'")

            corrected = x - baseline
            if valid_mask is not None and zero_invalid:
                corrected = np.where(valid_mask, corrected, 0.0)

        out[f"{ch}_baseline"] = baseline
        out[f"{ch}_corrected"] = corrected

    return out





def build_invalid_mask(ch, zero_min_run=30, zero_guard=20,
                        outlier_k=8.0, outlier_guard=20,
                        flat_window=50, flat_std_thresh=0.05, flat_min_run=100):
    valid_zero = detect_zero_padding(ch, min_run=zero_min_run, guard=zero_guard)
    valid_outlier = detect_amplitude_outliers(ch, k=outlier_k, guard=outlier_guard)
    valid_flat = detect_flat_regions(ch, window=flat_window,
                                      std_thresh=flat_std_thresh, min_run=flat_min_run)
    return ~(valid_zero & valid_outlier & valid_flat)



def compute_background_std(ch, center, invalid_mask,
                            peak_exclude_half_width=300,
                            edge_margin=50, min_valid=200):
    """
    Robust background std using nearly the whole window as the
    candidate pool -- everything except a region around the peak and
    a small margin at the very edges (where ALS boundary artifacts
    are most likely to sit) -- rather than two small 100-sample
    flanks. A short residual edge artifact becomes a tiny minority of
    a large pool and gets sigma-clipped away, instead of potentially
    dominating a small one.

    Returns (baseline, std) or (np.nan, np.nan) if too few valid
    samples remain -- caller MUST check for nan before using this.
    """
    L = len(ch)
    idx = np.arange(L)
    candidate = (
        (np.abs(idx - center) > peak_exclude_half_width) &
        (idx >= edge_margin) & (idx < L - edge_margin) &
        (~invalid_mask[idx])
    )
    valid_idx = idx[candidate]
    if len(valid_idx) < min_valid:
        return np.nan, np.nan
    return sigma_clipped_baseline_std(ch[valid_idx])


def ALS_BGremoved_Ch2scaled(X, progress=True):
    center = X.shape[-1] // 2
    X_scaled = np.empty_like(X)  # ALS-corrected + Ch2-scaled windows, fed to the CNN

    n_scaling_failed = 0
    for i in tqdm(range(X.shape[0]), disable=not progress):
        X_df = pd.DataFrame(X[i].T, columns=["Ch1", "Ch2"])

        BG_subtracted_ALS = remove_baseline(X_df, method="als", crop_dead_time=True)

        ch1 = BG_subtracted_ALS["Ch1_corrected"].to_numpy()
        invalid_mask1 = build_invalid_mask(ch1)
        ch1_baseline, std_ch1 = compute_background_std(ch1, center, invalid_mask1)

        ch2 = BG_subtracted_ALS["Ch2_corrected"].to_numpy()
        invalid_mask2 = build_invalid_mask(ch2)
        ch2_baseline, std_ch2 = compute_background_std(ch2, center, invalid_mask2)

        if np.isnan(std_ch1) or np.isnan(std_ch2) or std_ch2 < 1e-9:
            # not enough reliable background to scale -- don't guess, flag it
            ch2_scaled = ch2.copy()
            scaling_failed = True
        else:
            ch2_scaled = ch2 * (std_ch1 / std_ch2)
            scaling_failed = False

        if scaling_failed:
            n_scaling_failed += 1

        X_scaled[i, 0] = ch1
        X_scaled[i, 1] = ch2_scaled

    return X_scaled, n_scaling_failed


