"""
peak_detection.py
-----------------
Laser-ON segmentation, rolling-median background removal and adaptive
threshold peak detection (Methods S1.1-S1.2).

Clean subset of the original PreProcess module, containing only the
functions used by the pulsed-stitched / regular preprocessing pipeline.

Changes vs. the original PreProcess:
  * PreProcessing no longer normalizes the signal:
      - the Ch2 std-matching rescale was removed
      - the min-max (0..1) normalization was removed
    It now returns the background-subtracted Ch1 / Ch2 (original units),
    the detected+filtered Ch1 peak indices, and the (un-normalized)
    threshold curve.
  * pulse_start_end_index has a safeguard so the short-pulse zeroing
    can never index out of bounds.

Everything else is kept identical to the original implementation.
"""
import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import find_peaks


from . import parameters as var

# distance (in samples) below which two Ch1 peaks are merged, keeping the taller
distance_threshold = var.narrow_win_len


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_scan_data(file_path):
    """Load a .mat scan into a DataFrame with columns ['Ch1', 'Ch2']."""
    try:
        mat = scipy.io.loadmat(file_path)
        if 'data' in mat:
            data = pd.DataFrame(mat['data'], columns=['Ch1', 'Ch2'])
        else:
            print(f"'data' key not found in {file_path}")
            return None
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# ----------------------------------------------------------------------
# Otsu threshold (used to separate ON vs OFF regions of the pulse train)
# ----------------------------------------------------------------------
def OtsuThreshold(data, bins_num):
    # Get the image histogram
    hist, bin_edges = np.histogram(data, bins=bins_num)

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


# ----------------------------------------------------------------------
# Pulse ON/OFF boundary detection
# ----------------------------------------------------------------------
def pulse_start_end_index(data, pulse_cutoff, fs, background_sliding_window, on_overshoot, off_overshoot):

    pulse_start_idx = np.where(np.diff((data['Ch1'] > pulse_cutoff).astype(int)) == 1)[0]
    pulse_stop_idx = np.where(np.diff((data['Ch1'] > pulse_cutoff).astype(int)) == -1)[0]

    # Zero out pulses that are too short to be real ON regions.
    # Safeguard: clamp the widened window to valid [0, len(data)] bounds so
    # a pulse near the start/end can't wrap around and zero the wrong region.
    for start, end in zip(pulse_start_idx, pulse_stop_idx):
        if (end - start) < background_sliding_window:
            safe_start = max(0, start - fs)
            safe_end = min(len(data), end + fs)
            data.iloc[safe_start:safe_end] = 0

    pulse_start_idx = np.where(np.diff((data['Ch1'] > pulse_cutoff).astype(int)) == 1)[0] + on_overshoot
    pulse_stop_idx = np.where(np.diff((data['Ch1'] > pulse_cutoff).astype(int)) == -1)[0] - off_overshoot

    if (pulse_start_idx[0] > pulse_stop_idx[0]) and (len(pulse_start_idx) == len(pulse_stop_idx)):
        pulse_start_idx = np.insert(pulse_start_idx, 0, 0)
        pulse_stop_idx = np.insert(pulse_stop_idx, len(pulse_stop_idx), len(data))

    if len(pulse_start_idx) > len(pulse_stop_idx):
        pulse_stop_idx = np.insert(pulse_stop_idx, len(pulse_stop_idx), len(data))

    if len(pulse_start_idx) < len(pulse_stop_idx):
        pulse_start_idx = np.insert(pulse_start_idx, 0, 0)

    return pulse_start_idx, pulse_stop_idx


# ----------------------------------------------------------------------
# Adaptive-threshold peak detection on a background-subtracted channel
# ----------------------------------------------------------------------
def pre_proc_ch(data_bs, fs, rel_thresh, min_peak, max_peak, prominence_factor):
    # This function pre-processes each data channel according to user-defined
    # parameters and finds peak candidates based on an adaptive threshold method.

    # Turn off warnings
    np.seterr(all='ignore')

    snr_win_length = 1  # window length (in seconds) for sliding estimate of background standard deviation
    mov_thresh_int = 2#60  # analysis interval in seconds for the adaptive threshold
    sd_alpha = 0.5  # smoothing factor for standard-deviation estimation

    sig_std = data_bs.rolling(center=True, window=snr_win_length * fs).std().bfill().ffill().to_numpy()

    std_smooth = np.zeros(len(sig_std))
    std_smooth[0] = sig_std[0]
    std_int = np.zeros(len(std_smooth))

    for indx in range(1, len(std_smooth)):
        if sig_std[indx] > 1.05 * std_smooth[indx - 1]:
            std_smooth[indx] = std_smooth[indx - 1]
        else:
            std_smooth[indx] = sd_alpha * std_smooth[indx - 1] + (1 - sd_alpha) * sig_std[indx]
            std_int[indx] = 100

    s_dev = np.mean(std_smooth)

    # Generate a moving detection threshold based on the computed standard
    # deviation estimation and the user-defined PFA using the inverse-Q function
    seg_length = int(mov_thresh_int * fs)  # length of segment to consider in samples
    scan_length_mins = int(np.floor(len(std_smooth) / seg_length))

    if scan_length_mins >= 1:
        moving_threshold = np.zeros_like(std_smooth)

        for k in range(0, scan_length_mins):
            moving_threshold[k * seg_length:(k + 1) * seg_length] = \
                np.mean(std_smooth[k * seg_length:(k + 1) * seg_length]) * rel_thresh

        if (scan_length_mins * seg_length) < len(moving_threshold):
            moving_threshold[scan_length_mins * seg_length:] = \
                np.mean(std_smooth[scan_length_mins * seg_length:]) * rel_thresh
    else:
        moving_threshold = np.mean(std_smooth) * rel_thresh

    moving_threshold = np.maximum(moving_threshold, min_peak)
    moving_threshold = np.minimum(moving_threshold, max_peak)

    pks, pks_heights = [], []
    ch_info = {}

    # Search for peaks using the adaptive threshold in each interval
    if scan_length_mins >= 1:

        thresh_curve = np.zeros_like(data_bs)

        for pk_indx in range(0, scan_length_mins):

            if pk_indx != scan_length_mins:
                start_idx = pk_indx * seg_length
                end_idx = (pk_indx + 1) * seg_length + 1
            else:
                start_idx = pk_indx * seg_length + 1
                end_idx = None

            peak_th = moving_threshold[start_idx]

            data_bs_interval = data_bs[start_idx:end_idx]
            peaks, peaks_properties = find_peaks(data_bs_interval,
                                                 height=peak_th,
                                                 prominence=peak_th / prominence_factor)

            thresh_curve[start_idx:end_idx] = peak_th

            # Update pks and locs data
            peaks = [p + start_idx for p in peaks]
            pks.extend(peaks)

        if (scan_length_mins * seg_length + 1) < len(data_bs):
            start_idx = scan_length_mins * seg_length + 1

            peak_th = moving_threshold[start_idx]

            data_bs_interval = data_bs[start_idx:]
            peaks, peaks_properties = find_peaks(data_bs_interval,
                                                 height=peak_th,
                                                 prominence=peak_th / prominence_factor)

            thresh_curve[start_idx:] = peak_th

            # Update pks and locs data
            peaks = [p + start_idx for p in peaks]
            pks.extend(peaks)

    else:
        peak_th = moving_threshold
        peaks, peaks_properties = find_peaks(data_bs,
                                             height=peak_th,
                                             prominence=peak_th / prominence_factor)

        thresh_curve = np.full_like(data_bs, peak_th)
        pks.extend(peaks)

    ch_info['peaks'] = pks
    ch_info['thresh_curve'] = thresh_curve

    return s_dev, ch_info


# ----------------------------------------------------------------------
# Merge peaks that are closer than distance_threshold, keeping the taller
# ----------------------------------------------------------------------
def filter_peaks(time_series, peak_indexes, distance_threshold):

    differences = np.diff(peak_indexes)
    small_diff = np.where(differences < distance_threshold)[0]
    small_diff_next = small_diff + 1

    while (len(small_diff) > 0):
        differences = np.diff(peak_indexes)
        small_diff = np.where(differences < distance_threshold)[0]
        small_diff_next = small_diff + 1

        close_indexes = peak_indexes[small_diff]
        close_indexes_next = peak_indexes[small_diff_next]

        close_peaks = time_series[close_indexes]
        close_peaks_next = time_series[close_indexes_next]

        stacked_close_peaks = np.vstack((close_peaks, close_peaks_next))
        stacked_close_indexes = np.vstack((close_indexes, close_indexes_next))
        max_args = np.argmax(stacked_close_peaks, axis=0)
        max_peaks_indexes = stacked_close_indexes[max_args, range(stacked_close_indexes.shape[1])]

        indexes_to_remove = np.setdiff1d(stacked_close_indexes.flatten(), max_peaks_indexes.flatten())
        filtered_peaks_indexes = np.setdiff1d(peak_indexes, indexes_to_remove)
        peak_indexes = filtered_peaks_indexes

    return peak_indexes


# ----------------------------------------------------------------------
# Full preprocessing for one (sub-)signal: background subtraction + peak
# detection. NO Ch2 rescale and NO normalization (removed on purpose).
# ----------------------------------------------------------------------
def PreProcessing(data, fs, background_sliding_window, rel_thresh, min_peak, max_peak, prominence_factor):
    # Background subtraction (rolling median) for both channels
    data_bs1 = data['Ch1'].rolling(window=background_sliding_window, center=True).median().bfill().ffill()
    data_bs1 = data['Ch1'] - data_bs1

    data_bs2 = data['Ch2'].rolling(window=background_sliding_window, center=True).median().bfill().ffill()
    data_bs2 = data['Ch2'] - data_bs2

    # Adaptive-threshold peak detection on Ch1
    (s_dev1, ch1_info) = pre_proc_ch(data_bs1, fs, rel_thresh[0], min_peak[0], max_peak[0], prominence_factor)

    # NOTE: the Ch2 std-matching rescale and the min-max normalization that
    # used to live here were intentionally removed. The returned signals are
    # the raw background-subtracted traces in their original units.

    filtered_peaks_1 = filter_peaks(data_bs1.to_numpy(), np.array(ch1_info['peaks']), distance_threshold)

    return data_bs1, data_bs2, filtered_peaks_1, ch1_info['thresh_curve']


def load_and_concat(file_paths):
    """Load each .mat file into a DataFrame and concatenate in order.
    Missing/unloadable files are skipped with a warning rather than
    silently guessed at."""
    frames = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"    [MISSING] {fp} -- skipping this file")
            continue
        d = load_scan_data(fp)
        if d is None:
            print(f"    [SKIP] could not load {fp}")
            continue
        frames.append(d)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
