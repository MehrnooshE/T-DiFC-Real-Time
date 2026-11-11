"""
Pipeline parameters (Methods S1.1-S1.2), copied from the fusion pipeline's
scripts/parameters.py: acquisition, peak detection and the classifier cut-off.
Used by peak_detection.py, scan_preprocessing.py, streaming.py and the app.
"""
fs = 2000                                   # sampling rate (Hz)
background_sliding_window = 2 * fs + 1      # 2 s rolling-median background window (samples)
rel_thresh = [4.5, 4.5]                     # adaptive peak threshold (x background std), Ch1 / Ch2
min_peak = [0, 0]
max_peak = [100000, 100000]
prominence_factor = 1
narrow_win_len = 200                        # Ch1 candidates closer than this (100 ms) are merged

# ---- classifier ----
cut_off = 0.6                               # 5-seed mean probability >= cut_off -> detected peak
