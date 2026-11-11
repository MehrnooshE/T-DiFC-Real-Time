# t-DiFC-Real-Time

Real-time peak (CTC) detection for t-DiFC with the time + spectral **`fusion_lr1e-4_wd_ls`** classifier of the t-DiFC clearance-cohort pipeline (`t-DiFC-clearance cohort/fusion_lr1e-4_wd_ls_pipeline`), **retrained in PyTorch** (no TensorFlow / Keras). The app watches the DiDAQ output folder, preprocesses the signal as it arrives exactly as the pipeline does, scores every peak candidate with the 5-seed ensemble and shows the live trace, the barcode of detected peaks and the 1-s window of any candidate.

The folder is standalone: the code, the labeled training data, the trained model and the Python environment are all in it. Nothing outside it is imported or read (except the recordings the app watches).

## Layout

```
├── tdifc_app.py               the real-time app (GUI), run from the terminal
├── tdifc_app_desktop.py       the same app with a start-up window, for the macOS app (see Desktop app)
├── tdifc_app_desktop.spec     PyInstaller recipe of "t-DiFC Live.app"
├── build_mac.sh               builds dist/t-DiFC Live.app and the zip to share
├── requirements-build.txt     extra packages for the build (PyInstaller)
├── train_fusion_torch.py      trains the 5 seeds in PyTorch -> models/
├── evaluate_fusion_torch.py   val / test metrics at the 0.6 cut-off; comparison with the Keras seeds
├── check_streaming.py         real-time vs offline preprocessing on a recorded file
├── simulate_acquisition.py    replays a recorded scan as DiDAQ 10 s chunks (test the app without the DAQ)
├── tdifc/                     library
│   ├── parameters.py          sampling rate, peak-detection parameters, cut-off (copied from the pipeline)
│   ├── peak_detection.py      Otsu, laser-ON segments, background removal, peak detection (verbatim copy)
│   ├── signal_functions.py    ALS baseline + Ch2 scaling for the spectral input (verbatim copy, trimmed)
│   ├── scan_preprocessing.py  per-file preprocessing and windowing (copy of full_scan_preprocessing.py)
│   ├── spectral.py            spectral input: centre spectrum, standardizer (copy of fusion_model.py, no TF)
│   ├── model.py               PyTorch architecture; FusionEnsemble (loads and scores)
│   ├── streaming.py           causal (real-time) version of the preprocessing
│   ├── metrics.py             metrics, scan-level bootstrap (copy of common.py)
│   └── paths.py               folder paths, split loading
├── data/
│   ├── train-val-test/Joint-Standardization-Augmented/   labeled windows (copy of the pipeline's splits)
│   └── train-val-test-als-cache/                         their ALS windows (rebuilt if deleted, ~4 min)
├── models/
│   ├── fusion_lr1e-4_wd_ls_torch.pt     the 5 seeds + spectral standardizer + cut-off (what the app loads)
│   ├── training_history.csv             per-epoch train / val BCE and lr; restored epoch flagged
│   ├── predictions_{val,test}.csv       per-seed and 5-seed mean probabilities
│   ├── metrics_at_0.6.csv               Table S2 for the PyTorch model
│   ├── torch_vs_keras.csv               PyTorch vs Keras metrics and agreement
│   └── keras_reference/                 the Keras model's predictions / metrics / history (for comparison)
├── results/<exp_name>/        written by the app: candidates.csv, summary.json
├── logs/                      output of the training, evaluation and streaming checks
├── requirements.txt
└── t-difc-rt-env/             Python environment (not code; recreate any time, see Setup)
```

## Setup

Python 3.11 and `requirements.txt` (NumPy, SciPy, pandas, scikit-learn, tqdm, PyTorch, pyqtgraph, PyQt6). The environment `t-difc-rt-env/` is already created in this folder. To recreate it:

```
cd t-DiFC-Real-Time
/path/to/python3.11 -m venv t-difc-rt-env      # e.g. ~/miniconda3/bin/python3.11
t-difc-rt-env/bin/python -m pip install -r requirements.txt
```

Activate it in every new terminal: `source t-difc-rt-env/bin/activate`, or call `t-difc-rt-env/bin/python` directly.

## Running the app

```
cd t-DiFC-Real-Time
source t-difc-rt-env/bin/activate
python tdifc_app.py --save_dir "/Volumes/Public/Jane/Balbc MM" --exp_name "250723-BCMM1_MM-CFSE"
```

`--save_dir` is the folder that contains `<exp_name>/F1/`, `F2/`, … as DiDAQ writes them: `<exp_name>/F<k>/<exp_name>_F<k>_<n>.mat`, n = 1, 2, …, each a 10 s chunk with `data` (samples × channels, mV) and `time`. Ch1 and Ch2 are the first two columns of F1, F2, … joined in fiber order. Both layouts are recognized from the first chunk: both channels in F1, or Ch1 in F1 and Ch2 in F2 (`--n_fibers` forces it). Chunks already in the folder when the app starts are analysed first as by the offline pipeline (see *Data saved before the app started* below; 4 h take about half a minute), then new ones are analysed in real time, picked up within 1 s. The folder is polled, so a NAS share works (macOS: Finder → Go → Connect to Server → `smb://129.10.224.25/Public`, mounted at `/Volumes/Public`). A missing chunk is skipped after 60 s if later chunks exist; times always follow DiDAQ's `time` vector.

**Any folder of .mat files** (`tdifc_app.py` and the desktop app): the data folder does not need the DiDAQ layout. If it has no `F1/<name>_F1_<n>.mat` chunks, the app reads the folder's own `.mat` files (or those in its `F1` folder) in the order of their names, with the numbers sorted as numbers (`x_2` before `x_10`). Any naming works: `exp_1.mat, exp_2.mat, ...`, `exp_1_F1.mat, ...`, `exp_F1_1.mat, ...`, or one long file without a number. Files with an `F2` tag next to the `F1` ones give Ch2, as in the DiDAQ layout. Only the files with the most common name are used; others (e.g. `notes.mat`) are listed as not used in the start-up window and the log. Files that appear while the app runs are added at the end. From the terminal: `python tdifc_app.py --folder "/path/to/folder"` (or `tdifc_app_desktop.py`); in the desktop app, pick the folder in the start-up window.

**Ch1 in F1, Ch2 in F2**: with *Fiber folders* = `auto`, Ch1 and Ch2 come from F1 if its files have two columns, else Ch2 comes from the matching F2 file. With `2`, Ch1 is the first column of the F1 file and Ch2 the first column of the F2 file, also when the F1 files have more columns. This works for DiDAQ folders and for `F1/` and `F2/` folders with any file names (F2 files with the same name without the F1/F2 tag, else in the same order).

**Laser mode** (`--mode auto`): pulsed needs, besides the pipeline's test, long laser-ON runs and a dark OFF level (OFF median below 25% of the ON median; real pulsed data ~0.3%, a decaying continuous signal 40-100%). The decision is fixed for the rest of the recording once it has stayed the same for 5 min, so a continuous recording is no longer treated as pulsed for stretches of its slow decay.

**Signal y range**: if the signal in the first 30 s goes above the y maximum (2500 mV by default), the trace's limit becomes that maximum + 200 mV (`Y_AUTO_SECS`, `Y_AUTO_MARGIN`).

**Files longer than 10 s** (e.g. the 10-s chunks stitched into one file, named `<exp_name>_F1_1.mat` in `F1/`) are handled too. In `real-time` mode such a file is fed to the real-time algorithm in 10-s slices, as if DiDAQ had written it chunk by chunk (same candidates and probabilities as the 10-s chunks; pause works between slices). In the offline modes it is cut into the usual 2-h blocks. A gap inside a file's `time` vector (a step of more than 50 ms, e.g. a DAQ pause in stitched data) starts a new stretch with its own start time, so every peak keeps its recording time; a time vector that runs backwards is logged and the clock is continued. A file that cannot be used (no `data` variable, a MATLAB v7.3 file, only one channel) is skipped as a missing chunk; the reason is logged and shown in red in the status box.

**Analysis mode** (`--analysis`), i.e. what happens to the data already saved when the app starts:

| `--analysis` | Data saved before the start | New data | Use for |
|---|---|---|---|
| `real-time` (default) | real-time algorithm, chunk by chunk from the first chunk (progress bar for the saved part) | real time | results identical to an app that ran from the start |
| `offline-then-real-time` | offline pipeline, 2-h blocks (see *Data saved before the app started*) | real time | a faster start when hours are already saved |
| `offline` | offline pipeline, 2-h blocks, to the very end | not read; the app reports *Offline analysis finished* and stays open for reviewing | a finished experiment |

The mode is shown in the window title and written to `summary.json` (`analysis`) and to `candidates.csv` (`analysed_as` per candidate). On 4 h of BCMM1 with no new data: `offline` 26 s, `offline-then-real-time` 40 s (both 516 candidates, 385 detected), `real-time` 45 s (524 candidates, 384 detected, identical to the real-time analysis from t = 0).

Options: `--mode auto|pulsed|continuous` (laser illumination; default auto, decided from the last 5 min of Ch1), `--out_dir` (default `results/<exp_name>/`), `--cut_off` (default 0.6, from the model file), `--y_min / --y_max` (trace range, mV; default 0–2500), `--model`, `--log_level`. Other settings are at the top of `tdifc_app.py`.

**Window.** Top: the status line (laser mode, time recorded, laser-ON time, candidates and detected peaks so far) and the raw Ch1 / Ch2 trace of the last 60 s; every candidate is highlighted with its probability (red: detected, p ≥ 0.6; grey: rejected) when the view shows 10 min or less (only the candidates in and next to the view are drawn, so panning stays fast however long the recording). The trace is full resolution (every sample) during laser-ON; with the pulsed laser, the laser-OFF gaps are drawn as 0 (their signal is not used by the analysis), which keeps about 4x less data. The time axis reads as a recording clock (h:mm:ss). Drag / scroll to move along time (last hour kept), **A** (bottom left) shows the whole hour; in a view wider than 10 min, click a laser pulse to zoom to it (the pulse ± 2 s, with its candidates highlighted). **Reset view** returns to the newest data and follows it again. **🔍 Magnifier** opens a panel beside the trace (the trace gets narrower) and a 2-s window on the trace that follows the cursor while it is over the trace, and stays where it was when the cursor leaves. Click the trace to stop its live scrolling there; the window keeps following the cursor over the stopped trace (**Reset view** returns to live). The panel shows Ch1 and Ch2 in the window at full resolution, each on its own y-axis that auto-scales to the window. With **fixed y (per pulse)** ticked (under the Magnifier button), each y-axis is instead fixed to the min / max of the laser-ON pulse under the window (without its first / last 25 ms, + 5%), so the scale does not change while the window moves inside a pulse; over a laser-OFF gap it is automatic, with the continuous laser it is set per 60-s block. Candidates in the window are marked with their probability. Showing a candidate (barcode, highlight, review panel) moves the window onto it. **⏸ Pause** stops reading new chunks (after the one in progress); the DAQ keeps saving them, the status line shows how long processing has been paused and how much data is waiting (e.g. *4.0 h of data waiting*), and the window stays usable (review, magnifier, saving). **▶ Resume** catches up on the waiting chunks (about 70x real time) and continues; nothing is skipped, the results are the same as without the pause. Pause also works while the saved data is being analysed offline at start-up: it takes effect at the next chunk read, before a block's preprocessing or between its scoring batches (within a few seconds), the status line then counts everything not analysed yet, and Resume continues the offline analysis where it stopped. Display settings (`DISPLAY_BIN`, `DISPLAY_OFF_AS_ZERO`, history length) are at the top of `tdifc_app.py`. Middle: barcode of all detected peaks since the start (time axis in whole hours), **Save barcode** as PNG. Click a barcode line or a highlight to jump there and show the candidate's 1-s window (background-subtracted Ch1 / Ch2, full resolution) with its time, sample, probability, laser-edge flag and file; **Save clip** as PNG, ✕ closes it.

**Reviewing calls.** The panel right of the clip steps through the candidates of one set: detected peaks, near misses (0.5 ≤ p < cut-off), all rejected, all candidates, or *Reviewed* (the ones you have already judged). **⏮ First / ◀ Prev / Next ▶ / Latest ⏭** move through the set in time order (keys Home, ←, →, End); the counter shows the candidate's position in the set and how many are left to review. For the candidate on screen, **✓ Peak** (Y) or **✗ Artifact** (N) records your verdict; pressing the other button changes it, the same button again clears it. *skip reviewed* leaves out candidates already judged (not in the *Reviewed* set), *next after verdict* moves on automatically. The panel counts what has been reviewed: of the detected peaks, how many you confirmed (a live precision estimate); of the near misses and the other rejected candidates, how many you judged to be peaks (missed peaks). Verdicts go to `results/<exp_name>/review.csv` (see below). The app always uses a light theme, also when macOS is in dark mode.

**Output** (`results/<exp_name>/`; `candidates.csv` and `summary.json` are rewritten from the start each time the app is started, `review.csv` is kept):
- `candidates.csv`: one row per candidate as soon as it is scored: `sample`, `time_s`, `time_h`, `chunk`, `source_file`, `laser_mode`, `at_laser_edge` (≤ 25 ms from a laser ON/OFF edge), `p_fusion_lr1e-4_wd_ls_torch` (5-seed mean), `detected` (p ≥ cut-off), `analysed_as` (`offline`: saved before the app started; `real-time`).
- `review.csv`: one row per verdict, appended, never wiped: `sample`, `time_s`, `time_h`, `source_file`, the model probability, `model_call` (peak / artifact at the cut-off), `operator_label` (peak / artifact; empty = verdict cleared), `reviewed_at`. The last row of a sample is its verdict. After a restart the verdicts are matched back to the candidates by sample number (the analysis is deterministic, so the same candidates come back).
- `summary.json`: hours recorded, laser-ON hours, candidates, detected peaks (all and not at a laser edge), detected peaks per laser-ON hour, and `saved_data_analysed_offline` (chunks, 2-h blocks, the time from which real time took over, seconds taken); updated with every chunk.

**Testing without the DAQ.** Replay a recorded scan into a folder and point the app at it (two terminals):

```
python simulate_acquisition.py --mat "<recording>.mat" --save_dir sim --exp_name TEST --speed 10
python tdifc_app.py --save_dir sim --exp_name TEST
```

`--speed 10` writes a 10 s chunk every second (`0` = all at once); `--layout single` puts both channels in F1; `--start_min`, `--minutes` select part of the recording; several `--mat` files are replayed one after the other; `--first_chunk` continues the numbering of an earlier run (e.g. first write 3 h at `--speed 0` as already-saved data, start the app, then add live chunks).

## Desktop app (macOS)

`tdifc_app_desktop.py` is `tdifc_app.py` with a start-up window, packaged as **t-DiFC Live.app**, so the app can be shared with people who do not use the terminal (no Python or environment needed).

**Build** (on a Mac, in this folder, with the environment of *Setup*):

```
./build_mac.sh
```

It takes about 3 min and writes `dist/t-DiFC Live.app` (~500 MB, PyTorch included) and **`dist/t-DiFC-Live-mac-arm64.zip`** (~180 MB), the file to share (e.g. as a GitHub Release asset or by Teams / OneDrive). A build on Apple silicon runs on Apple-silicon Macs (M1 or later, macOS 12+); build on an Intel Mac for Intel Macs. `build/` and `dist/` are not committed.

**Use.** Unzip, move *t-DiFC Live* to Applications and double-click it. The start-up window asks for:
- **Data**: the experiment folder DiDAQ writes into (the one with `F1`, `F2`, …; choosing `F1` itself also works). The experiment name is read from the chunk files, and the window shows what it found (e.g. *7490 chunk files, about 20.8 h*). A folder that is still empty is fine: the app waits for DiDAQ.
- **Analysis**: real time (default), offline then real time, or offline only (as `--analysis`).
- **Results**: by default `~/Documents/t-DiFC results/<experiment name>/`, with `candidates.csv`, `summary.json`, `review.csv` and the log `tdifc_app.log`.
- **Advanced options**: laser illumination, fiber folders, probability cut-off, signal y-range.

**Start** opens the main window (as `tdifc_app.py`). The choices are remembered for the next start. From the terminal, the desktop version still takes the options of `tdifc_app.py`; with `--save_dir` and `--exp_name` it skips the start-up window.

**First start on another Mac.** The app is not signed with an Apple Developer ID, so macOS blocks it the first time: *"t-DiFC Live" cannot be opened* (or *Apple could not verify…*). Once per Mac:
- macOS 14 and older: right-click (Control-click) the app → **Open** → **Open**.
- macOS 15 and newer: double-click once, then **System Settings → Privacy & Security**, scroll down to *"t-DiFC Live" was blocked* → **Open Anyway**.

macOS then asks once for access to the network volume (NAS) and the Documents folder: allow both. (Signing and notarizing the app with an Apple Developer account removes the warning.)

## How the real-time preprocessing relates to the pipeline

The app runs the pipeline's own functions (`peak_detection.py`, `scan_preprocessing.py`, `spectral.py`, copied) on the data stream (`tdifc/streaming.py`):

- **Pulsed laser** (~10 s ON / ~40 s OFF): laser-ON segments by Otsu thresholding of Ch1, the same 5-sample ON/OFF overshoots, segments shorter than the 2 s background window treated as OFF, and each segment's 2 s rolling-median background removal and adaptive 4.5σ peak detection (candidates < 100 ms apart merged). A segment is processed once it is complete (laser OFF again), i.e. whole, as offline; its candidates appear at the end of the segment, about 10 s after the peak. The only difference from offline: the Otsu threshold is computed from the last 5 min of Ch1 (it follows slow drifts of the laser level) instead of once per recording file.
- **Continuous laser**: the signal from 30 s before the last finalized sample to the newest one is preprocessed as one segment; candidates up to 3 s before the newest sample are finalized (delay ~3–13 s).
- **Windows and scoring** as offline: 2,001-sample window of the background-subtracted signal (OFF gaps = 0) around each candidate, invalid mask, joint standardization; spectral input (ALS + Ch2 scaling, centre ±100 ms, 0–300 Hz); 5-seed mean probability; detected if ≥ 0.6.
- Processing starts after the first minute of data (needed for the pulsed / continuous decision).
- **Data saved before the app started** (app started during an experiment, or restarted): these chunks are analysed as by the offline pipeline, in 2-h blocks (its recording-file length), each with its own pulsed / continuous decision and Otsu threshold (`preprocess_pulsed` / `preprocess_regular`, `build_windows`), and scored in one batch; the display is updated once per block. The status line says which part of the recording is being analysed and the step (e.g. *0:00 - 2:00 h (block 1 of 2) - scoring the candidates*), with a progress bar under it for the current block that starts again for the next block. Unlike the offline files, a block ends in a laser-OFF gap of at least 2 s, so no pulse is cut in two. The real-time analysis then continues from the last laser-OFF gap before the end of the saved data, with the last minutes as context, so no candidate is counted twice or missed. On 4 h of BCMM1 (3.83 h saved, the rest arriving live): 34 s for the saved part; compared with the offline pipeline on the two 2-h files, 516 of 518 candidates in common (the others at the file boundary), the same 385 detected peaks and identical probabilities.

`check_streaming.py "<file>.mat" --score` replays a recording through the real-time path and compares it with the offline per-file preprocessing. On two 2-h pulsed clearance-cohort files (`logs/check_streaming.log`):

| File | Candidates offline / real time / common | Same model window (common) | Same call (common) | Detected offline / real time |
|---|---|---|---|---|
| BCMM1 `_Pulsed_19` | 150 / 160 / 147 | 97% identical | 100% | 72 / 71 |
| BCL12-3-1 `_Pulsed_25` | 61 / 59 / 59 | 97% identical | 100% | 2 / 2 |

The candidates found by only one of the two lie within ~0.5 s of a laser edge: a threshold a few mV different moves a segment's start by a few samples, which changes the peak detection of near-threshold candidates (offline too: it depends on the threshold of the whole file).

## The PyTorch model

`train_fusion_torch.py` is the PyTorch port of the pipeline's `05_train.py`, on the same splits (copied to `data/`):

- **Architecture** (`tdifc/model.py`, same layers as `fusion_model.py`): time branch, 5 × [Conv1d 16/32/64/64/64, k 9/9/7/5/5, BatchNorm, ReLU, MaxPool 2] + global average and max pooling (128-d) on the jointly standardized 2 × 2001 window; spectral branch, 3 × [Conv1d 16/32/32, k 5/5/3, BatchNorm, ReLU (MaxPool 2 ×2)] + Flatten + Dropout 0.3 + Dense 64 on the 61 × 3 centre spectrum; head Dropout 0.3, Dense 32, Dropout 0.3, Dense 1 (sigmoid). Keras defaults kept: Glorot-uniform kernels, zero biases, BatchNorm ε 1e-3, momentum 0.99.
- **Training**: Adam 1e-4, decoupled weight decay 1e-4 on the kernels after every step, label smoothing 0.05, batch 64, ≤ 60 epochs, ReduceLROnPlateau (×0.5, patience 4, min 1e-6) and early stopping (patience 8, best weights restored) on the plain validation BCE, seeds 0–4; spectral input standardized with the training-set statistics. About 3 min on Apple silicon (`--device auto` uses the GPU, MPS; `--device cpu` ~5× slower).
- **Model file** `models/fusion_lr1e-4_wd_ls_torch.pt`: the 5 seeds' weights, the spectral standardizer (so the app needs no training data) and the cut-off; loaded with `torch.load(..., weights_only=True)`.

Test and validation results at the 0.6 cut-off (`python evaluate_fusion_torch.py`; 95% scan-level bootstrap CIs; `models/metrics_at_0.6.csv`, `models/torch_vs_keras.csv`):

| | PyTorch, test | Keras (original), test | PyTorch, val | Keras, val |
|---|---|---|---|---|
| AUC | 0.978 [0.878, 0.990] | 0.980 [0.882, 0.991] | 0.993 [0.969, 0.998] | 0.991 [0.960, 0.997] |
| Sensitivity | 0.848 [0.310, 0.923] | 0.843 [0.297, 0.920] | 0.915 [0.704, 0.950] | 0.872 [0.676, 0.908] |
| Specificity | 0.977 [0.947, 0.997] | 0.986 [0.964, 1.000] | 0.987 [0.968, 1.000] | 0.987 [0.970, 1.000] |
| TP / FP / TN / FN | 367 / 10 / 423 / 66 | 365 / 6 / 427 / 68 | 214 / 3 / 231 / 20 | 204 / 3 / 231 / 30 |

On the same windows the two ensembles agree closely (test: Pearson r 0.996 between probabilities, same call at 0.6 for 99.3% of windows; val: r 0.994, 97.4%). The PyTorch model calls slightly more peaks (test: 2 more true and 4 more false positives); all differences are within the CIs. Retraining is not bit-identical across machines (GPU), so a retrained model gives slightly different numbers.

Retrain / re-evaluate:

```
python train_fusion_torch.py        # overwrites models/fusion_lr1e-4_wd_ls_torch.pt
python evaluate_fusion_torch.py
```
