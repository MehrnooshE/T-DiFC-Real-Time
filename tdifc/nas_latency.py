"""
DAQ laptop -> NAS latency from the chunk files alone (the app's Latency tab).

For every chunk file the app has analysed (recorded while it was loaded: no extra reads):
  Created on   the save time MATLAB writes into the .mat header (DAQ laptop clock, whole seconds),
               from the header scipy's loadmat returns (__header__)
  NAS created  the file's creation time as the file server reports it (NAS clock, sub-second),
               from the os.stat the loader makes anyway
  data time    the end of the chunk's data in recording time (its time vector, from the app's
               chunk table)

Analysis (no clock-drift correction):
  - "Created on" steps vs the data steps (to the header's 1 s); long chunks (more data than usual)
  - the time between files arriving on the NAS
  - delay_i = NAS created_i - (Created on_ref + data time since the ref chunk) - offset, with ONE
    constant offset chosen so that the fastest file has delay 0. The fixed part (how long the
    fastest file took) cannot be separated from the clock offset; a drift between the two clocks
    shows as a slope and a clock jump as a step (not corrected here).
"""
import datetime as dt
import re

import numpy as np

BURST_FRACTION = 0.5       # a NAS step shorter than this share of the data step: files arrived in a burst
LONG_CHUNK_FACTOR = 1.05   # a chunk with more data than this x the usual length is a long chunk
_HEADER_TIME = re.compile(rb"Created on: +([A-Z][a-z]{2} +[A-Z][a-z]{2} +\d+ +\d+:\d\d:\d\d +\d{4})")


def header_time(header):
    """'Created on' of a MATLAB v5 .mat header (bytes, as loadmat's __header__) -> epoch s, read in
    this computer's time zone; None if absent."""
    if not isinstance(header, (bytes, bytearray)):
        return None
    m = _HEADER_TIME.search(header)
    if not m:
        return None
    try:
        return dt.datetime.strptime(re.sub(r" +", " ", m.group(1).decode("ascii")), "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None


def analyse(times, chunks):
    """times: {idx: (path, Created on, NAS created)}; chunks: {idx: (fname, data start s, data end s)}.
    Returns a dict of per-chunk arrays and summary numbers, or None if fewer than 2 files are known."""
    idx = np.array(sorted(k for k in times if k in chunks and times[k][2] is not None), dtype=int)
    if len(idx) < 2:
        return None
    H = np.array([np.nan if times[k][1] is None else times[k][1] for k in idx], float)
    N = np.array([times[k][2] for k in idx], float)
    t0 = np.array([chunks[k][1] for k in idx], float)
    t1 = np.array([chunks[k][2] for k in idx], float)
    length = t1 - t0
    usual = float(np.median(length))
    long_ = length > LONG_CHUNK_FACTOR * usual
    out = dict(idx=idx, hours=t1 / 3600, created_on=H, nas=N, data_end=t1, length=length, usual_len=usual,
               long_chunks=idx[long_].tolist(), n_files=len(idx))
    # steps between consecutive chunk numbers (a missing chunk breaks the pair)
    nxt = np.where(np.diff(idx) == 1)[0]
    d_data = t1[nxt + 1] - t1[nxt]
    d_nas = N[nxt + 1] - N[nxt]
    out.update(step_idx=idx[nxt + 1], step_hours=t1[nxt + 1] / 3600, nas_step=d_nas, data_step=d_data,
               nas_n_ms=(d_nas - d_data) * 1e3)
    out["bursts"] = int((d_nas < BURST_FRACTION * d_data).sum())
    out["shortest_nas_step"] = float(d_nas.min()) if len(d_nas) else None
    hd = H[nxt + 1] - H[nxt]
    ok = np.isfinite(hd)
    out["header_steps"] = int(ok.sum())
    out["header_on_time"] = int((np.abs(hd[ok] - d_data[ok]) < 1.0).sum())       # within the 1-s resolution
    # delay with one constant offset (zero = the fastest file)
    have = np.isfinite(H)
    if have.any():
        r = int(np.argmax(have))                                  # reference: the first file with a header
        raw = N - (H[r] + (t1 - t1[r]))
        k0 = int(np.nanargmin(raw))
        offset = float(raw[k0])
        delay = (raw - offset) * 1e3
        out.update(delay_ms=delay, offset_s=offset, fastest=int(idx[k0]), ref=int(idx[r]))
    else:
        out.update(delay_ms=None, offset_s=None)
    return out


def write_csv(path, a):
    """Per-chunk table of an analyse() result."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk", "data_end_s", "data_length_s", "created_on_epoch_s", "nas_created_epoch_s", "delay_ms"])
        for i in range(a["n_files"]):
            d = a["delay_ms"][i] if a["delay_ms"] is not None else np.nan
            w.writerow([int(a["idx"][i]), f"{a['data_end'][i]:.4f}", f"{a['length'][i]:.4f}",
                        "" if not np.isfinite(a["created_on"][i]) else f"{a['created_on'][i]:.0f}",
                        f"{a['nas'][i]:.6f}", "" if not np.isfinite(d) else f"{d:.3f}"])
