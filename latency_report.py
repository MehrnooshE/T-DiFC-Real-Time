"""
Latency of the real-time path, stage by stage, from the latency_*.csv files of
`tdifc_app.py --latency` (or, with --history, from the files of a past recording alone).

    event in the signal
      A  waiting for the 10-s chunk to fill            DAQ time vector (exact)
      B  chunk end -> saved on the DAQ laptop          .mat header time (laptop clock, 1 s), as the
                                                       delay beyond the fastest save of the recording
      C  saved on the laptop -> on the NAS (sync)      NAS creation time - header time - (laptop - NAS clock)
      D  on the NAS -> the app finds it                app clock - NAS clock (probe, latency_clock.csv)
      E  loading                                       app clock
      F  preprocessing (+ display data)                app clock
      G  model scoring                                 app clock
      H  queued -> drawn in the window                 app clock
      W  detector wait: the candidate is reported after a later chunk (end of the laser-ON segment
         / 3 s after the peak), in recording time
    event -> window = A + W + (B..H of the chunk after which it was reported)

Clocks. The DAQ laptop, the NAS and this computer each have their own clock (laptop - NAS was
~13 min in July 2025). NAS - this computer is measured by the app (probe file). DAQ laptop - NAS
is given here, either as a number (--laptop_minus_nas) or from a file the laptop writes straight
to the NAS (not into the synced folder) just before or after the run, in PowerShell:

    (Get-Date).ToString("o") | Set-Content -Encoding ascii "\\\\129.10.224.25\\Public\\laptop_clock.txt"

(--laptop_clock_file /Volumes/Public/laptop_clock.txt: its content is the laptop's time, its
creation time the NAS's). Without it, C cannot be separated from the clock offset: the report then
takes the offset that makes the fastest sync 0 s, so C and the totals are LOWER BOUNDS. The same is
done for D when the probe could not write to the NAS. The header time has 1-s resolution and the
fastest save is taken as instantaneous (it is at most ~0.2 s + the write), so every time that
involves the DAQ laptop is uncertain by about +-0.5 s and slightly low.

Usage:
    python latency_report.py results/<exp_name> [--laptop_minus_nas 788.5 | --laptop_clock_file F]
    python latency_report.py --history "/Volumes/Public/Jane/Balbc MM/250723-BCMM1_MM-CFSE" --every 10
Writes latency_report_chunks.csv, latency_report_candidates.csv and latency_report_summary.csv
next to the input (history: latency_history.csv in --out, default results/<exp_name>/).
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

from tdifc import paths
from tdifc.latency import created_time, mat_header_time

ROLL = 361            # chunks (~1 h) of the rolling minimum that gives the fastest save
HALF_S = 0.5          # header times are whole seconds rounded down: +0.5 s on average
STARTUP_S = 60        # StreamProcessor(min_start_s): no candidate is reported before this much signal


def read_laptop_clock_file(path):
    """(laptop - NAS clock) from a file the DAQ laptop wrote to the NAS: content = laptop time."""
    raw = open(path, "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    t_laptop = dt.datetime.fromisoformat(txt.strip().splitlines()[0].strip()).timestamp()
    return t_laptop - created_time(path)


def stats(s):
    s = pd.Series(s, dtype=float).dropna()
    if not len(s):
        return dict(n=0)
    return dict(n=len(s), median=s.median(), mean=s.mean(), p5=s.quantile(0.05), p95=s.quantile(0.95),
                min=s.min(), max=s.max())


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'stage':<46}{'n':>7}{'median':>9}{'p5':>8}{'p95':>8}{'max':>9}   (s)")
    for name, st, note in rows:
        if not st.get("n"):
            print(f"  {name:<46}{0:>7}   -{('   ' + note) if note else ''}")
            continue
        print(f"  {name:<46}{st['n']:>7}{st['median']:>9.2f}{st['p5']:>8.2f}{st['p95']:>8.2f}{st['max']:>9.2f}"
              f"{('   ' + note) if note else ''}")


def save_base(x):
    """Rolling minimum of (header time - chunk end in DAQ time): session start + the fastest save."""
    return pd.Series(x).rolling(ROLL, center=True, min_periods=1).min().to_numpy()


# ───────────────────────────────────────────────────────── app run (--latency) ───
def report_run(res_dir, laptop_minus_nas=None):
    ch = pd.read_csv(os.path.join(res_dir, "latency_chunks.csv")).sort_values("chunk").reset_index(drop=True)
    cand = pd.read_csv(os.path.join(res_dir, "latency_candidates.csv"))
    clock = pd.read_csv(os.path.join(res_dir, "latency_clock.csv"))
    if ch.empty:
        sys.exit(f"{res_dir}: no chunks in latency_chunks.csv")
    live = ch["live"] == 1
    print(f"{res_dir}: {len(ch)} chunks ({int(live.sum())} arrived while the app was running), "
          f"{len(cand)} candidates")

    # ---- clocks
    mac_ntp = clock.loc[clock["what"] == "mac_minus_ntp", "offset_s"].dropna()
    probes = clock.loc[clock["what"] == "nas_minus_mac", "offset_s"].dropna()
    notes = []
    if len(probes):
        nas_mac, nas_mac_how = float(probes.median()), f"probe (n = {len(probes)}, spread {np.ptp(probes):.2f} s)"
    else:
        nas_mac = float((ch.loc[live, "nas_created"] - ch.loc[live, "seen"]).max())
        nas_mac_how = "BOUND: no probe - the fastest pick-up taken as 0 s"
        notes.append("D is a lower bound (no NAS clock probe)")
    H = ch["laptop_saved"] + HALF_S
    if laptop_minus_nas is None:
        laptop_minus_nas = float((H - ch["nas_created"])[live].max())
        lap_how = "BOUND: not given - the fastest sync taken as 0 s"
        notes.append("C and the totals are lower bounds (laptop clock not measured)")
    else:
        lap_how = "given"
    print("\nClocks (+ = ahead)")
    if len(mac_ntp):
        print(f"  this computer - NTP     {mac_ntp.iloc[0]:+10.3f} s")
    print(f"  NAS - this computer     {nas_mac:+10.3f} s   {nas_mac_how}")
    print(f"  DAQ laptop - NAS        {laptop_minus_nas:+10.3f} s   {lap_how}")
    mac_ntp0 = float(mac_ntp.iloc[0]) if len(mac_ntp) else 0.0
    print(f"  => NAS - true time      {nas_mac + mac_ntp0:+10.3f} s")
    print(f"  => DAQ laptop - true    {laptop_minus_nas + nas_mac + mac_ntp0:+10.3f} s")
    if abs(laptop_minus_nas) > 1800:
        print("  !! more than 30 min: check that the DAQ laptop and this computer use the same time zone")

    # ---- per chunk (seconds); every time converted to this computer's clock
    x = ch["laptop_saved"] - ch["t_last_s"]                    # laptop clock - DAQ time
    base = save_base(x.to_numpy())                            # session start + fastest save
    lap_to_mac = -(laptop_minus_nas + nas_mac)                 # add to a laptop time -> this computer
    chunk_end = base + HALF_S + ch["t_last_s"] + lap_to_mac
    saved = H + lap_to_mac
    on_nas = ch["nas_created"] - nas_mac
    ch["B_save"] = saved - chunk_end
    ch["C_sync"] = on_nas - saved
    ch["D_pickup"] = ch["seen"] - on_nas
    ch["E_load"] = ch["loaded"] - ch["seen"]
    ch["F_preprocess"] = ch["processed"] - ch["loaded"]
    ch["G_score"] = ch["scored"] - ch["processed"]
    ch["H_to_window"] = ch["drawn"] - ch["scored"]
    ch["chunk_end_to_window"] = ch["drawn"] - chunk_end
    ch["chunk_end_wall"] = chunk_end
    ch.to_csv(os.path.join(res_dir, "latency_report_chunks.csv"), index=False)

    lv = ch[live]
    with_c = lv[lv["n_candidates"] > 0]
    rows = [("B  chunk end -> saved on laptop (beyond fastest)", stats(lv["B_save"]), "+-0.5 s"),
            ("C  laptop -> NAS (sync)", stats(lv["C_sync"]), "+-0.5 s"),
            ("D  NAS -> app finds it (1 s polling)", stats(lv["D_pickup"]), ""),
            ("E  load", stats(lv["E_load"]), ""),
            ("F  preprocess", stats(lv["F_preprocess"]), ""),
            ("G  model scoring (all chunks)", stats(lv["G_score"]), ""),
            ("G  model scoring (chunks with candidates)", stats(with_c["G_score"]), ""),
            ("H  queued -> drawn", stats(lv["H_to_window"]), ""),
            ("chunk end -> drawn in the window", stats(lv["chunk_end_to_window"]), "+-0.5 s")]
    print_table("Per chunk (live chunks only)", rows)

    # ---- per candidate
    summary = [dict(stage=n, **s) for n, s, _ in rows]
    if len(cand):
        t_last = ch.drop_duplicates("chunk").set_index("chunk")["t_last_s"]
        info = ch.set_index("lat_id")
        cand = cand[cand["lat_id"].isin(info.index[info["live"] == 1])].copy()
        t0 = float(ch["t_first_s"].min())                     # the detector starts after STARTUP_S of signal
        early = cand["time_s"] < t0 + STARTUP_S
        if early.any():
            notes.append(f"{int(early.sum())} candidate(s) in the first {STARTUP_S} s left out of the "
                         f"candidate statistics (the detector first needs {STARTUP_S} s of signal to decide "
                         f"the laser mode); they are in latency_report_candidates.csv")
        cand["startup"] = early.astype(int)
        cand["A_fill"] = cand["chunk_of_peak"].map(t_last) - cand["time_s"]
        cand["W_detector"] = cand["chunk_reported"].map(t_last) - cand["chunk_of_peak"].map(t_last)
        rep = info.loc[cand["lat_id"]]
        cand["event_to_window"] = (rep["drawn"].to_numpy() - rep["chunk_end_wall"].to_numpy()
                                   + cand["chunk_reported"].map(t_last).to_numpy() - cand["time_s"].to_numpy())
        cand.to_csv(os.path.join(res_dir, "latency_report_candidates.csv"), index=False)
        cand = cand[cand["startup"] == 0]
        det = cand[cand["detected"] == 1]
        crow = [("A  peak -> end of its chunk (recording time)", stats(cand["A_fill"]), ""),
                ("W  detector wait (later chunk), recording time", stats(cand["W_detector"]),
                 f"{(cand['W_detector'] > 0).mean():.0%} wait"),
                ("event -> drawn, all candidates", stats(cand["event_to_window"]), "+-0.5 s"),
                ("event -> drawn, detected peaks", stats(det["event_to_window"]), "+-0.5 s")]
        print_table("Per candidate (reported after a live chunk)", crow)
        summary += [dict(stage=n, **s) for n, s, _ in crow]
    pd.DataFrame(summary).to_csv(os.path.join(res_dir, "latency_report_summary.csv"), index=False)
    for n in notes:
        print(f"\nNOTE: {n}")
    print(f"\nwritten: latency_report_chunks.csv, latency_report_candidates.csv, latency_report_summary.csv "
          f"in {res_dir}")


# ─────────────────────────────────────────────────── past recording (--history) ───
def report_history(folder, every=1, out_dir=None, laptop_minus_nas=None, max_missing=30):
    folder = folder.rstrip("/")
    exp = os.path.basename(folder)
    rows, n, missing = [], 1, 0
    while missing < max_missing:
        p = os.path.join(folder, "F1", f"{exp}_F1_{n}.mat")
        if not os.path.exists(p):
            missing += 1
            n += every
            continue
        missing = 0
        files = [p] + [q for k in range(2, 5) if os.path.exists(q := os.path.join(folder, f"F{k}", f"{exp}_F{k}_{n}.mat"))]
        t = np.asarray(sio.loadmat(p, variable_names=["time"])["time"]).ravel()
        rows.append(dict(chunk=n, t_first_s=float(t[0]), t_last_s=float(t[-1]),
                         laptop_saved=max(mat_header_time(f) for f in files),
                         nas_created=max(created_time(f) for f in files)))
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} chunks read (chunk {n})", flush=True)
        n += every
    if not rows:
        sys.exit(f"no chunks found in {folder}/F1")
    h = pd.DataFrame(rows)
    x = h["laptop_saved"] - h["t_last_s"]
    h["B_save"] = x - save_base(x.to_numpy())
    h["nas_minus_saved"] = h["nas_created"] - (h["laptop_saved"] + HALF_S)
    if laptop_minus_nas is not None:
        h["C_sync"] = h["nas_minus_saved"] + laptop_minus_nas
    h["daq_gap_s"] = (h["t_first_s"] - h["t_first_s"].shift() - 10.0 * h["chunk"].diff()).round(4)
    out_dir = out_dir or os.path.join(paths.RESULTS, exp)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "latency_history.csv")
    h.to_csv(out, index=False)
    d = h["nas_minus_saved"]
    print(f"{folder}: {len(h)} chunks (every {every}), {h['t_last_s'].iloc[-1] / 3600:.2f} h")
    rows = [("B  chunk end -> saved on laptop (beyond fastest)", stats(h["B_save"]), "+-0.5 s"),
            ("NAS creation - laptop save (clocks + sync)", stats(d), "+-0.5 s")]
    if "C_sync" in h:
        rows.append(("C  laptop -> NAS (sync)", stats(h["C_sync"]), "+-0.5 s"))
    print_table("Past recording", rows)
    print(f"\n  NAS - laptop save: drift over the recording {d.iloc[-1] - d.iloc[0]:+.2f} s; a constant value "
          f"is a clock offset with a steady sync,\n  steps or a spread of seconds are sync delays "
          f"(the offset and the sync delay cannot be told apart without --laptop_minus_nas).")
    gaps = h[h["daq_gap_s"].abs() > 0.05]
    print(f"  DAQ time vector: {len(gaps)} jump(s) between the chunks read, {gaps['daq_gap_s'].sum():+.1f} s in total"
          + (" (data missing in the recording)" if len(gaps) else ""))
    print(f"\nwritten: {out}")


def main():
    ap = argparse.ArgumentParser(description="latency of the real-time path, stage by stage")
    ap.add_argument("results", nargs="?", help="results folder of a tdifc_app.py --latency run")
    ap.add_argument("--laptop_minus_nas", type=float, default=None, help="DAQ laptop clock - NAS clock (s)")
    ap.add_argument("--laptop_clock_file", nargs="+", default=None,
                    help="file(s) the DAQ laptop wrote to the NAS with its time (see the top of this file)")
    ap.add_argument("--history", default=None, help="a past experiment folder (with F1/...): stages B and C only")
    ap.add_argument("--every", type=int, default=1, help="--history: read every n-th chunk")
    ap.add_argument("--out", default=None, help="--history: output folder (default results/<exp_name>/)")
    a = ap.parse_args()
    lmn = a.laptop_minus_nas
    if a.laptop_clock_file:
        vals = [read_laptop_clock_file(f) for f in a.laptop_clock_file]
        print("DAQ laptop - NAS from " + ", ".join(f"{os.path.basename(f)}: {v:+.3f} s"
                                                   for f, v in zip(a.laptop_clock_file, vals)))
        lmn = float(np.mean(vals))
    if a.history:
        report_history(a.history, a.every, a.out, lmn)
    elif a.results:
        report_run(a.results, lmn)
    else:
        ap.error("give a results folder or --history")


if __name__ == "__main__":
    main()
