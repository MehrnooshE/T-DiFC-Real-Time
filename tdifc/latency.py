"""
Timing of the real-time path, from the DAQ to the window (tdifc_app.py --latency).

Nothing in DiDAQ / MATLAB is changed: every time comes from what the files already carry or from
this computer's clock.

  laptop_saved  the "Created on:" time MATLAB writes into the header of every .mat chunk
                (DAQ laptop clock, whole seconds, rounded down)
  nas_created   the chunk file's creation time on the NAS (NAS clock; the sync tool does not keep
                the laptop's times, so the NAS stamps the file when it arrives)
  seen ... drawn  this computer's clock (ms): the worker finds the chunk, has loaded it, has
                preprocessed it, has scored its candidates, has queued it for the window; the
                window has drawn it

The three clocks differ (DAQ laptop vs NAS: ~13 min in July 2025), so the app also measures the
NAS clock against this computer: every PROBE_EVERY_S it creates a small file in the probe folder,
reads the creation time the NAS gives it and deletes it at once (latency_clock.csv). The DAQ laptop
vs NAS difference cannot be measured from here; latency_report.py takes it as a number or from a
file the laptop writes to the NAS (see latency_report.py).

Files (in the results folder, rewritten at every start, as candidates.csv):
  latency_chunks.csv      one row per chunk analysed in real time
  latency_candidates.csv  one row per candidate: the chunk its peak is in and the chunk after which
                          it was reported (the detector waits for the end of a laser-ON segment,
                          or for 3 s of signal after the peak with the continuous laser)
  latency_clock.csv       NAS clock - this computer's clock, and this computer vs an NTP server
"""
import csv
import datetime as dt
import logging
import os
import re
import socket
import subprocess
import threading
import time

log = logging.getLogger("latency")

PROBE_EVERY_S = 600
NTP_SERVER = "time.apple.com"
_HEADER_TIME = re.compile(rb"Created on: +([A-Z][a-z]{2} +[A-Z][a-z]{2} +\d+ +\d+:\d\d:\d\d +\d{4})")
_FIBER_TAG = re.compile(r"(?:^|(?<=[_\-. ]))F(\d+)(?=$|[_\-. ])")

CHUNK_COLS = ["lat_id", "chunk", "file", "live", "n_samples", "t_first_s", "t_last_s",
              "laptop_saved", "nas_created", "seen", "loaded", "processed", "scored", "queued", "drawn",
              "n_candidates", "n_detected"]
CAND_COLS = ["lat_id", "chunk_reported", "chunk_of_peak", "sample", "time_s", "prob", "detected"]
CLOCK_COLS = ["mac_time", "what", "offset_s", "uncertainty_s", "note"]


def mat_header_time(path):
    """Save time in a MATLAB v5 .mat header ("Created on: Wed Jul 23 17:21:29 2025"), as epoch
    seconds read in this computer's time zone (the DAQ laptop's must be the same). None if absent."""
    with open(path, "rb") as f:
        head = f.read(116)
    m = _HEADER_TIME.search(head)
    if not m:
        return None
    s = re.sub(r" +", " ", m.group(1).decode("ascii"))
    return dt.datetime.strptime(s, "%a %b %d %H:%M:%S %Y").timestamp()


def created_time(path):
    """Creation time of a file as the file server reports it (macOS: st_birthtime), else mtime."""
    st = os.stat(path)
    return getattr(st, "st_birthtime", None) or st.st_mtime


def chunk_paths(root, names):
    """Full paths of a chunk's files from their names: <root>/F<k>/<name> (DiDAQ) or <root>/<name>."""
    out = []
    for n in names:
        m = _FIBER_TAG.search(os.path.splitext(n)[0])
        cands = ([os.path.join(root, f"F{m.group(1)}", n)] if m else []) + [os.path.join(root, n)]
        out.append(next((p for p in cands if os.path.exists(p)), cands[0]))
    return out


def ntp_offset(server=NTP_SERVER):
    """(this computer - NTP time, uncertainty) in s from macOS sntp, or None."""
    try:
        out = subprocess.run(["sntp", "-t", "3", server], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"([+-]\d+\.\d+) \+/- (\d+\.\d+)", out)
        # sntp prints the correction to apply to this clock: clock - NTP = -correction
        return (-float(m.group(1)), float(m.group(2))) if m else None
    except Exception:
        return None


class LatencyLog:
    """Thread-safe: the worker thread fills a chunk's row, the GUI thread adds when it was drawn;
    the row is written once both the drawing time and the file times are in."""

    def __init__(self, out_dir, probe_dir=None, probe_every=PROBE_EVERY_S):
        self.out_dir = out_dir
        self._lock = threading.Lock()
        self._rows = {}
        self._next = 0
        self.paths = {k: os.path.join(out_dir, f"latency_{k}.csv") for k in ("chunks", "candidates", "clock")}
        for k, cols in (("chunks", CHUNK_COLS), ("candidates", CAND_COLS), ("clock", CLOCK_COLS)):
            with open(self.paths[k], "w", newline="") as f:
                csv.writer(f).writerow(cols)
        self.probe_dir = probe_dir
        self._stop = threading.Event()
        threading.Thread(target=self._clock_loop, args=(probe_every,), daemon=True).start()
        log.info("latency log: %s (NAS clock probe in %s every %d s)", out_dir, probe_dir or "-", probe_every)

    # ---------------------------------------------------------------- chunks ---
    def new(self, idx, live, seen):
        """A chunk found ready at time `seen`; live = the worker was waiting for it (not a backlog)."""
        with self._lock:
            self._next += 1
            self._rows[self._next] = dict(lat_id=self._next, chunk=idx, live=int(bool(live)), seen=seen)
            return self._next

    def mark(self, lat_id, key, t=None, **extra):
        if lat_id is None:
            return
        with self._lock:
            r = self._rows.get(lat_id)
            if r is not None:
                r[key] = time.time() if t is None else t
                r.update(extra)

    def file_times(self, lat_id, root, names):
        """Header (laptop) and NAS creation times of the chunk's files: the latest of them (a chunk
        is complete when its last fiber file is there). Run after the chunk was queued (off the
        critical path: a few small reads on the NAS)."""
        if lat_id is None:
            return
        saved, created = None, None
        for p in chunk_paths(root, names):
            try:
                h, c = mat_header_time(p), created_time(p)
            except OSError as e:
                log.debug("file times of %s: %s", p, e)
                continue
            saved = h if saved is None or (h is not None and h > saved) else saved
            created = c if created is None or c > created else created
        with self._lock:
            r = self._rows.get(lat_id)
            if r is None:
                return
            r.update(file=names[0] if names else "", laptop_saved=saved, nas_created=created, files_done=True)
            done = "drawn" in r
        if done:
            self._write(lat_id)

    def drawn(self, lat_ids, t=None):
        """GUI thread: these chunks are on the screen."""
        t = time.time() if t is None else t
        ready = []
        with self._lock:
            for i in lat_ids:
                r = self._rows.get(i)
                if r is not None and "drawn" not in r:
                    r["drawn"] = t
                    if r.get("files_done"):
                        ready.append(i)
        for i in ready:
            self._write(i)

    def _write(self, lat_id):
        with self._lock:
            r = self._rows.pop(lat_id, None)
        if r is None:
            return
        def fmt(k):
            v = r.get(k)
            if v is None:
                return ""
            return f"{v:.3f}" if isinstance(v, float) else v
        with open(self.paths["chunks"], "a", newline="") as f:
            csv.writer(f).writerow([fmt(k) for k in CHUNK_COLS])

    # ------------------------------------------------------------ candidates ---
    def candidates(self, lat_id, idx, rows):
        """rows: (chunk_of_peak, sample, time_s, prob, detected) of the candidates reported after chunk idx."""
        if lat_id is None or not rows:
            return
        with open(self.paths["candidates"], "a", newline="") as f:
            w = csv.writer(f)
            for c, s, t, p, d in rows:
                w.writerow([lat_id, idx, c, s, f"{t:.4f}", f"{p:.4f}", int(d)])

    # ----------------------------------------------------------------- clocks ---
    def _clock(self, what, offset, unc, note=""):
        with open(self.paths["clock"], "a", newline="") as f:
            csv.writer(f).writerow([f"{time.time():.3f}", what,
                                    "" if offset is None else f"{offset:.3f}",
                                    "" if unc is None else f"{unc:.3f}", note])

    def probe_nas(self):
        """NAS clock - this computer's clock: create a file, read the creation time the NAS gave it.
        Returns (offset, uncertainty) or None."""
        if not self.probe_dir:
            return None
        p = os.path.join(self.probe_dir, f"tdifc_clock_probe_{socket.gethostname()}_{os.getpid()}.tmp")
        try:
            t0 = time.time()
            fd = os.open(p, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
            t1 = time.time()
            os.close(fd)
            c = created_time(p)
            return c - (t0 + t1) / 2, (t1 - t0) / 2
        except OSError as e:
            self._clock("nas_minus_mac", None, None, f"probe failed: {e}")
            log.warning("NAS clock probe in %s failed (%s) - latency_report.py will bound the clock offset "
                        "instead", self.probe_dir, e)
            return None
        finally:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _clock_loop(self, every):
        ntp = ntp_offset()
        self._clock("mac_minus_ntp", *(ntp or (None, None)), NTP_SERVER if ntp else "sntp failed")
        while self.probe_dir:
            r = self.probe_nas()
            if r is not None:
                self._clock("nas_minus_mac", r[0], r[1], self.probe_dir)
            if self._stop.wait(every):
                return

    def stop(self):
        self._stop.set()
