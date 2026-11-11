"""
Replay a recorded t-DiFC scan as DiDAQ chunk files, to test tdifc_app.py without the DAQ.

Reads a recording (.mat with `data` = N x 2, as in the clearance-cohort raw folders) and
writes, every chunk_s / speed seconds, one 10 s chunk per fiber folder exactly as
DiDAQ's DiFC_SaveData does:
    <save_dir>/<exp_name>/F1/<exp_name>_F1_<n>.mat     (data, time, params)
    <save_dir>/<exp_name>/F2/<exp_name>_F2_<n>.mat     (--layout split, the default)
--layout split : Ch1 in F1, Ch2 in F2 (one channel per fiber folder, as the lab's DiDAQ
                 simulation output);  --layout single : both channels in F1.
Files are written under a temporary name and renamed, so a reader never sees half a file.

Usage:  python simulate_acquisition.py --mat "<recording>.mat" ["<next>.mat" ...] --save_dir sim --exp_name TEST
            [--speed 10] [--start_min 0] [--minutes 60] [--layout split|single] [--first_chunk 1]
Several --mat files are replayed one after the other (e.g. the 2-h files of one scan);
--first_chunk continues the numbering of an earlier run (to add live chunks to saved ones).
Then, in another terminal:  python tdifc_app.py --save_dir sim --exp_name TEST
"""
import argparse
import os
import time

import numpy as np
import scipy.io as sio

FS = 2000


def main():
    ap = argparse.ArgumentParser(description="replay a recording as DiDAQ 10 s chunk files")
    ap.add_argument("--mat", required=True, nargs="+", help="recording .mat file(s) (data = N x 2), in order")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--chunk_s", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="x real time (0 = as fast as possible)")
    ap.add_argument("--start_min", type=float, default=0.0)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--layout", default="split", choices=["split", "single"])
    ap.add_argument("--first_chunk", type=int, default=1, help="number of the first chunk written")
    args = ap.parse_args()

    data = np.vstack([np.asarray(sio.loadmat(m, variable_names=["data"])["data"], dtype=np.float64)
                      for m in args.mat])
    a = int(args.start_min * 60 * FS)
    b = len(data) if args.minutes is None else min(len(data), a + int(args.minutes * 60 * FS))
    data = data[a:b]
    step = int(args.chunk_s * FS)
    fibers = {1: [0], 2: [1]} if args.layout == "split" else {1: [0, 1]}
    root = os.path.join(args.save_dir, args.exp_name)
    for f in fibers:
        os.makedirs(os.path.join(root, f"F{f}"), exist_ok=True)
    n_chunks = int(np.ceil(len(data) / step))
    print(f"{args.mat}: {len(data) / FS / 60:.1f} min -> {n_chunks} chunks of {args.chunk_s:g} s in {root} "
          f"({args.layout}), {args.speed:g}x real time")
    t_next = time.time()
    k0 = args.first_chunk - 1                   # time continues with the chunk numbering
    for k in range(n_chunks):
        d = data[k * step:(k + 1) * step]
        t = ((k0 + k) * step + np.arange(len(d))) / FS
        for f, cols in fibers.items():
            name = f"{args.exp_name}_F{f}_{k0 + k + 1}.mat"
            dest = os.path.join(root, f"F{f}", name)
            tmp = os.path.join(root, f"F{f}", f".{name}.part")
            params = {"name": np.array([f"Ch{c + 1}" for c in cols], dtype=object), "units": "mV", "fiber": f}
            with open(tmp, "wb") as fh:
                sio.savemat(fh, {"data": d[:, cols], "time": t.reshape(-1, 1), "params": params})
            os.replace(tmp, dest)
        if (k + 1) % 30 == 0 or k + 1 == n_chunks:
            print(f"  chunk {k0 + k + 1} ({k + 1}/{n_chunks}, t = {(k0 + k + 1) * args.chunk_s / 60:.1f} min)", flush=True)
        if args.speed > 0:
            t_next += args.chunk_s / args.speed
            time.sleep(max(0.0, t_next - time.time()))


if __name__ == "__main__":
    main()
