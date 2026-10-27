"""
Bulk-populate the GuardCode database with simulated frames.

Every scheme x channel (AWGN, burst, fading) x Eb/N0 (0..8 dB) combination is simulated
and EVERY frame is stored as one row of the `frames` table.

    python scripts/populate_db.py                    # 1,000,000 frames (default)
    python scripts/populate_db.py --frames 5000000   # larger database
    python scripts/populate_db.py --crc CRC-8,CRC-16 # both CRCs (frames split between them)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardcode.channels import ChannelConfig       # noqa: E402
from guardcode.db import Database                  # noqa: E402
from guardcode.linksim import run_batch            # noqa: E402
from guardcode.schemes import crc, load_schemes    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--crc", default="CRC-8", help="CRC-8, CRC-16 or both: CRC-8,CRC-16")
    a = ap.parse_args()

    db = Database()
    con = db.connect()
    run_id = db.new_run("bulk", f"populate_db.py --frames {a.frames} --crc {a.crc}", con)
    rng = np.random.default_rng(a.seed)
    ebnos = list(range(0, 9))
    plan = []
    for crc_name in a.crc.split(","):
        for prof in ("awgn", "burst", "fading"):
            for s in load_schemes(prof, crc_name=crc_name):
                for e in ebnos:
                    plan.append((prof, s, float(e)))
    per = max(1, a.frames // len(plan))
    print(f"run {run_id}: {len(plan)} combinations x {per} frames = {len(plan) * per:,} rows -> {db.path}")
    t0 = time.time()
    done = 0
    base_ts = time.time()
    for i, (prof, s, e) in enumerate(plan):
        left, seq = per, 0
        while left:
            B = min(5000, left)
            r = run_batch(s, crc(s.crc_name), e, ChannelConfig(prof), B, rng)
            score = r["llr"] if s.gate == "llr" else r["risk"]
            ts = base_ts + done / 1e4
            rows = [(run_id, ts, seq + j, s.name, prof, e, int(r["bit_err"][j]), int(r["outcome"][j]),
                     float(score[j]), 1) for j in range(B)]
            db.insert_frames(rows, con, crc=s.crc_name)
            left -= B
            seq += B
            done += B
        if (i + 1) % 9 == 0:
            el = time.time() - t0
            print(f"  {s.crc_name:6s} {prof:6s} {s.name:16s} done  | {done:,} rows | {done / el:,.0f} rows/s", flush=True)
    con.close()
    st = db.stats()
    print(f"finished in {time.time() - t0:.0f}s - database now holds {st['frames']:,} frames ({st['size_mb']} MB)")


if __name__ == "__main__":
    main()
