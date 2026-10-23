"""
Benchmark sweep: every scheme x channel profile x Eb/N0, results to CSV + PNG plots,
and every data point also stored in the SQLite database (table `benchmark`).

    python -m guardcode.evaluate --frames 200000          (quick)
    python -m guardcode.evaluate --frames 2000000         (report quality)
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .channels import ChannelConfig
from .linksim import run_batch, simulate
from .schemes import ROOT, crc, load_schemes, MODELS

RESULTS = os.path.join(ROOT, "results")


def calibrate(profile, ebno=3.0, frr=0.01, n=200000, seed=11):
    """Calibrate the gates so that each rejects the same share (frr) of CORRECTLY decoded
    frames at the calibration Eb/N0 - i.e. both gates get the same retransmission budget."""
    from .schemes import HammingScheme, NeuralScheme
    c = crc()
    cfg = ChannelConfig(profile)
    out = {}
    jobs = [("ssg_hamming", HammingScheme(interleave=(profile == "burst")), "risk")]
    base = os.path.join(MODELS, f"base_{profile}.npz")
    if os.path.exists(base):
        nb = NeuralScheme(base, "cal")
        jobs += [("llr", nb, "llr"), ("ssg", nb, "risk")]
    for key, s, score in jobs:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n // 20000):
            r = run_batch(s, c, ebno, cfg, 20000, rng)
            vals.append(r[score][~r["wrong"]])
        v = np.concatenate(vals)
        out[key] = float(np.quantile(v, frr)) if score == "llr" else float(np.quantile(v, 1 - frr))
    with open(os.path.join(MODELS, f"thresholds_{profile}.txt"), "w") as f:
        for k, v in out.items():
            f.write(f"{k}={v:.8g}\n")
    return out


def sweep(profiles=("awgn", "burst"), ebnos=None, frames=200000, db=True):
    os.makedirs(RESULTS, exist_ok=True)
    ebnos = ebnos if ebnos is not None else [0, 1, 2, 3, 4, 5, 6, 7]
    c = crc()
    rows = []
    for prof in profiles:
        thr = calibrate(prof)
        print(f"[{prof}] calibrated gate thresholds (1% false-reject at 3 dB): {thr}", flush=True)
        for s in load_schemes(prof):
            for e in ebnos:
                t0 = time.time()
                r = simulate(s, c, e, ChannelConfig(prof), frames, seed=1000 + int(e * 10))
                r.update(profile=prof, scheme=s.name, ebno_db=e, n_sym=s.n, rate=round(s.rate, 4),
                         seconds=round(time.time() - t0, 2))
                rows.append(r)
                print(f"{prof:5s} {s.name:15s} {e:4.1f} dB  BER {r['ber']:.2e}  FER {r['fer']:.2e}  "
                      f"UER {r['uer']:.2e} ({r['undetected']})  goodput {r['goodput']:.3f}", flush=True)
    keys = ["profile", "scheme", "ebno_db", "n_sym", "rate", "frames", "bit_errors", "frame_errors",
            "detected", "undetected", "crc_undetected", "est_undetected", "false_rej", "ber", "fer", "uer",
            "reject_rate", "goodput", "seconds"]
    with open(os.path.join(RESULTS, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    if db:
        from .db import Database
        d = Database()
        d.insert_benchmark(rows)
    return rows


STYLE = {
    "Uncoded+CRC": dict(color="#7f7f7f", marker="x", ls=":"),
    "Hamming+CRC": dict(color="#1f77b4", marker="s", ls="--"),
    "Hamming+CRC+SSG": dict(color="#17becf", marker="D", ls="-"),
    "Neural+CRC": dict(color="#ff7f0e", marker="^", ls="-."),
    "Neural+CRC+LLR": dict(color="#9467bd", marker="v", ls="--"),
    "GuardCode": dict(color="#d62728", marker="o", ls="-"),
}


def plot(rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rows is None:
        with open(os.path.join(RESULTS, "benchmark.csv")) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in r:
                if k not in ("profile", "scheme"):
                    r[k] = float(r[k])
    profiles = sorted({r["profile"] for r in rows})
    metrics = [("ber", "Bit error rate (info bits)", True), ("fer", "Frame error rate", True),
               ("uer", "Undetected frame error rate", True), ("goodput", "ARQ goodput (info bits / channel use)", False)]
    for prof in profiles:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (m, title, logy) in zip(axes.flat, metrics):
            for sch in STYLE:
                pts = [(r["ebno_db"], r[m], r["frames"]) for r in rows if r["profile"] == prof and r["scheme"] == sch]
                if not pts:
                    continue
                x = [p[0] for p in pts]
                if m == "uer":
                    # zero counts are drawn at the 1/N resolution floor with a hollow marker
                    y = [p[1] if p[1] > 0 else np.nan for p in pts]
                    ax.plot(x, y, label=sch, **STYLE[sch])
                    floor = [(p[0], 1.0 / p[2]) for p in pts if p[1] == 0]
                    if floor:
                        ax.scatter([f[0] for f in floor], [f[1] for f in floor], marker="v",
                                   facecolors="none", edgecolors=STYLE[sch]["color"], s=50)
                else:
                    y = [p[1] if (p[1] > 0 or not logy) else np.nan for p in pts]
                    ax.plot(x, y, label=sch, **STYLE[sch])
            ax.set_title(title)
            ax.set_xlabel("Eb/N0 (dB)")
            if logy:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"GuardCode benchmark - {prof.upper()} channel (32-bit payload, CRC-8, 70 coded symbols)"
                     + ("\n(hollow triangles: zero undetected errors observed, drawn at 1/N)" ), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f"benchmark_{prof}.png"), dpi=150)
        plt.close(fig)
    # undetected fraction bar chart (pooled over SNRs)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.13
    schemes = [s for s in STYLE if any(r["scheme"] == s for r in rows)]
    for i, sch in enumerate(schemes):
        vals = []
        for prof in profiles:
            sub = [r for r in rows if r["profile"] == prof and r["scheme"] == sch]
            fe = sum(r["frame_errors"] for r in sub)
            ue = sum(r["undetected"] for r in sub)
            vals.append(100.0 * ue / fe if fe else 0)
        ax.bar(np.arange(len(profiles)) + i * width, vals, width, label=sch, color=STYLE[sch]["color"])
    ax.set_xticks(np.arange(len(profiles)) + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels([p.upper() for p in profiles])
    ax.set_ylabel("% of frame errors delivered undetected")
    ax.set_yscale("log")
    ax.set_title("Share of frame errors that slip past the receiver (pooled over all Eb/N0)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "undetected_share.png"), dpi=150)
    plt.close(fig)


def plot_training():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, col in (("base_awgn", "#d62728"), ("base_burst", "#1f77b4")):
        p = os.path.join(MODELS, f"{name}_history.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))
        axes[0].plot([r["step"] for r in h], [r["bce"] for r in h], label=name, color=col)
        axes[1].plot([r["step"] for r in h], [r["fer"] for r in h], label=name, color=col)
    axes[0].set_title("Bit-metric cross-entropy (training loss)")
    axes[1].set_title("Training-batch frame error rate (Eb/N0 ~ U[1,6] dB)")
    for a in axes:
        a.set_xlabel("training step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200000)
    ap.add_argument("--profiles", default="awgn,burst")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--import-csv", action="store_true",
                    help="load an existing results/benchmark.csv into the database instead of re-simulating")
    a = ap.parse_args()
    if a.import_csv:
        from .db import Database
        with open(os.path.join(RESULTS, "benchmark.csv")) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in r:
                if k not in ("profile", "scheme"):
                    r[k] = float(r[k])
        Database().insert_benchmark(rows)
        print(f"imported {len(rows)} benchmark rows into the database")
        raise SystemExit(0)
    rows = sweep(a.profiles.split(","), frames=a.frames, db=not a.no_db)
    plot(rows)
    plot_training()
    print("done ->", RESULTS)
