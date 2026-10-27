"""
Fit the Gilbert-Elliott burst model to the real CRAWDAD niit/bit_errors 802.15.4 traces.

    python scripts/fit_crawdad.py

Needs the traces in data/crawdad/Traces_802.15.4/*.txt (download them from IEEE DataPort,
DOI 10.15783/C7S88K) and numba. Writes
    models/crawdad_ge.json        fitted parameters used by the 'crawdad' channel profile
    results/crawdad_traces.json   per-trace statistics and fits
    results/crawdad_traces.png    per-trace error rates, burst lengths, model check
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardcode.channels import ge_states                                   # noqa: E402
from guardcode.crawdad import (FIT_PATH, environment, fit_gilbert_elliott,  # noqa: E402
                               load_trace, noise_rise_db, trace_files)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_lengths(err):
    """Lengths of runs of consecutive erroneous bits inside each frame."""
    out = []
    for row in err[err.any(axis=1)]:
        d = np.diff(np.concatenate([[0], row, [0]]).astype(np.int8))
        out.extend((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist())
    return np.array(out, dtype=np.int64)


def main():
    files = trace_files()
    if not files:
        raise SystemExit("no traces in data/crawdad/Traces_802.15.4 - see the docstring")
    per, pooled, by_env = [], [], {}
    t0 = time.time()
    for p in files:
        err, skipped = load_trace(p)
        x = err.reshape(-1)
        fit = fit_gilbert_elliott(x)
        rec = dict(trace=os.path.basename(p), environment=environment(p), frames=int(len(err)),
                   skipped_lines=int(skipped), ber=float(x.mean()), fer=float(err.any(axis=1).mean()),
                   max_errors_per_frame=int(err.sum(axis=1).max()), **fit)
        per.append(rec)
        pooled.append(x)
        by_env.setdefault(rec["environment"], []).append(x)
        print(f"{rec['trace']:22s} BER {rec['ber']:.2e}  FER {rec['fer']:.3f}  p_gb {fit['p_gb']:.2e}  "
              f"p_bg {fit['p_bg']:.3f}  e_G {fit['e_good']:.1e}  e_B {fit['e_bad']:.3f}", flush=True)
    env_fits = {}
    for env, xs in by_env.items():
        f = fit_gilbert_elliott(np.concatenate(xs))
        f["bad_extra_db"] = noise_rise_db(f["e_good"], f["e_bad"])
        env_fits[env] = f
        print(f"[{env}] p_gb {f['p_gb']:.2e} p_bg {f['p_bg']:.3f} e_G {f['e_good']:.1e} e_B {f['e_bad']:.3f} "
              f"-> bad state +{f['bad_extra_db']:.1f} dB", flush=True)
    allx = np.concatenate(pooled)
    f = fit_gilbert_elliott(allx)
    f["bad_extra_db"] = noise_rise_db(f["e_good"], f["e_bad"])
    print(f"[all 802.15.4 traces] p_gb {f['p_gb']:.2e} p_bg {f['p_bg']:.3f} e_G {f['e_good']:.1e} "
          f"e_B {f['e_bad']:.3f} -> bad state +{f['bad_extra_db']:.1f} dB  ({time.time() - t0:.0f}s)", flush=True)
    out = dict(source="CRAWDAD niit/bit_errors (IEEE DataPort, DOI 10.15783/C7S88K), 802.15.4 traces",
               bits=int(allx.size), traces=len(files), pooled=f, environments=env_fits)
    os.makedirs(os.path.dirname(FIT_PATH), exist_ok=True)
    with open(FIT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "crawdad_traces.json"), "w") as fh:
        json.dump(dict(per_trace=per, **out), fh, indent=1)
    plot(per, pooled, f)
    print("wrote", FIT_PATH)


def plot(per, pooled, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    envs = sorted({r["environment"] for r in per})
    cols = dict(zip(envs, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]))
    ax = axes[0]
    for i, r in enumerate(per):
        ax.bar(i, max(r["ber"], 1e-7), color=cols[r["environment"]])
    ax.set_yscale("log")
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels([r["trace"].replace(".txt", "") for r in per], rotation=90, fontsize=7)
    ax.set_ylabel("measured bit error rate")
    ax.set_title("22 real 802.15.4 traces (colour = setup)")
    for env, c in cols.items():
        ax.bar(0, 0, color=c, label=env)
    ax.legend(fontsize=8)
    # burst (error-run) lengths: measured vs fitted model
    err = np.concatenate([x.reshape(-1, 160) for x in pooled])
    meas = run_lengths(err)
    rng = np.random.default_rng(0)
    n_sim = min(len(err), 200000)
    st = ge_states(n_sim, 160, fit["p_gb"], fit["p_bg"], rng)
    sim = np.where(st, rng.random(st.shape) < fit["e_bad"], rng.random(st.shape) < fit["e_good"]).astype(np.uint8)
    simr = run_lengths(sim)
    bins = np.arange(1, 12)
    ax = axes[1]
    for data, lab, c in ((meas, "measured", "#d62728"), (simr, "fitted Gilbert-Elliott", "#1f77b4")):
        h = np.array([(data == k).sum() for k in bins[:-1]] + [(data >= bins[-1]).sum()], dtype=float)
        ax.plot(bins, h / max(h.sum(), 1), "o-", color=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("length of a run of consecutive bit errors (11 = 11+)")
    ax.set_ylabel("share of runs")
    ax.set_title("Burst lengths")
    ax.legend(fontsize=8)
    ax = axes[2]
    k = np.arange(0, 41)
    for data, n, lab, c in ((err, len(err), "measured", "#d62728"), (sim, n_sim, "fitted model", "#1f77b4")):
        cnt = np.bincount(np.minimum(data.sum(axis=1), 40), minlength=41) / n
        ax.semilogy(k, np.where(cnt > 0, cnt, np.nan), "o-", ms=3, color=c, label=lab)
    ax.set_xlabel("bit errors in a 160-bit frame (40 = 40+)")
    ax.set_ylabel("share of frames")
    ax.set_title("Errors per frame")
    ax.legend(fontsize=8)
    fig.suptitle("CRAWDAD niit/bit_errors (802.15.4, MicaZ, 20-byte frames): measured vs fitted burst model", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "crawdad_traces.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
