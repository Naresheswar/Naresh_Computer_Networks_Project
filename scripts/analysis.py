"""
Supporting analyses for the report / patent specification.

  results/tradeoff_*.png     undetected-error rate vs false-reject rate (SSG vs classic LLR test)
  results/calibration.png    SSG predicted risk vs observed error frequency
  results/adaptive.png       channel-adaptive profile switching timeline
  results/analysis.json      the numbers behind the figures

    python scripts/analysis.py [--frames 600000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from guardcode.adaptive import AdaptiveController                       # noqa: E402
from guardcode.channels import ChannelConfig, transmit_np                # noqa: E402
from guardcode.linksim import run_batch                                  # noqa: E402
from guardcode.schemes import MODELS, ROOT, HammingScheme, NeuralScheme, crc, ssg, thresholds  # noqa: E402

RES = os.path.join(ROOT, "results")


def collect(scheme, prof, ebnos, n, seed, bitwise=False):
    c, rng = crc(), np.random.default_rng(seed)
    W, OKc, R, L, RB = [], [], [], [], []
    for e in ebnos:
        for _ in range(max(1, n // len(ebnos) // 20000)):
            r = run_batch(scheme, c, e, ChannelConfig(prof), 20000, rng)
            W.append(r["wrong"]); OKc.append(r["crc_ok"]); R.append(r["risk"]); L.append(r["llr"])
            if bitwise:   # ablation: guard computed from independent bit marginals only
                RB.append(ssg().risk(r["p1"]))
    out = [np.concatenate(W), np.concatenate(OKc), np.concatenate(R), np.concatenate(L)]
    return out + ([np.concatenate(RB)] if bitwise else [])


def tradeoff(prof, n, out):
    ham = HammingScheme(interleave=(prof == "burst"))
    nn = NeuralScheme(os.path.join(MODELS, f"base_{prof}.npz"), "nn")
    fig, ax = plt.subplots(figsize=(7, 5))
    res = {}
    for label, s, col in (("Hamming", ham, "#1f77b4"), ("Neural", nn, "#d62728")):
        wrong, ok, risk, llr, riskb = collect(s, prof, [2.0, 3.0, 4.0], n, 5, bitwise=True)
        good = ~wrong
        ue_mask = wrong & ok
        N = len(wrong)
        frrs = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
        ssg_ue, llr_ue, bit_ue = [], [], []
        for f in frrs:
            if f == 0:
                ssg_ue.append(ue_mask.sum()); llr_ue.append(ue_mask.sum()); bit_ue.append(ue_mask.sum()); continue
            tr = np.quantile(risk[good & ok], 1 - f)
            tb = np.quantile(riskb[good & ok], 1 - f)
            tl = np.quantile(llr[good & ok], f)
            ssg_ue.append(int((ue_mask & (risk <= tr)).sum()))
            bit_ue.append(int((ue_mask & (riskb <= tb)).sum()))
            llr_ue.append(int((ue_mask & (llr >= tl)).sum()))
        ssg_ue, llr_ue, bit_ue = np.array(ssg_ue) / N, np.array(llr_ue) / N, np.array(bit_ue) / N
        ax.plot(100 * frrs, llr_ue, "--", marker="v", color=col, label=f"{label} + classic min-|LLR| test")
        ax.plot(100 * frrs, bit_ue, ":", marker="s", color=col, label=f"{label} + SSG, bit-wise variant")
        ax.plot(100 * frrs, ssg_ue, "-", marker="o", color=col, label=f"{label} + Soft-Syndrome Guard (group WHT)")
        res[label] = dict(frames=int(N), crc_undetected=int(ue_mask.sum()), frr_pct=(100 * frrs).tolist(),
                          uer_llr=llr_ue.tolist(), uer_ssg=ssg_ue.tolist(), uer_ssg_bitwise=bit_ue.tolist())
    ax.set_yscale("log")
    ax.set_xlabel("Correct frames rejected (extra retransmissions), %")
    ax.set_ylabel("Undetected frame error rate")
    ax.set_title(f"Undetected errors vs retransmission cost - {prof.upper()}, Eb/N0 2-4 dB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"tradeoff_{prof}.png"), dpi=150)
    plt.close(fig)
    out[f"tradeoff_{prof}"] = res


def calibration(n, out):
    nn = NeuralScheme(os.path.join(MODELS, "base_awgn.npz"), "nn")
    ham = HammingScheme()
    fig, ax = plt.subplots(figsize=(6, 5))
    res = {}
    for label, s, col in (("Neural decoder", nn, "#d62728"), ("Hamming decoder", ham, "#1f77b4")):
        wrong, ok, risk, _ = collect(s, "awgn", [1.0, 2.0, 3.0], n, 9)
        m = ok                                    # SSG is applied to CRC-passing frames
        edges = np.array([0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
        px, py, cnt = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = m & (risk > lo) & (risk <= hi)
            if sel.sum() >= 200:
                px.append(risk[sel].mean()); py.append(max(wrong[sel].mean(), 1e-6)); cnt.append(int(sel.sum()))
        ax.loglog(px, py, "o-", color=col, label=label)
        res[label] = dict(pred=px, obs=py, n=cnt, total_pred=float(risk[m].sum()),
                          total_obs=int((wrong & m).sum()))
    lim = [1e-5, 1]
    ax.plot(lim, lim, "k:", label="perfect calibration")
    ax.set_xlabel("SSG predicted risk  R(y)")
    ax.set_ylabel("Observed fraction of CRC-passing frames that are wrong")
    ax.set_title("Soft-Syndrome Guard calibration (AWGN, 1-3 dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "calibration.png"), dpi=150)
    plt.close(fig)
    out["calibration"] = res


def adaptive_demo(out, ebno=4.0, blocks=(60, 60, 60), B=300):
    ta, tb = thresholds("awgn"), thresholds("burst")
    nn_a = NeuralScheme(os.path.join(MODELS, "base_awgn.npz"), "a")
    nn_b = NeuralScheme(os.path.join(MODELS, "base_burst.npz"), "b")
    fixed = nn_a.with_gate("ssg", ta["ssg"], "fixed")
    ctl = AdaptiveController({"awgn": nn_a.with_gate("ssg", ta["ssg"], "ad"),
                              "burst": nn_b.with_gate("ssg", tb["ssg"], "ad")})
    c, rng = crc(), np.random.default_rng(4)
    chans = ["awgn"] * blocks[0] + ["burst"] * blocks[1] + ["awgn"] * blocks[2]
    tl = dict(bidx=[], prof=[], fer_ad=[], fer_fx=[], gp_ad=[], gp_fx=[])
    for ch in chans:
        msgs = rng.integers(0, 2, (B, 32)).astype(np.uint8)
        fb = c.encode(msgs)
        for key, s in (("ad", ctl.scheme), ("fx", fixed)):
            x = s.encode(fb.astype(np.float64))
            y, h, _ = transmit_np(x, ebno, s.rate, ChannelConfig(ch), rng)
            bits, acc, _, _ = s.decide(y, h, ebno)
            wrong = (bits != fb).any(axis=1)
            tl[f"fer_{key}"].append(float(wrong.mean()))
            tl[f"gp_{key}"].append(float((acc & ~wrong).mean() * 32 / s.n))
            if key == "ad":
                b, p = ctl.observe(y, bits)
                tl["bidx"].append(b); tl["prof"].append(p)
    t = np.arange(len(chans))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, tl["bidx"], color="#344054", label="burst index (EWMA)")
    axes[0].axhline(ctl.hi, color="#d62728", ls="--", lw=1, label="switch-up threshold")
    axes[0].axhline(ctl.lo, color="#1f77b4", ls="--", lw=1, label="switch-down threshold")
    for i, ch in enumerate(chans):
        if ch == "burst":
            axes[0].axvspan(i - 0.5, i + 0.5, color="#fde2e1", lw=0)
            axes[1].axvspan(i - 0.5, i + 0.5, color="#fde2e1", lw=0)
    axes[0].set_ylabel("burst index")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].plot(t, tl["gp_fx"], color="#ff7f0e", label="GuardCode, fixed AWGN profile")
    axes[1].plot(t, tl["gp_ad"], color="#079455", label="GuardCode-Adaptive")
    axes[1].set_ylabel("goodput (bits/use)")
    axes[1].set_xlabel(f"block index ({B} frames per block, Eb/N0 = {ebno} dB; shaded = burst channel)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Channel-adaptive profile selection from decoder residuals (no pilots, no extra bits)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "adaptive.png"), dpi=150)
    plt.close(fig)
    burst = np.array(chans) == "burst"
    out["adaptive"] = dict(
        switches=ctl.switches,
        goodput_burst_fixed=float(np.mean(np.array(tl["gp_fx"])[burst])),
        goodput_burst_adaptive=float(np.mean(np.array(tl["gp_ad"])[burst])),
        goodput_awgn_fixed=float(np.mean(np.array(tl["gp_fx"])[~burst])),
        goodput_awgn_adaptive=float(np.mean(np.array(tl["gp_ad"])[~burst])),
        profile_correct_pct=float(100 * np.mean([(p == ch) for p, ch in zip(tl["prof"], chans)])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600000)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    out = {}
    for prof in ("awgn", "burst"):
        tradeoff(prof, a.frames, out)
        print("tradeoff", prof, "done", flush=True)
    calibration(a.frames, out)
    print("calibration done", flush=True)
    adaptive_demo(out)
    print("adaptive done", json.dumps(out["adaptive"]), flush=True)
    with open(os.path.join(RES, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
