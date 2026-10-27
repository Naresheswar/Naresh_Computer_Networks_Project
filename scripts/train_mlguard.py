"""
Train the learned residual-risk classifier (GuardCode-ML) for each channel profile.

    python scripts/train_mlguard.py --frames 1500000
    python scripts/train_mlguard.py --crc CRC-16 --frames 6000000

Simulates CRC-passing frames (Eb/N0 spread over 0-5 dB, seeds disjoint from the benchmark and
the gate calibration), fits a HistGradientBoostingClassifier to predict "frame is wrong",
reports hold-out ranking quality against the SSG risk alone, and writes
models/mlguard_<profile>[_crc16].joblib. Re-run `python -m guardcode.evaluate` afterwards to
calibrate its threshold and benchmark it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardcode import mlguard                                    # noqa: E402
from guardcode.channels import ChannelConfig, transmit_np        # noqa: E402
from guardcode.schemes import NeuralScheme, _VALUE_BITS, crc, neural_path, ssg, suffix         # noqa: E402


def collect(profile, frames, seed, ebno_range=(0.0, 5.0), batch=20000, crc_name="CRC-8"):
    scheme = NeuralScheme(neural_path(profile, crc_name), "ml-train")
    c, cfg = crc(crc_name), ChannelConfig(profile)
    rng = np.random.default_rng(seed)
    X, Y, N = [], [], 0
    while N < frames:
        e = rng.uniform(*ebno_range)
        msgs = rng.integers(0, 2, (batch, 32)).astype(np.uint8)
        fb = c.encode(msgs)
        y, h, _ = transmit_np(scheme.encode(fb.astype(np.float64)), e, scheme.rate, cfg, rng)
        P = scheme.groups(y, h, e)
        m = int(np.log2(P.shape[2]))
        bits = _VALUE_BITS[m][P.argmax(axis=2)].reshape(batch, -1)
        ok = c.check(bits)
        _, risk = ssg(crc_name).risk_groups(P, rows=ok)
        p1 = np.clip(P @ _VALUE_BITS[m].astype(np.float64), 1e-15, 1 - 1e-15).reshape(batch, -1)
        llr = np.abs(np.log(p1) - np.log1p(-p1)).min(axis=1)
        sel = ok
        X.append(mlguard.features(scheme, P[sel], risk[sel], llr[sel], y[sel], h[sel], bits[sel]))
        Y.append((bits[sel] != fb[sel]).any(axis=1))
        N += batch
        print(f"  [{profile}] {N}/{frames} frames, {sum(map(len, Y))} CRC-pass, {int(sum(a.sum() for a in Y))} wrong", flush=True)
    return np.vstack(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1500000)
    ap.add_argument("--profiles", default="awgn,burst")
    ap.add_argument("--crc", default="CRC-8", choices=["CRC-8", "CRC-16"])
    a = ap.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score
    for prof in a.profiles.split(","):
        X, y = collect(prof, a.frames, seed=7000 + len(prof), crc_name=a.crc)
        Xt, yt = collect(prof, a.frames // 3, seed=8000 + len(prof), crc_name=a.crc)
        clf = mlguard.train_classifier(X, y)
        ml = mlguard.MLGuard(clf, dict(profile=prof, crc=a.crc, train_frames=a.frames, positives=int(y.sum())))
        s_ml, s_ssg = ml.score(Xt), Xt[:, 0]                    # column 0 = log10 SSG risk
        rep = dict(profile=prof, crc=a.crc, test_crc_pass=int(len(yt)), test_wrong=int(yt.sum()),
                   auc_ml=float(roc_auc_score(yt, s_ml)), auc_ssg=float(roc_auc_score(yt, s_ssg)),
                   ap_ml=float(average_precision_score(yt, s_ml)), ap_ssg=float(average_precision_score(yt, s_ssg)))
        print(json.dumps(rep), flush=True)
        ml.meta.update(rep)
        ml.save(mlguard.model_path(prof, a.crc))
        with open(os.path.join(mlguard.MODELS, f"mlguard_{prof}{suffix(a.crc)}_report.json"), "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
