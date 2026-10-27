"""
Real-world burst-error statistics from the CRAWDAD niit/bit_errors dataset (IEEE DataPort,
DOI 10.15783/C7S88K): residual bit-error traces of 802.15.4 MicaZ motes, 20-byte frames,
four indoor setups (OutRoom, PhdLab, Stair, Up_Floor), 5-12 m from the base station.

Trace format (trace_report.pdf): one received frame per line, hexadecimal bytes; the first
11 bytes are header, the next 20 are payload. A payload byte 0xAA means "received
correctly"; XOR with 0xAA gives the bits in error. Lines with a damaged length are skipped.

A two-state Gilbert-Elliott hidden Markov model (Good / Bad, per-bit error probability
e_G / e_B, transitions p_gb / p_bg) is fitted to each trace's bit-error sequence by
Baum-Welch. The project's 'crawdad' channel profile then uses
    * the measured burst timing  p_gb, p_bg   (per channel symbol = per bit), and
    * the noise rise of the Bad state, bad_extra_db = 10 log10(SNR_G / SNR_B), where the
      SNRs are those at which BPSK gives the fitted error probabilities e_G and e_B.

    python scripts/fit_crawdad.py        -> models/crawdad_ge.json, results/crawdad_*.{png,json}
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "data", "crawdad", "Traces_802.15.4")
FIT_PATH = os.path.join(ROOT, "models", "crawdad_ge.json")
HEADER, PAYLOAD = 11, 20


def trace_files(trace_dir=TRACE_DIR):
    return sorted(glob.glob(os.path.join(trace_dir, "*.txt")))


def load_trace(path):
    """Returns the (frames, 160) 0/1 matrix of payload bit errors and the number of skipped lines."""
    rows, skipped = [], 0
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) != HEADER + PAYLOAD:
                skipped += bool(tok)
                continue
            rows.append([int(x, 16) for x in tok[HEADER:]])
    data = np.array(rows, dtype=np.uint8).reshape(-1, PAYLOAD)
    return np.unpackbits(data ^ 0xAA, axis=1), skipped


def environment(path):
    """'OutRoom_3_8m.txt' -> 'OutRoom'."""
    name = os.path.basename(path)
    return name.rsplit("_", 2)[0]


try:
    import numba

    @numba.njit(cache=True)
    def _baum_welch_pass(x, a, b, e):
        """One scaled forward-backward pass of the 2-state HMM on 0/1 sequence x.
        a = P(G->B), b = P(B->G), e = (e_G, e_B). Returns expected counts and log-likelihood."""
        T = x.shape[0]
        A = np.array([[1 - a, a], [b, 1 - b]])
        pi = np.array([b / (a + b), a / (a + b)])
        alpha = np.empty((T, 2))
        scale = np.empty(T)
        for t in range(T):
            for s in range(2):
                em = e[s] if x[t] else 1 - e[s]
                if t == 0:
                    alpha[t, s] = pi[s] * em
                else:
                    alpha[t, s] = (alpha[t - 1, 0] * A[0, s] + alpha[t - 1, 1] * A[1, s]) * em
            scale[t] = alpha[t, 0] + alpha[t, 1]
            alpha[t, 0] /= scale[t]
            alpha[t, 1] /= scale[t]
        trans = np.zeros((2, 2))
        occ = np.zeros(2)
        err = np.zeros(2)
        beta = np.ones(2)
        for t in range(T - 1, -1, -1):
            g0 = alpha[t, 0] * beta[0]
            g1 = alpha[t, 1] * beta[1]
            gs = g0 + g1
            occ[0] += g0 / gs
            occ[1] += g1 / gs
            if x[t]:
                err[0] += g0 / gs
                err[1] += g1 / gs
            if t > 0:
                emis = np.empty(2)
                for s in range(2):
                    emis[s] = e[s] if x[t] else 1 - e[s]
                nb = np.empty(2)
                for i in range(2):
                    acc = 0.0
                    for j in range(2):
                        trans[i, j] += alpha[t - 1, i] * A[i, j] * emis[j] * beta[j] / scale[t]
                        acc += A[i, j] * emis[j] * beta[j]
                    nb[i] = acc
                beta[0] = nb[0] / scale[t]
                beta[1] = nb[1] / scale[t]
        return trans, occ, err, np.log(scale).sum()
except ImportError:
    _baum_welch_pass = None


def fit_gilbert_elliott(x, iters=60, tol=1e-6):
    """Baum-Welch fit of a 2-state Gilbert-Elliott HMM to the 0/1 error sequence x."""
    if _baum_welch_pass is None:
        raise ImportError("fitting needs numba (pip install numba)")
    x = np.ascontiguousarray(x, dtype=np.uint8)
    rate = max(x.mean(), 1e-7)
    a, b = 1e-3, 0.2
    e = np.array([rate * 0.1, min(0.5, rate * 50 + 0.05)])
    prev = -np.inf
    for _ in range(iters):
        trans, occ, err, ll = _baum_welch_pass(x, a, b, e)
        a = trans[0, 1] / max(trans[0, 0] + trans[0, 1], 1e-300)
        b = trans[1, 0] / max(trans[1, 0] + trans[1, 1], 1e-300)
        a, b = float(np.clip(a, 1e-8, 0.5)), float(np.clip(b, 1e-6, 0.999))
        e = np.clip(err / np.maximum(occ, 1e-300), 1e-9, 0.5)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    if e[0] > e[1]:                                   # label the noisier state 'Bad'
        a, b, e = b, a, e[::-1]
    return dict(p_gb=a, p_bg=b, e_good=float(e[0]), e_bad=float(e[1]), loglik=float(ll),
                bad_share=a / (a + b), mean_bad_run=1.0 / b, mean_good_run=1.0 / a)


def bpsk_snr_for_ber(p):
    """Es/N0 (linear) at which uncoded BPSK has bit error probability p."""
    from math import erfc, sqrt
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = sqrt(lo * hi)
        if 0.5 * erfc(sqrt(mid)) > p:
            lo = mid
        else:
            hi = mid
    return sqrt(lo * hi)


def noise_rise_db(e_good, e_bad, floor=1e-6, lo=3.0, hi=20.0):
    snr_g = bpsk_snr_for_ber(max(e_good, floor))
    snr_b = bpsk_snr_for_ber(min(max(e_bad, floor * 10), 0.45))
    return float(np.clip(10 * np.log10(snr_g / snr_b), lo, hi))


def load_fit(path=FIT_PATH):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
