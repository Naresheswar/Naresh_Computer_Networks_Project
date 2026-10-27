"""
Unified link-layer schemes. Every scheme carries the same 32-bit payload plus a CRC:
    CRC-8  (default): 40-bit frame, coded schemes use 70 channel symbols (rate 32/70)
    CRC-16          : 48-bit frame, coded schemes use 84 channel symbols (rate 32/84)
The two CRCs keep separate neural models, thresholds and results (file suffix '_crc16').

Every decoder exposes group posteriors (bytes for the neural decoder, 4-bit blocks for
Hamming, single bits for uncoded) over the frame bits, so that any acceptance rule can
be applied on top:

    gate = None   accept iff CRC passes                            (standard receiver)
    gate = 'llr'  ... and min |LLR| >= tau                          (classic threshold test)
    gate = 'ssg'  ... and Soft-Syndrome-Guard risk R(y) <= tau      (this invention)
    gate = 'ml'   ... and learned gradient-boosted risk score <= tau (guardcode/mlguard.py)

Line-up used in the experiments
    Uncoded+CRC            40 (CRC-8) or 48 (CRC-16) BPSK symbols
    Hamming+CRC            Hamming(7,4) soft decoding (block interleaved on burst channels)
    Hamming+CRC+SSG        same decoder + Soft-Syndrome Guard  (shows SSG is decoder-agnostic)
    Neural+CRC             neural code, plain bit-metric training (ablation)
    Neural+CRC+LLR         neural code + classic min-|LLR| threshold test (prior-art style)
    GuardCode              neural code + Soft-Syndrome Guard (this invention)
    GuardCode-ML           neural code + learned gradient-boosted risk classifier
"""
import os

import numpy as np

from .channels import sigma_from_ebno
from .crc import CRC
from .guard import SoftSyndromeGuard
from .hamming import Hamming74
from .mlguard import MLGuard, model_path
from .model import NeuralCodeNP
from . import mlguard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
K_INFO = 32
CRC_NAME = "CRC-8"                        # default; "CRC-16" is also supported
CRC_CHOICES = ("CRC-8", "CRC-16")
_CRC, _SSG = {}, {}
# bits of every group value, MSB first: _VALUE_BITS[m] has shape (2^m, m)
_VALUE_BITS = {m: ((np.arange(2 ** m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8) for m in (1, 2, 4, 8)}


def crc(name=CRC_NAME):
    if name not in _CRC:
        _CRC[name] = CRC(K_INFO, name)
    return _CRC[name]


def ssg(name=CRC_NAME):
    if name not in _SSG:
        _SSG[name] = SoftSyndromeGuard(crc(name))
    return _SSG[name]


def suffix(crc_name=CRC_NAME):
    """File-name suffix for models, thresholds and results: '' for CRC-8, '_crc16' for CRC-16."""
    return "" if crc_name == CRC_NAME else "_" + crc_name.lower().replace("-", "")


class Scheme:
    name = "base"
    n = 40
    crc_name = CRC_NAME
    gate = None
    tau = None
    ml = None                     # MLGuard instance, used when gate == 'ml'

    @property
    def rate(self):
        return K_INFO / self.n

    def encode(self, frames):
        raise NotImplementedError

    def groups(self, y, h, ebno_db):
        """Return group posteriors (B, G, 2^m) over the frame bits (40 for CRC-8, 48 for CRC-16)."""
        raise NotImplementedError

    def with_gate(self, gate, tau, name, ml=None):
        import copy
        s = copy.copy(self)
        s.gate, s.tau, s.name, s.ml = gate, tau, name, ml
        return s

    def decide(self, y, h, ebno_db):
        """Returns bits, accept-mask (CRC + gate), crc_ok, score (risk or min|LLR|)."""
        P = self.groups(y, h, ebno_db)
        B, G, M = P.shape
        m = int(np.log2(M))
        c = crc(self.crc_name)
        bits = _VALUE_BITS[m][P.argmax(axis=2)].reshape(B, G * m)
        crc_ok = c.check(bits)
        # CRC-16: the 65,536-term risk is only needed where the CRC passes (1.0 elsewhere)
        _, risk = ssg(self.crc_name).risk_groups(P, rows=crc_ok if c.r > 8 else None)
        p1 = np.clip(P @ _VALUE_BITS[m].astype(np.float64), 1e-15, 1 - 1e-15).reshape(B, G * m)
        llr = np.abs(np.log(p1) - np.log1p(-p1)).min(axis=1)
        accept = crc_ok.copy()
        extra = {}
        if self.gate == "llr":
            accept &= llr >= self.tau
        elif self.gate == "ssg":
            accept &= risk <= self.tau
        elif self.gate == "ml":
            extra["ml"] = ml_score = self.ml.score(mlguard.features(self, P, risk, llr, y, h, bits))
            accept &= ml_score <= self.tau
        return bits, accept, crc_ok, dict(llr=llr, risk=risk, p1=p1, **extra)


class Uncoded(Scheme):
    name = "Uncoded+CRC"

    def __init__(self, crc_name=CRC_NAME):
        self.crc_name = crc_name
        self.n = crc(crc_name).n

    def encode(self, frames):
        return 1.0 - 2.0 * frames

    def groups(self, y, h, ebno_db):
        s2 = sigma_from_ebno(ebno_db, self.rate) ** 2
        llr = 2.0 * y * h / s2                      # log P(0)/P(1)
        p1 = 1.0 / (1.0 + np.exp(np.clip(llr, -700, 700)))
        return np.stack([1.0 - p1, p1], axis=2)     # groups of 1 bit


class HammingScheme(Scheme):
    def __init__(self, interleave=False, name=None, crc_name=CRC_NAME):
        self.crc_name = crc_name
        self.code = Hamming74(crc(crc_name).n, interleave)
        self.n = self.code.n
        self.name = name or "Hamming+CRC"

    def encode(self, frames):
        return self.code.encode(frames)

    def groups(self, y, h, ebno_db):
        s2 = sigma_from_ebno(ebno_db, self.rate) ** 2
        return self.code.block_posteriors(y, h, s2)     # (B, frame bits / 4, 16)


class NeuralScheme(Scheme):
    def __init__(self, path, name):
        self.model = NeuralCodeNP(path)
        self.n = self.model.n
        self.name = name
        self.meta = self.model.meta
        self.crc_name = "CRC-16" if self.meta.get("crc", CRC_NAME).startswith("CRC-16") else self.meta.get("crc", CRC_NAME)

    def encode(self, frames):
        return self.model.encode(frames)

    def groups(self, y, h, ebno_db):
        return self.model.byte_posteriors(y * h)        # (B, frame bytes, 256)


def thresholds(profile, crc_name=CRC_NAME):
    """Gate thresholds calibrated by guardcode/evaluate.py (equal false-reject budget)."""
    path = os.path.join(MODELS, f"thresholds_{profile}{suffix(crc_name)}.txt")
    out = dict(llr=4.0, ssg=1e-3, ssg_hamming=1e-3)
    if os.path.exists(path):
        for line in open(path):
            k, v = line.split("=")
            out[k.strip()] = float(v)
    return out


def neural_path(profile, crc_name=CRC_NAME):
    return os.path.join(MODELS, f"base_{profile}{suffix(crc_name)}.npz")


def load_schemes(profile="awgn", models_profile=None, crc_name=CRC_NAME):
    """Ordered line-up for a channel profile ('awgn' | 'burst' | 'fading' | 'crawdad') and CRC
    ('CRC-8' | 'CRC-16'). The measured 'crawdad' channel is a test channel: it reuses the burst
    profile's codes and thresholds, which were never tuned on the real traces."""
    bursty = profile in ("burst", "crawdad")
    mp = models_profile or ("burst" if bursty else "awgn")
    th = thresholds(mp, crc_name)
    ham = HammingScheme(interleave=bursty, crc_name=crc_name)
    s = [Uncoded(crc_name), ham, ham.with_gate("ssg", th["ssg_hamming"], "Hamming+CRC+SSG")]
    base = neural_path(mp, crc_name)
    if os.path.exists(base):
        nb = NeuralScheme(base, "Neural+CRC")
        s += [nb, nb.with_gate("llr", th["llr"], "Neural+CRC+LLR"),
              nb.with_gate("ssg", th["ssg"], "GuardCode")]
        mlp = model_path(mp, crc_name)
        if os.path.exists(mlp) and "ml" in th:
            s.append(nb.with_gate("ml", th["ml"], "GuardCode-ML", ml=MLGuard.load(mlp)))
    return s
