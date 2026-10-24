"""
Unified link-layer schemes. Every scheme carries the same 32-bit payload + CRC-8
(a 40-bit frame). All coded schemes use 70 channel symbols (rate 32/70).

Every decoder exposes group posteriors (bytes for the neural decoder, 4-bit blocks for
Hamming, single bits for uncoded) over the 40 frame bits, so that any acceptance rule can
be applied on top:

    gate = None   accept iff CRC passes                            (standard receiver)
    gate = 'llr'  ... and min |LLR| >= tau                          (classic threshold test)
    gate = 'ssg'  ... and Soft-Syndrome-Guard risk R(y) <= tau      (this invention)
    gate = 'ml'   ... and learned gradient-boosted risk score <= tau (guardcode/mlguard.py)

Line-up used in the experiments
    Uncoded+CRC            40 BPSK symbols
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
CRC_NAME = "CRC-8"
_CRC = CRC(K_INFO, CRC_NAME)
_SSG = SoftSyndromeGuard(_CRC)
# bits of every group value, MSB first: _VALUE_BITS[m] has shape (2^m, m)
_VALUE_BITS = {m: ((np.arange(2 ** m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8) for m in (1, 2, 4, 8)}


def crc():
    return _CRC


def ssg():
    return _SSG


class Scheme:
    name = "base"
    n = 40
    gate = None
    tau = None
    ml = None                     # MLGuard instance, used when gate == 'ml'

    @property
    def rate(self):
        return K_INFO / self.n

    def encode(self, frames):
        raise NotImplementedError

    def groups(self, y, h, ebno_db):
        """Return group posteriors (B, G, 2^m) over the 40 frame bits."""
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
        dec, risk = _SSG.risk_groups(P)
        bits = _VALUE_BITS[m][dec].reshape(B, G * m)
        crc_ok = _CRC.check(bits)
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
    n = 40

    def encode(self, frames):
        return 1.0 - 2.0 * frames

    def groups(self, y, h, ebno_db):
        s2 = sigma_from_ebno(ebno_db, self.rate) ** 2
        llr = 2.0 * y * h / s2                      # log P(0)/P(1)
        p1 = 1.0 / (1.0 + np.exp(np.clip(llr, -700, 700)))
        return np.stack([1.0 - p1, p1], axis=2)     # groups of 1 bit


class HammingScheme(Scheme):
    def __init__(self, interleave=False, name=None):
        self.code = Hamming74(40, interleave)
        self.n = self.code.n
        self.name = name or "Hamming+CRC"

    def encode(self, frames):
        return self.code.encode(frames)

    def groups(self, y, h, ebno_db):
        s2 = sigma_from_ebno(ebno_db, self.rate) ** 2
        return self.code.block_posteriors(y, h, s2)     # (B, 10, 16)


class NeuralScheme(Scheme):
    def __init__(self, path, name):
        self.model = NeuralCodeNP(path)
        self.n = self.model.n
        self.name = name
        self.meta = self.model.meta

    def encode(self, frames):
        return self.model.encode(frames)

    def groups(self, y, h, ebno_db):
        return self.model.byte_posteriors(y * h)        # (B, 5, 256)


def thresholds(profile):
    """Gate thresholds calibrated by guardcode/evaluate.py (equal false-reject budget)."""
    path = os.path.join(MODELS, f"thresholds_{profile}.txt")
    out = dict(llr=4.0, ssg=1e-3, ssg_hamming=1e-3)
    if os.path.exists(path):
        for line in open(path):
            k, v = line.split("=")
            out[k.strip()] = float(v)
    return out


def load_schemes(profile="awgn", models_profile=None):
    """Ordered line-up for a channel profile ('awgn' | 'burst' | 'fading')."""
    mp = models_profile or ("burst" if profile == "burst" else "awgn")
    th = thresholds(mp)
    ham = HammingScheme(interleave=(profile == "burst"))
    s = [Uncoded(), ham, ham.with_gate("ssg", th["ssg_hamming"], "Hamming+CRC+SSG")]
    base = os.path.join(MODELS, f"base_{mp}.npz")
    if os.path.exists(base):
        nb = NeuralScheme(base, "Neural+CRC")
        s += [nb, nb.with_gate("llr", th["llr"], "Neural+CRC+LLR"),
              nb.with_gate("ssg", th["ssg"], "GuardCode")]
        if os.path.exists(model_path(mp)) and "ml" in th:
            s.append(nb.with_gate("ml", th["ml"], "GuardCode-ML", ml=MLGuard.load(model_path(mp))))
    return s
