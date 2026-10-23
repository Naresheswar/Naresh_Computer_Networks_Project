"""
CRC utilities for GuardCode.

A CRC (with zero init / zero xor-out) is a linear block code over GF(2):
    crc(m) = m @ G  (mod 2)
so a full frame c = [m | crc(m)] satisfies H @ c = 0 (mod 2) with H = [G^T | I_r].

Besides the ordinary hard CRC check, this module provides the matrix W used by the
*exact soft-syndrome* computation (dual-code / Walsh expansion):

    P(H e = 0) = 2^-r * sum_{u in {0,1}^r} prod_i (1 - 2*p_i)^{(u^T H)_i}

where p_i is the probability that bit i is in error. This is what makes the
"detector-blind error" loss differentiable (see guardcode/losses.py).
"""
import numpy as np

# name: (degree, polynomial without the leading x^r term)
CRC_POLYS = {
    "CRC-4-ITU": (4, 0x3),        # x^4 + x + 1
    "CRC-8": (8, 0x07),           # x^8 + x^2 + x + 1   (CRC-8/SMBUS, ATM HEC)
    "CRC-8-CCITT": (8, 0x07),
    "CRC-16-CCITT": (16, 0x1021), # x^16 + x^12 + x^5 + 1  (HDLC / X.25 / Bluetooth)
}


def crc_bits_reference(msg_bits, r, poly):
    """Bit-serial CRC by polynomial long division (the textbook / lab algorithm).
    msg_bits: 1-D array of 0/1. Returns r CRC bits (MSB first)."""
    reg = 0
    mask = (1 << r) - 1
    for b in msg_bits:
        top = ((reg >> (r - 1)) & 1) ^ int(b)
        reg = (reg << 1) & mask
        if top:
            reg ^= poly
    return np.array([(reg >> (r - 1 - i)) & 1 for i in range(r)], dtype=np.uint8)


class CRC:
    """Vectorised linear CRC for fixed message length k."""

    def __init__(self, k, name="CRC-8"):
        if name not in CRC_POLYS:
            raise ValueError(f"unknown CRC {name}; choose from {list(CRC_POLYS)}")
        self.name = name
        self.k = k
        self.r, self.poly = CRC_POLYS[name]
        self.n = k + self.r
        # generator: row i = CRC of unit vector e_i
        G = np.zeros((k, self.r), dtype=np.uint8)
        for i in range(k):
            e = np.zeros(k, dtype=np.uint8)
            e[i] = 1
            G[i] = crc_bits_reference(e, self.r, self.poly)
        self.G = G
        self.H = np.concatenate([G.T, np.eye(self.r, dtype=np.uint8)], axis=1)  # r x n
        self._W = None

    # ---------- hard operations ----------
    def encode(self, msgs):
        """msgs: (B, k) 0/1 -> frames (B, k+r) with CRC appended."""
        msgs = np.asarray(msgs, dtype=np.uint8)
        crc = (msgs.astype(np.int64) @ self.G) % 2
        return np.concatenate([msgs, crc.astype(np.uint8)], axis=1)

    def syndrome(self, frames):
        frames = np.asarray(frames, dtype=np.int64)
        return (frames @ self.H.T) % 2

    def check(self, frames):
        """True where the CRC passes."""
        return ~self.syndrome(frames).any(axis=1)

    # ---------- soft syndrome support ----------
    @property
    def W(self):
        """(2^r, n) matrix: row u = u^T H mod 2 (the dual code of the CRC frame code)."""
        if self._W is None:
            if self.r > 16:
                raise ValueError("soft syndrome expansion is limited to r <= 16")
            U = ((np.arange(2 ** self.r)[:, None] >> np.arange(self.r - 1, -1, -1)) & 1)
            self._W = (U @ self.H) % 2
        return self._W.astype(np.uint8)

    def prob_syndrome_zero(self, p_err):
        """Exact P(H e = 0) for independent bit errors e_i ~ Bern(p_err_i).
        p_err: (B, n). numpy version (used for analysis); torch version lives in losses.py"""
        t = 1.0 - 2.0 * np.asarray(p_err, dtype=np.float64)    # (B, n)
        W = self.W.astype(bool)                                 # (U, n)
        out = np.empty(t.shape[0])
        for b in range(t.shape[0]):
            prod = np.where(W, t[b][None, :], 1.0).prod(axis=1)
            out[b] = prod.mean()
        return out

    def min_distance(self, max_w=6):
        """Smallest weight of a non-zero frame codeword (the CRC's Hamming distance
        at this length) by exhaustive search up to max_w. Used in the report."""
        from itertools import combinations
        n = self.n
        Hc = self.H.T.astype(np.int64)
        for w in range(1, max_w + 1):
            for idx in combinations(range(n), w):
                if not (Hc[list(idx)].sum(axis=0) % 2).any():
                    return w
        return None


if __name__ == "__main__":
    c = CRC(32, "CRC-8")
    rng = np.random.default_rng(0)
    m = rng.integers(0, 2, (5, 32))
    f = c.encode(m)
    assert c.check(f).all()
    for row in m:  # vectorised == bit-serial
        assert (c.encode(row[None])[0, 32:] == crc_bits_reference(row, 8, 0x07)).all()
    f[0, 3] ^= 1
    print("check after 1 flip:", c.check(f))
    print("CRC-8 Hamming distance on 40-bit frames:", c.min_distance())
