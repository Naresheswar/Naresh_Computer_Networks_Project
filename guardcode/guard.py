"""
Soft-Syndrome Guard (SSG) - the receiver-side core of the GuardCode invention.

The decoder delivers posteriors over GROUPS of m frame bits (a byte for the neural
decoder, a 4-bit Hamming block, or a single bit for uncoded BPSK):
        P_c(v) = P(group c of the frame = v | y),   v in {0..2^m - 1}.
The hard decision x takes the most likely value of every group.

Knowing that the transmitted frame b is a CRC codeword (H b = 0), the probability that
the decision is correct is

        P(b = x | y, Hb = 0) = prod_c P_c(x_c)  /  P(Hb = 0 | y)

and the normaliser is computed EXACTLY by a dual-code expansion in which each group
contributes its Walsh-Hadamard transform:

        P(Hb = 0 | y) = 2^-r * sum_{u in {0,1}^r}  prod_c  F_c( s_c(u) ),
        F_c(s) = sum_v P_c(v) (-1)^{popcount(s & v)}          (WHT of the group posterior)
        s_c(u) = the m bits of u^T H that fall in group c.

Using group posteriors (instead of independent bits) keeps the correlation between
bits of the same byte/block, which makes R(y) far better calibrated.

The residual risk  R(y) = 1 - P(b = x | y, Hb = 0)  is the posterior probability that a
frame which PASSES the CRC is nevertheless wrong. The guard rejects (requests ARQ
retransmission of) CRC-passing frames with R(y) > tau. No extra bits are sent and the
CRC polynomial is not modified.
"""
import numpy as np


_HAD = {}


def hadamard(n):
    """Sylvester Hadamard matrix with entries (-1)^popcount(s & v)."""
    if n not in _HAD:
        v = np.arange(n)
        bits = np.bitwise_and(v[:, None], v[None, :])
        par = np.zeros_like(bits)
        while bits.any():
            par ^= bits & 1
            bits >>= 1
        _HAD[n] = (1 - 2 * par).astype(np.float64)
    return _HAD[n]


def fwht(a):
    """Walsh-Hadamard transform along the last axis (length 2^m), unnormalised:
    out[s] = sum_v a[v] * (-1)^{popcount(s & v)}.  (A dense matrix product is the
    fastest form in numpy for the group sizes used here, 2^m <= 256.)"""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[-1]
    return (a.reshape(-1, n) @ hadamard(n)).reshape(a.shape)


class SoftSyndromeGuard:
    def __init__(self, crc):
        self.crc = crc
        self.W = crc.W.astype(np.int64)            # (2^r, n) rows u^T H
        self._idx = {}

    def _mask_index(self, m):
        """(U, G) index of s_c(u) for groups of m bits (MSB first inside a group)."""
        if m not in self._idx:
            U, n = self.W.shape
            G = n // m
            w = self.W.reshape(U, G, m)
            weights = 1 << np.arange(m - 1, -1, -1)
            self._idx[m] = (w * weights).sum(axis=2)    # (U, G)
        return self._idx[m]

    def risk_groups(self, P):
        """P: (B, G, 2^m) group posteriors. Returns (decision values (B,G), risk (B,))."""
        P = np.asarray(P, dtype=np.float64)
        B, G, M = P.shape
        m = int(np.log2(M))
        P = P / P.sum(axis=2, keepdims=True)
        dec = P.argmax(axis=2)
        log_px = np.log(np.maximum(np.take_along_axis(P, dec[..., None], axis=2)[..., 0], 1e-300)).sum(axis=1)
        F = fwht(P)                                     # (B, G, M)
        idx = self._mask_index(m)                       # (U, G)
        U = idx.shape[0]
        if m == 1:
            # single-bit groups: prod_i t_i^{W_ui} with t = 1 - 2p, via log-magnitude / sign matmuls
            t = F[:, :, 1]
            Wf = idx.astype(np.float64).T                  # (G, U)
            mag = np.exp(np.log(np.maximum(np.abs(t), 1e-300)) @ Wf)
            sign = 1.0 - 2.0 * np.mod((t < 0).astype(np.float64) @ Wf, 2)
            pz = (sign * mag).mean(axis=1)
        else:
            acc = np.ones((B, U))
            for c in range(G):                          # product over groups, one gather each
                acc *= F[:, c, :][:, idx[:, c]]
            pz = acc.mean(axis=1)
        pz = np.maximum(pz, np.exp(log_px))            # numerical guard: P(code) >= P(x)
        risk = np.clip(1.0 - np.exp(log_px) / pz, 0.0, 1.0)
        return dec, risk

    # bit-level version (groups of 1 bit) kept for reference / ablation
    def risk(self, p1, bits=None):
        p1 = np.clip(np.asarray(p1, dtype=np.float64), 1e-15, 1 - 1e-15)
        P = np.stack([1 - p1, p1], axis=2)
        return self.risk_groups(P)[1]
