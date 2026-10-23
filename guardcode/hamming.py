"""
Classical baseline: Hamming(7,4) with soft-decision maximum-likelihood decoding per
7-symbol block, optionally with a block interleaver (for burst channels).

A 40-bit frame (32 info + 8 CRC) -> 10 Hamming blocks -> 70 BPSK symbols (rate 4/7).
"""
import numpy as np

# systematic generator G = [I4 | P]
_P = np.array([[1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 1, 1]], dtype=np.uint8)
G74 = np.concatenate([np.eye(4, dtype=np.uint8), _P], axis=1)
_ALL_MSG = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
CODEBOOK = (_ALL_MSG @ G74) % 2                   # (16, 7)
CODEBOOK_BPSK = 1.0 - 2.0 * CODEBOOK              # bit 0 -> +1, bit 1 -> -1


def interleaver_perm(n_blocks, block_len=7):
    """Row-in / column-out block interleaver permutation of length n_blocks*block_len."""
    idx = np.arange(n_blocks * block_len).reshape(n_blocks, block_len)
    return idx.T.reshape(-1)


class Hamming74:
    def __init__(self, n_bits, interleave=False):
        assert n_bits % 4 == 0
        self.n_bits = n_bits
        self.blocks = n_bits // 4
        self.n = self.blocks * 7
        self.interleave = interleave
        self.perm = interleaver_perm(self.blocks) if interleave else np.arange(self.n)
        self.inv = np.argsort(self.perm)

    def encode(self, frames):
        B = frames.shape[0]
        blk = frames.reshape(B, self.blocks, 4)
        cw = (blk.astype(np.int64) @ G74) % 2
        x = (1.0 - 2.0 * cw).reshape(B, self.n)
        return x[:, self.perm]

    def decode(self, y, h=None):
        """Soft ML per block. Returns (bits (B,n_bits), reliability (B,) = min block margin)."""
        if h is not None:
            y = y * h  # matched filter with known gain
        y = y[:, self.inv]
        B = y.shape[0]
        yb = y.reshape(B, self.blocks, 7)
        corr = yb @ CODEBOOK_BPSK.T                       # (B, blocks, 16)
        best = corr.argmax(axis=2)
        srt = np.sort(corr, axis=2)
        margin = (srt[:, :, -1] - srt[:, :, -2]).min(axis=1)
        bits = _ALL_MSG[best].reshape(B, self.n_bits)
        return bits, margin

    def bit_posteriors(self, y, h, sigma2):
        """Exact per-bit posteriors P(bit=1|y) from block-wise soft decoding (AWGN metric)."""
        if h is not None:
            y = y * h
        y = y[:, self.inv]
        B = y.shape[0]
        yb = y.reshape(B, self.blocks, 7)
        # log-likelihood of codeword c: -|y - h c|^2 / 2s2 = const + (h y . c)/s2  (codewords have equal energy)
        metric = (yb @ CODEBOOK_BPSK.T) / sigma2
        metric -= metric.max(axis=2, keepdims=True)
        p = np.exp(metric)
        p /= p.sum(axis=2, keepdims=True)                       # (B, blocks, 16)
        p1 = p @ _ALL_MSG.astype(np.float64)                     # (B, blocks, 4)
        return p1.reshape(B, self.n_bits)

    def block_posteriors(self, y, h, sigma2):
        """Exact posterior over the 16 messages of every Hamming block: (B, blocks, 16)."""
        if h is not None:
            y = y * h
        y = y[:, self.inv]
        B = y.shape[0]
        metric = (y.reshape(B, self.blocks, 7) @ CODEBOOK_BPSK.T) / sigma2
        metric -= metric.max(axis=2, keepdims=True)
        p = np.exp(metric)
        return p / p.sum(axis=2, keepdims=True)
