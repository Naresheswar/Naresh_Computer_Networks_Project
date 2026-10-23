"""
Neural encoder / decoder ("learned constellation + frame-level decoder").

Encoder : the 40-bit frame (32 payload + 8 CRC) is cut into C = 5 bytes. Each byte
          (256 values) is mapped by a *learned* table to a point in R^14 (a learned
          (14, 8) block code / constellation), the 5 points are concatenated and the
          70-symbol frame is power-normalised. An optional fixed interleaver spreads
          each byte across the frame (burst profile).
Decoder : one MLP sees all 70 received symbols at once and outputs, for each byte,
          256 log-posteriors. Because it sees the whole frame it can exploit the CRC
          structure across bytes - which is exactly what the detector-blind loss
          teaches it to do.

Training uses PyTorch (guardcode/train.py). Weights are exported to a plain .npz so
the simulator and the live dashboard run inference with numpy only (NeuralCodeNP).
"""
import json
import numpy as np

CHUNK = 8                                      # bits per learned symbol
_IDX_BITS = ((np.arange(2 ** CHUNK)[:, None] >> np.arange(CHUNK - 1, -1, -1)) & 1).astype(np.uint8)  # (256, 8)


def frame_perm(n_sym, chunks, interleave):
    if not interleave:
        return np.arange(n_sym)
    per = n_sym // chunks
    return np.arange(n_sym).reshape(chunks, per).T.reshape(-1)


try:
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self, n_bits, n_sym, interleave=False):
            super().__init__()
            self.C = n_bits // CHUNK
            self.per = n_sym // self.C
            self.table = nn.Parameter(torch.randn(2 ** CHUNK, self.per))
            self.register_buffer("pow2", 2 ** torch.arange(CHUNK - 1, -1, -1))
            self.register_buffer("perm", torch.as_tensor(frame_perm(n_sym, self.C, interleave)))

        def forward(self, bits):
            B = bits.shape[0]
            idx = (bits.view(B, self.C, CHUNK).long() * self.pow2).sum(-1)      # (B, C)
            x = self.table[idx].reshape(B, -1)
            x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt()
            return x[:, self.perm]

    class Decoder(nn.Module):
        def __init__(self, n_sym, n_bits, hidden=512, layers=3):
            super().__init__()
            self.C = n_bits // CHUNK
            mods, d = [], n_sym
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ELU()]
                d = hidden
            mods += [nn.Linear(d, self.C * 2 ** CHUNK)]
            self.net = nn.Sequential(*mods)
            self.register_buffer("idx_bits", torch.as_tensor(_IDX_BITS, dtype=torch.float32))

        def forward(self, y):
            """Returns (symbol log-posteriors (B,C,256), bit logits (B, C*8))."""
            B = y.shape[0]
            logp = torch.log_softmax(self.net(y).view(B, self.C, 2 ** CHUNK), dim=-1)
            p1 = torch.exp(logp) @ self.idx_bits                               # (B, C, 8)
            p1 = p1.clamp(1e-7, 1 - 1e-7)
            bit_logits = (torch.log(p1) - torch.log1p(-p1)).reshape(B, -1)
            return logp, bit_logits

    def export_npz(enc, dec, path, meta):
        arrs = {"enc_table": enc.table.detach().cpu().numpy().astype(np.float32),
                "perm": enc.perm.cpu().numpy()}
        lin = [m for m in dec.net if isinstance(m, nn.Linear)]
        for i, l in enumerate(lin):
            arrs[f"dec_W{i}"] = l.weight.detach().cpu().numpy().astype(np.float32)
            arrs[f"dec_b{i}"] = l.bias.detach().cpu().numpy().astype(np.float32)
        arrs["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrs)
except ImportError:
    torch = None


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NeuralCodeNP:
    """numpy inference for an exported GuardCode / neural-baseline model."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.table = z["enc_table"].astype(np.float64)
        self.perm = z["perm"].astype(np.int64)
        self.inv = np.argsort(self.perm)
        self.dec = []
        i = 0
        while f"dec_W{i}" in z:
            self.dec.append((z[f"dec_W{i}"].astype(np.float32), z[f"dec_b{i}"].astype(np.float32)))
            i += 1
        self.n = self.meta["n_sym"]
        self.n_bits = self.meta["n_bits"]
        self.C = self.n_bits // CHUNK
        self.pow2 = 2 ** np.arange(CHUNK - 1, -1, -1)

    def encode(self, frames):
        B = frames.shape[0]
        idx = (frames.reshape(B, self.C, CHUNK).astype(np.int64) * self.pow2).sum(-1)
        x = self.table[idx].reshape(B, -1)
        x = x / np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        return x[:, self.perm]

    def logits(self, y):
        x = y.astype(np.float32)
        for j, (W, b) in enumerate(self.dec):
            x = x @ W.T + b
            if j < len(self.dec) - 1:
                x = _elu(x)
        return x.reshape(y.shape[0], self.C, 2 ** CHUNK)

    def byte_posteriors(self, y):
        lg = self.logits(y).astype(np.float64)
        lg -= lg.max(axis=2, keepdims=True)
        p = np.exp(lg)
        return p / p.sum(axis=2, keepdims=True)          # (B, C, 256)

    def bit_posteriors(self, y):
        lg = self.logits(y).astype(np.float64)
        lg -= lg.max(axis=2, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(axis=2, keepdims=True)
        return (p @ _IDX_BITS.astype(np.float64)).reshape(y.shape[0], -1)

    def decode(self, y, h=None):
        """Returns (bits (B, n_bits), reliability (B,)).
        reliability = smallest log-posterior gap between best and runner-up byte."""
        if h is not None:
            y = y * h
        lg = self.logits(y)
        if self.meta.get("decision", "byte") == "bit":
            # bit-wise MAP decisions from byte posteriors
            m = lg.max(axis=2, keepdims=True)
            p = np.exp(lg - m)
            p /= p.sum(axis=2, keepdims=True)
            p1 = p @ _IDX_BITS.astype(np.float32)                     # (B, C, 8)
            bits = (p1 > 0.5).astype(np.uint8).reshape(y.shape[0], -1)
            llr = np.abs(np.log(np.clip(p1, 1e-12, 1)) - np.log(np.clip(1 - p1, 1e-12, 1)))
            return bits, llr.reshape(y.shape[0], -1).min(axis=1).astype(np.float64)
        best = lg.argmax(axis=2)
        part = np.partition(lg, -2, axis=2)
        gap = (part[:, :, -1] - part[:, :, -2]).min(axis=1)
        bits = _IDX_BITS[best].reshape(y.shape[0], -1)
        return bits, gap.astype(np.float64)
