"""
Channel models (real-valued, unit-energy symbols).

  * AWGN                      - memoryless thermal noise
  * Gilbert-Elliott (GE)      - two-state Markov burst-noise channel (Good / Bad)
  * Rayleigh block fading     - per-frame fading gain with perfect receiver CSI

Eb/N0 convention: every scheme is compared at the same energy per *information* bit.
For a frame carrying k_info information bits in n channel symbols, Es/N0 = Eb/N0 * k_info/n
and the per-dimension noise variance is sigma^2 = 1 / (2 * Es/N0).

Each channel has a numpy implementation (simulation / dashboard) and a torch
implementation (differentiable training).
"""
import numpy as np


def sigma_from_ebno(ebno_db, rate):
    esno = 10 ** (np.asarray(ebno_db, dtype=np.float64) / 10.0) * rate
    return np.sqrt(1.0 / (2.0 * esno))


class ChannelConfig:
    def __init__(self, kind="awgn", p_gb=0.03, p_bg=0.25, bad_extra_db=8.0):
        self.kind = kind            # 'awgn' | 'burst' | 'fading'
        self.p_gb = p_gb            # P(Good -> Bad) per symbol
        self.p_bg = p_bg            # P(Bad -> Good) per symbol
        self.bad_extra_db = bad_extra_db  # extra noise power in Bad state

    def to_dict(self):
        return dict(kind=self.kind, p_gb=self.p_gb, p_bg=self.p_bg, bad_extra_db=self.bad_extra_db)


def ge_states(B, n, p_gb, p_bg, rng):
    """Sample Gilbert-Elliott state sequences, (B, n) bool, True = Bad."""
    pi_bad = p_gb / (p_gb + p_bg)
    s = np.empty((B, n), dtype=bool)
    s[:, 0] = rng.random(B) < pi_bad
    u = rng.random((B, n))
    for t in range(1, n):
        prev = s[:, t - 1]
        s[:, t] = np.where(prev, u[:, t] >= p_bg, u[:, t] < p_gb)
    return s


def noise_scale(B, n, cfg, rng):
    """Per-symbol noise multiplier (1 in Good, 10^(extra/20) in Bad)."""
    if cfg.kind != "burst":
        return np.ones((B, n))
    st = ge_states(B, n, cfg.p_gb, cfg.p_bg, rng)
    return np.where(st, 10 ** (cfg.bad_extra_db / 20.0), 1.0)


def transmit_np(x, ebno_db, rate, cfg, rng):
    """x: (B, n) unit-energy real symbols. Returns (y, h, scale) where the receiver
    gets y and knows h (fading gain). scale is returned for analysis only."""
    B, n = x.shape
    sigma = sigma_from_ebno(ebno_db, rate)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64).reshape(-1, 1) if np.ndim(sigma) else sigma, (B, 1))
    scale = noise_scale(B, n, cfg, rng)
    if cfg.kind == "fading":
        h = np.sqrt(rng.exponential(1.0, (B, 1)))  # Rayleigh amplitude, E[h^2]=1
    else:
        h = np.ones((B, 1))
    y = h * x + sigma * scale * rng.standard_normal((B, n))
    return y, np.broadcast_to(h, (B, n)), scale


# ---------------------------------------------------------------- torch
try:
    import torch

    def transmit_torch(x, ebno_db, rate, cfg, gen=None):
        """Differentiable channel. ebno_db: float or (B,) tensor."""
        B, n = x.shape
        dev = x.device
        ebno = torch.as_tensor(ebno_db, dtype=x.dtype, device=dev)
        if ebno.dim() == 0:
            ebno = ebno.expand(B)
        sigma = torch.sqrt(1.0 / (2.0 * (10 ** (ebno / 10.0)) * rate)).unsqueeze(1)
        scale = torch.ones_like(x)
        if cfg.kind == "burst":
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
            st = ge_states(B, n, cfg.p_gb, cfg.p_bg, rng)
            scale = torch.as_tensor(np.where(st, 10 ** (cfg.bad_extra_db / 20.0), 1.0), dtype=x.dtype, device=dev)
        if cfg.kind == "fading":
            h = torch.sqrt(torch.distributions.Exponential(1.0).sample((B, 1)).to(dev))
        else:
            h = torch.ones(B, 1, device=dev, dtype=x.dtype)
        y = h * x + sigma * scale * torch.randn(B, n, device=dev, dtype=x.dtype, generator=gen)
        return y, h.expand(B, n)
except ImportError:  # torch is optional at runtime (dashboard runs on numpy only)
    torch = None
