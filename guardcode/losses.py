"""
Detector-blind error loss (the core of the GuardCode invention).

An *undetected error* (UE) happens when the decoded frame is wrong but its CRC still
passes. With the decoder's per-bit posterior q_i = P(decoded bit i is correct), and
error indicator e_i ~ Bern(1 - q_i), the probability that the frame passes the CRC is

    P(H e = 0) = 2^-r * sum_u prod_i (2 q_i - 1)^{W_ui}        (exact, dual-code expansion)

and since a correct frame always passes,

    P(UE) = P(H e = 0) - prod_i q_i.

Both terms are smooth functions of the decoder outputs, so P(UE) can be minimised by
back-propagation *through the decoder, the channel and the encoder* while the CRC
itself remains an unmodified, standards-compliant polynomial.
"""
import torch
import torch.nn.functional as F


class SoftCRC(torch.nn.Module):
    def __init__(self, crc):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(crc.W, dtype=torch.bool))  # (2^r, n)
        self.r = crc.r

    def p_syndrome_zero(self, q):
        """q: (B, n) probability each bit is correct. Exact P(Hx = 0)."""
        t = (2.0 * q.double() - 1.0)                       # (B, n)
        # log|t| and sign handled separately for numerical stability
        logabs = torch.log(t.abs().clamp_min(1e-300))      # (B, n)
        neg = (t < 0).double()
        Wd = self.W.double()                               # (U, n)
        log_prod = logabs @ Wd.T                           # (B, U)
        sign = 1.0 - 2.0 * torch.remainder(neg @ Wd.T, 2)  # (B, U)
        return (sign * torch.exp(log_prod)).mean(dim=1)    # (B,)

    def forward(self, logits, frames):
        """Returns (p_ue (B,), p_err (B,)) for decoder logits vs. true frame bits."""
        p1 = torch.sigmoid(logits.double())
        q = torch.where(frames > 0.5, p1, 1.0 - p1).clamp(1e-12, 1.0)
        log_pc = torch.log(q).sum(dim=1)
        p_correct = torch.exp(log_pc)
        p_zero = self.p_syndrome_zero(q)
        p_ue = (p_zero - p_correct).clamp_min(0.0)
        return p_ue, 1.0 - p_correct


def guard_loss(logp, bit_logits, frames, sym_idx, soft_crc, lam, objective="bit"):
    """Total loss = per-byte cross-entropy
                    + lam * (expected undetected errors / expected frame errors).
    logp: (B, C, 256) byte log-posteriors, sym_idx: (B, C) true byte values."""
    if objective == "bit":     # bit-metric training (BICM-style): learns a Gray-like labelling
        ce = F.binary_cross_entropy_with_logits(bit_logits, frames)
    else:                      # symbol-metric training (classic autoencoder)
        ce = F.nll_loss(logp.reshape(-1, logp.shape[-1]), sym_idx.reshape(-1))
    if lam <= 0:
        return ce, ce.detach(), torch.zeros(()), torch.zeros(())
    p_ue, p_err = soft_crc(bit_logits, frames)
    blind = p_ue.sum() / (p_err.sum() + 1e-9)       # fraction of frame errors that slip past the CRC
    total = ce + lam * blind.float()
    return total, ce.detach(), blind.detach(), p_ue.mean().detach()
