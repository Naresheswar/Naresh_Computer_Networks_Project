"""Draws results/architecture.png - block diagram of the GuardCode link (used in report and patent)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def box(ax, x, y, w, h, text, ref, bold=False, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec="black", lw=1.4 if bold else 1.0))
    ax.text(x + w / 2, y + h / 2 + 0.07, text, ha="center", va="center", fontsize=8.6,
            fontweight="bold" if bold else "normal", family="serif")
    ax.text(x + w / 2, y + 0.1, ref, ha="center", va="bottom", fontsize=7, family="serif", color="#444444")


def arrow(ax, x1, y1, x2, y2, label=None, dashed=False, lx=0, ly=0.08):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1,
                                 color="black", linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="bottom", fontsize=7.2, family="serif")


fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 5.4)
ax.axis("off")
# transmitter row
box(ax, 0.2, 3.7, 1.6, 1.0, "Payload\n32 bits", "(101)")
box(ax, 2.2, 3.7, 1.7, 1.0, "CRC-8 encoder\n(standard g(x))", "(102)")
box(ax, 4.3, 3.7, 1.9, 1.0, "Learned encoder\nprofile k (table)", "(103)")
box(ax, 6.6, 3.7, 1.8, 1.0, "Channel\nAWGN / burst / fading", "(104)")
box(ax, 8.8, 3.7, 1.8, 1.0, "Soft-output decoder\n(neural / classical)", "(105)")
arrow(ax, 1.8, 4.2, 2.2, 4.2)
arrow(ax, 3.9, 4.2, 4.3, 4.2)
ax.text(4.1, 4.8, "40 bits", ha="center", fontsize=7.2, family="serif")
arrow(ax, 6.2, 4.2, 6.6, 4.2)
ax.text(6.4, 4.8, "70 symbols", ha="center", fontsize=7.2, family="serif")
arrow(ax, 8.4, 4.2, 8.8, 4.2)
ax.text(8.6, 4.8, "y", ha="center", fontsize=7.2, family="serif")
# receiver row
box(ax, 8.8, 2.1, 1.8, 1.0, "Hard decision +\nstandard CRC check", "(106)")
box(ax, 6.3, 2.1, 2.1, 1.0, "Soft-Syndrome Guard\nR(y) via dual-code sum", "(107)", bold=True, fill="#f2f2f2")
box(ax, 4.0, 2.1, 1.9, 1.0, "Accept if CRC ok\nand R(y) <= tau", "(108)", bold=True, fill="#f2f2f2")
box(ax, 1.6, 2.1, 2.0, 1.0, "Deliver payload /\nARQ NAK (retransmit)", "(109)")
arrow(ax, 9.7, 3.7, 9.7, 3.1, "p_i = P(b_i=1|y)", lx=0.75, ly=-0.05)
arrow(ax, 8.8, 2.6, 8.4, 2.6)
arrow(ax, 6.3, 2.6, 5.9, 2.6, "R(y)")
arrow(ax, 4.0, 2.6, 3.6, 2.6)
# adaptive loop
box(ax, 6.3, 0.35, 2.1, 1.0, "Burst index from\nre-encoded residuals", "(110)")
box(ax, 4.0, 0.35, 1.9, 1.0, "Profile selector\n(hysteresis) + tau_k", "(111)")
box(ax, 8.8, 0.35, 1.8, 1.0, "Risk ledger\nsum of R (accepted)", "(112)")
arrow(ax, 9.3, 2.1, 7.9, 1.35, dashed=True)
arrow(ax, 6.3, 0.85, 5.9, 0.85)
arrow(ax, 4.95, 1.35, 4.95, 2.1, dashed=True, label="tau_k", lx=0.3, ly=-0.1)
arrow(ax, 4.0, 0.85, 2.6, 0.85, dashed=True)
ax.plot([2.6, 2.6], [0.85, 2.1], "k--", lw=1)
ax.text(1.1, 1.0, "ACK/NAK + profile k\nfed back to (103)", fontsize=7.2, family="serif", ha="center")
arrow(ax, 10.1, 2.1, 9.9, 1.35, dashed=True)
ax.text(11.5, 0.85, "estimate of\nundetected errors\ndelivered", fontsize=7.2, family="serif", ha="center", va="center")
arrow(ax, 10.6, 0.85, 10.9, 0.85)
ax.text(6.2, 5.15, "GuardCode link: transmitter (top), receiver with Soft-Syndrome Guard (middle), adaptive loop (bottom)",
        ha="center", fontsize=10, family="serif", fontweight="bold")
fig.savefig(os.path.join(ROOT, "results", "architecture.png"), dpi=170, bbox_inches="tight")
print("saved")
