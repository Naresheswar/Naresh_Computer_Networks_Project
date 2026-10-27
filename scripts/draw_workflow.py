"""Draws results/workflow.png - the end-to-end workflow of the project (used in the report)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_DATA, C_TX, C_CH, C_RX, C_OUT = "#e8f1fb", "#fdf0e3", "#eeeeee", "#e6f6ee", "#f3ecfb"


def box(ax, x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333"))
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.08, label, fontsize=8, color="#444444", ha="center")


fig, ax = plt.subplots(figsize=(13, 8.0))
ax.set_xlim(0, 13)
ax.set_ylim(1.2, 9.2)
ax.axis("off")
W, H = 2.1, 0.95
# row 1: transmitter path
r1 = 5.8
box(ax, 0.2, r1, W, H, "32-bit payload\n(random messages)", C_DATA)
box(ax, 2.7, r1, W, H, "Append CRC\nCRC-8 (40-bit frame)\nor CRC-16 (48-bit)", C_TX)
box(ax, 5.2, r1, W, H, "FEC encoder\nneural code (70/84 sym.)\nor Hamming(7,4)", C_TX)
box(ax, 7.7, r1, 2.4, H, "Channel\nAWGN | Gilbert-Elliott burst |\nRayleigh fading | CRAWDAD", C_CH, bold=True)
box(ax, 10.6, r1, W, H, "Soft-output decoder\nbyte / block posteriors", C_RX)
for x in (2.3, 4.8, 7.3):
    arrow(ax, x, r1 + H / 2, x + 0.4, r1 + H / 2)
arrow(ax, 10.1, r1 + H / 2, 10.6, r1 + H / 2)
# row 2: receiver decision
r2 = 3.9
box(ax, 10.6, r2, W, H, "CRC check\non hard decision", C_RX)
arrow(ax, 11.65, r1, 11.65, r2 + H)
box(ax, 7.7, r2, 2.4, H, "Soft-Syndrome Guard\nexact risk R(y) via\ndual-code / WHT sum", C_RX, bold=True)
arrow(ax, 10.6, r2 + H / 2, 10.1, r2 + H / 2, "pass")
box(ax, 5.2, r2, W, H, "GuardCode-ML\ngradient-boosted trees\n(optional refinement)", C_RX)
arrow(ax, 7.7, r2 + H / 2, 7.3, r2 + H / 2)
box(ax, 2.7, r2, W, H, "Accept if risk <= tau\n(tau: 1 % false-reject\nbudget at 3 dB)", C_RX, bold=True)
arrow(ax, 5.2, r2 + H / 2, 4.8, r2 + H / 2)
box(ax, 0.2, r2, W, H, "Deliver frame\nor NAK -> ARQ\nretransmission", C_OUT)
arrow(ax, 2.7, r2 + H / 2, 2.3, r2 + H / 2)
ax.text(11.65, r2 - 0.25, "CRC fail -> ARQ retransmission", fontsize=8, color="#444444", ha="center")
# row 3: outputs and evaluation
r3 = 1.6
box(ax, 0.2, r3, W, H, "Outcome per frame\nOK / detected /\nundetected / false reject", C_OUT)
arrow(ax, 1.25, r2, 1.25, r3 + H)
box(ax, 2.7, r3, W, H, "SQLite database\nevery frame logged\n(WAL mode)", C_OUT)
box(ax, 5.2, r3, W, H, "Benchmark sweep\n0-7 dB, 300,000 frames\nper point, CSV + plots", C_OUT)
box(ax, 7.7, r3, 2.4, H, "Comparison tables\nGuardCode vs CRC-only,\nLLR test, Hamming", C_OUT, bold=True)
box(ax, 10.6, r3, W, H, "Live Flask dashboard\n+ SQL explorer", C_OUT)
for x in (2.3, 4.8, 7.3):
    arrow(ax, x, r3 + H / 2, x + 0.4, r3 + H / 2)
arrow(ax, 10.1, r3 + H / 2, 10.6, r3 + H / 2)
# side inputs (offline), above the transmitter row
r0 = 7.35
box(ax, 7.7, r0, 2.4, 0.85, "Real traces: CRAWDAD niit/bit_errors\n-> Baum-Welch Gilbert-Elliott fit", C_DATA)
ax.annotate("", xy=(8.9, r1 + H), xytext=(8.9, r0), arrowprops=dict(arrowstyle="-|>", ls="--", color="#777777", lw=1))
box(ax, 4.2, r0, 3.1, 0.85, "Offline training: PyTorch neural codes\n(encoder/decoder), scikit-learn ML gate", C_DATA)
ax.annotate("", xy=(6.25, r1 + H), xytext=(6.25, r0), arrowprops=dict(arrowstyle="-|>", ls="--", color="#777777", lw=1))
ax.text(6.5, 8.85, "GuardCode workflow: transmitter (top), receiver decision (middle), evaluation and outputs (bottom)",
        ha="center", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "workflow.png"), dpi=170)
print("wrote results/workflow.png")
