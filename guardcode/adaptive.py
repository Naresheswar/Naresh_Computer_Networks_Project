"""
Channel-adaptive profile controller (dependent claim of the GuardCode invention).

GuardCode keeps several detector-aware code profiles, each trained for a channel
class (here: 'awgn' and 'burst'). The receiver estimates which class the channel is
currently in WITHOUT pilots or extra bits:

  1. decode the frame, then RE-ENCODE the decoded bytes with the active profile;
  2. residual r = y - x_hat; residual energy per window of w symbols;
  3. burst index  b = log10( max window energy / median window energy );
  4. EWMA over frames + hysteresis -> requested profile, fed back to the
     transmitter piggy-backed on the ARQ ACK (1 bit), applied to the next block.

Gilbert-Elliott bursts concentrate noise in a few windows, so b separates the
two channel classes cleanly.
"""
import numpy as np


def burst_index(scheme, y, bits, h=None, window=7):
    x_hat = scheme.encode(bits.astype(np.float64))
    yy = y * h if h is not None else y
    r2 = (yy - x_hat) ** 2
    B, n = r2.shape
    w = r2[:, : (n // window) * window].reshape(B, -1, window).mean(axis=2)
    med = np.median(w, axis=1) + 1e-9
    return np.log10(w.max(axis=1) / med)


class AdaptiveController:
    def __init__(self, schemes, hi=0.44, lo=0.40, alpha=0.02, start="awgn"):
        self.schemes = schemes            # {'awgn': Scheme, 'burst': Scheme}
        self.hi, self.lo, self.alpha = hi, lo, alpha
        self.active = start
        self.ewma = None
        self.switches = 0

    @property
    def scheme(self):
        return self.schemes[self.active]

    def observe(self, y, bits, h=None):
        b = burst_index(self.scheme, y, bits, h)
        for v in b:                        # per-frame EWMA
            self.ewma = v if self.ewma is None else (1 - self.alpha) * self.ewma + self.alpha * v
        want = self.active
        if self.active == "awgn" and self.ewma > self.hi:
            want = "burst"
        elif self.active == "burst" and self.ewma < self.lo:
            want = "awgn"
        if want != self.active:
            self.active = want
            self.switches += 1
        return float(self.ewma), self.active
