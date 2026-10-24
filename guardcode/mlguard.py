"""
Learned residual-risk classifier ("GuardCode-ML").

The Soft-Syndrome Guard (guard.py) gives the exact posterior risk R(y) when the decoder's
posteriors are calibrated. A neural decoder is only approximately calibrated, so this
module adds a second opinion: a gradient-boosted decision-tree classifier
(scikit-learn HistGradientBoostingClassifier, open source, BSD-3) trained on CRC-passing
frames to predict "this frame is nevertheless wrong".

Features (all available at the receiver, no extra bits, no ground truth):
    * log10 SSG risk R(y)                      - the invention's own score as a prior
    * min |LLR| over the frame
    * per-group posterior confidence           - sorted log(1 - max_v P_c(v)), c = 1..G
    * per-group runner-up log gap              - sorted, smallest first
    * residual energy ||y - x_hat||^2 / n, and the burst index (max / median window energy)
      after re-encoding the decoded bits

The classifier is a drop-in gate (`gate = "ml"`): accept iff CRC passes and score <= tau,
with tau calibrated on the same retransmission budget as the other gates.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
WINDOW = 7


def model_path(profile):
    return os.path.join(MODELS, f"mlguard_{profile}.joblib")


def features(scheme, P, risk, llr, y, h, bits):
    """P: group posteriors (B, G, 2^m); y, h: received symbols and channel gain; bits: decisions."""
    B, G, M = P.shape
    Ps = np.sort(P, axis=2)
    pmax, p2 = Ps[:, :, -1], Ps[:, :, -2]
    conf = np.sort(np.log10(np.maximum(1.0 - pmax, 1e-12)), axis=1)[:, ::-1]      # least confident first
    gap = np.sort(np.log(np.maximum(pmax, 1e-300)) - np.log(np.maximum(p2, 1e-300)), axis=1)
    x_hat = scheme.encode(bits.astype(np.float64))
    r2 = (y * h - x_hat) ** 2
    n = r2.shape[1]
    w = r2[:, : (n // WINDOW) * WINDOW].reshape(B, -1, WINDOW).mean(axis=2)
    bidx = np.log10(w.max(axis=1) / (np.median(w, axis=1) + 1e-9) + 1e-9)
    return np.column_stack([np.log10(np.maximum(risk, 1e-12)), llr, conf, gap,
                            np.log10(r2.mean(axis=1) + 1e-9), bidx])


class MLGuard:
    """Wrapper around the trained classifier; `score` is P(frame wrong | CRC passed, features)."""

    def __init__(self, clf, meta=None):
        self.clf = clf
        self.meta = meta or {}

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path):
        import joblib
        joblib.dump(dict(clf=self.clf, meta=self.meta), path)

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        return cls(d["clf"], d.get("meta"))


def train_classifier(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=seed)
    return clf.fit(X, y)
