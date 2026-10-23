"""Train the neural code profiles used by the experiments and the dashboard.

    python scripts/train_all.py            (about 15-25 min on a laptop CPU)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guardcode.train import train  # noqa: E402

STEPS = int(os.environ.get("GUARDCODE_STEPS", 5000))
if __name__ == "__main__":
    for prof in sys.argv[1:] or ["awgn", "burst"]:
        print(f"=== training neural code profile '{prof}' ({STEPS} steps) ===", flush=True)
        train(f"base_{prof}", channel=prof, lam=0.0, steps=STEPS, objective="sym", seed=7)
