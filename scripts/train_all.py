"""Train the neural code profiles used by the experiments and the dashboard.

    python scripts/train_all.py                      CRC-8 profiles (about 15-25 min on a laptop CPU)
    python scripts/train_all.py --crc CRC-16         CRC-16 profiles (48-bit frames, 84 symbols)
    python scripts/train_all.py awgn --crc CRC-16    one profile only
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guardcode.train import train  # noqa: E402

from guardcode.schemes import suffix  # noqa: E402

STEPS = int(os.environ.get("GUARDCODE_STEPS", 5000))
if __name__ == "__main__":
    args = sys.argv[1:]
    crc_name = "CRC-8"
    if "--crc" in args:
        i = args.index("--crc")
        crc_name = args[i + 1]
        del args[i:i + 2]
    n_sym = 70 if crc_name == "CRC-8" else 84          # 14 symbols per frame byte
    for prof in args or ["awgn", "burst"]:
        print(f"=== training neural code profile '{prof}' with {crc_name} ({STEPS} steps) ===", flush=True)
        train(f"base_{prof}{suffix(crc_name)}", channel=prof, lam=0.0, steps=STEPS, objective="sym", seed=7,
              crc_name=crc_name, n_sym=n_sym)
