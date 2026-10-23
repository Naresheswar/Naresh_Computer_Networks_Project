# GuardCode: Soft-Syndrome Guard for Neural FEC Links

**Course:** BCSE308L / BCSE308P Computer Networks (VIT)
**Student:** Naresh Nireenjan (24BCE2353)

GuardCode is a data-link-layer receiver technique. When a frame passes its CRC, GuardCode computes the exact posterior probability that the frame is nevertheless wrong (an *undetected error*). It needs only two inputs: the decoder's soft output and the dual-code expansion of the standard, unmodified CRC polynomial. Frames whose residual risk is above a threshold are sent back for ARQ retransmission instead of being delivered corrupted.

The project contains:

| Part | Where |
|---|---|
| CRC, Hamming(7,4), channel models (AWGN, Gilbert-Elliott burst, Rayleigh fading) | `guardcode/` |
| Learned neural channel code (PyTorch training, numpy inference) | `guardcode/model.py`, `guardcode/train.py` |
| **Soft-Syndrome Guard**, the invention | `guardcode/guard.py` |
| Channel-adaptive profile controller | `guardcode/adaptive.py` |
| Monte-Carlo link simulator, ARQ goodput, benchmark sweep | `guardcode/linksim.py`, `guardcode/evaluate.py` |
| SQLite database: every frame logged; WAL mode for live use | `guardcode/db.py` → `data/guardcode.db` |
| Live dashboard (Flask + Server-Sent Events) and SQL explorer | `dashboard/` |
| Figures and CSV results | `results/` |
| Project report and provisional patent draft (Word + PDF) | `docs/` |

## Quick start (Windows)

1. Double-click **`setup.bat`**. It:
   - creates `.venv`
   - installs the dependencies
   - builds the 1,000,000-frame database
   - runs the benchmark
2. Double-click **`run_dashboard.bat`**. The live monitor opens at http://127.0.0.1:5000.

Manual equivalent:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\populate_db.py --frames 1000000
python -m guardcode.evaluate --frames 200000
python scripts\analysis.py
python dashboard\app.py
```

To retrain the neural codes, run `retrain_models.bat` or `python scripts\train_all.py`. The pre-trained profiles are already in `models/`.

## The live dashboard

- Frames are transmitted continuously, about 200 per scheme per second, through the selected channel. Every frame is written to `data/guardcode.db`. A retention cap of 10 million rows keeps the newest frames.
- Controls:
  - Eb/N0 slider
  - channel selector: AWGN, burst or fading
  - frame rate
  - a 20-second "burst storm" (bursty interference)
  - pause and reset
- Scoreboard columns: FER, BER, detected and undetected frames, and ARQ goodput for each scheme.
- Live charts:
  - cumulative undetected errors
  - rolling FER
  - goodput
  - the adaptive controller's burst index
- **Database explorer** (`/explorer`): read-only SQL with preset queries over `frames`, `runs`, `benchmark`, `adaptive_log`, and the `v_summary` view.

## Schemes compared

All schemes send a 32-bit payload plus CRC-8 (40-bit frame). Every coded scheme uses 70 channel symbols (rate 32/70).

| Scheme | Description |
|---|---|
| Uncoded+CRC | BPSK, error detection only |
| Hamming+CRC | Hamming(7,4), soft decoding (block-interleaved on burst channels) |
| Hamming+CRC+SSG | Same decoder with the Soft-Syndrome Guard, showing SSG is decoder-agnostic |
| Neural+CRC | Learned neural code, standard CRC acceptance |
| Neural+CRC+LLR | Neural code with the classic min-\|LLR\| threshold test (prior-art style gate) |
| **GuardCode** | Neural code with the Soft-Syndrome Guard |
| **GuardCode-Adaptive** | Neural code with SSG, plus residual-based switching between AWGN and burst profiles |

Both gates are calibrated to the same retransmission budget: 1% of correct frames rejected at 3 dB. Any reduction in undetected errors therefore comes from the better test, not from rejecting more frames.

## Headline results

These numbers come from `results/benchmark.csv`: 300,000 frames per point, Eb/N0 0-7 dB, pooled. Both gates were calibrated to the same retransmission budget.

| Channel | Receiver | Undetected frames |
|---|---|---|
| AWGN | Neural + CRC | 1,559 |
| AWGN | Neural + classic min-\|LLR\| test | 1,428 |
| AWGN | **GuardCode (Neural + SSG)** | **742** (48% fewer than the LLR test) |
| AWGN | Hamming + CRC → + SSG | 2,472 → 486 (80% fewer) |
| Burst | Neural + classic min-\|LLR\| test | 3,054 |
| Burst | **GuardCode** | **1,531** (50% fewer) |
| Burst | Hamming + CRC → + SSG | 4,161 → 2,135 (49% fewer) |

From 3 dB upward, ARQ goodput stays within about 1% of the unguarded receiver.
