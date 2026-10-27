# GuardCode: Soft-Syndrome Guard for Neural FEC Links

**Course:** BCSE308L / BCSE308P Computer Networks (VIT)
**Student:** Naresh Nireenjan (24BCE2353)

GuardCode is a data-link-layer receiver technique. When a frame passes its CRC, GuardCode computes the exact posterior probability that the frame is nevertheless wrong (an *undetected error*). It needs only two inputs: the decoder's soft output and the dual-code expansion of the standard, unmodified CRC polynomial. Frames whose residual risk is above a threshold are sent back for ARQ retransmission instead of being delivered corrupted.

The project contains:

| Part | Where |
|---|---|
| CRC-8 and CRC-16, Hamming(7,4), channel models (AWGN, Gilbert-Elliott burst, Rayleigh fading) | `guardcode/` |
| Real-world burst channel fitted to the CRAWDAD `niit/bit_errors` 802.15.4 traces | `guardcode/crawdad.py`, `scripts/fit_crawdad.py` |
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
| **GuardCode-ML** | Neural code with a learned risk classifier (scikit-learn `HistGradientBoostingClassifier`, `guardcode/mlguard.py`) trained on CRC-passing frames. Features: SSG risk, min-\|LLR\|, per-byte confidence, residual energy, burst index. Same 1% false-reject budget |
| **GuardCode-Adaptive** | Neural code with SSG, plus residual-based switching between AWGN and burst profiles |

Both gates are calibrated to the same retransmission budget: 1% of correct frames rejected at 3 dB. Any reduction in undetected errors therefore comes from the better test, not from rejecting more frames.

## Headline results

These numbers come from `results/benchmark.csv`: 300,000 frames per point, Eb/N0 0-7 dB, pooled. Both gates were calibrated to the same retransmission budget.

| Channel | Receiver | Undetected frames |
|---|---|---|
| AWGN | Neural + CRC | 1,559 |
| AWGN | Neural + classic min-\|LLR\| test | 1,428 |
| AWGN | **GuardCode (Neural + SSG)** | **742** (48% fewer than the LLR test) |
| AWGN | **GuardCode-ML** (Neural + learned gradient-boosted gate) | **324** (56% fewer than GuardCode) |
| AWGN | Hamming + CRC → + SSG | 2,472 → 486 (80% fewer) |
| Burst | Neural + classic min-\|LLR\| test | 3,054 |
| Burst | **GuardCode** | **1,531** (50% fewer) |
| Burst | **GuardCode-ML** | **1,165** (24% fewer than GuardCode) |
| Burst | Hamming + CRC → + SSG | 4,161 → 2,135 (49% fewer) |
| Real 802.15.4 (CRAWDAD) | Neural + classic min-\|LLR\| test | 1,846 |
| Real 802.15.4 (CRAWDAD) | **GuardCode** | **929** (50% fewer) |
| Real 802.15.4 (CRAWDAD) | **GuardCode-ML** | **768** (17% fewer than GuardCode) |
| Real 802.15.4 (CRAWDAD) | Hamming + CRC → + SSG | 2,806 → 1,027 (63% fewer) |

On the CRAWDAD channel nothing was re-tuned: it uses the burst profile's codes, gate thresholds and ML model, so these rows test how GuardCode generalises to measured burst statistics.

From 3 dB upward, ARQ goodput stays within about 1% of the unguarded receiver.

## GuardCode-ML (learned gate)

`python scripts/train_mlguard.py --frames 2000000` trains one gradient-boosted classifier per channel profile (`models/mlguard_<profile>.joblib`; hold-out AUC/AP in `models/mlguard_*_report.json`). Then run `python -m guardcode.evaluate` to calibrate its threshold and benchmark it. Training data and benchmark seeds are disjoint. It needs scikit-learn at run time; without the `.joblib` files the project behaves as before.

## Real-world channel: CRAWDAD niit/bit_errors

The `crawdad` channel profile is built from **real measured error traces**: the CRAWDAD `niit/bit_errors` dataset (Iqbal, Shahzad, Khayam, Cho; [IEEE DataPort](https://ieee-dataport.org/open-access/crawdad-niitbiterrors), DOI 10.15783/C7S88K).

- **What the data is:** 22 traces from 802.15.4 MicaZ motes at NUST: 20-byte frames, 5–12 m from the base station, in four indoor setups (OutRoom, PhdLab, Stair, Up_Floor). That is about 650,000 frames and 104 million payload bits.
- **Format:** one frame per line in hexadecimal. A payload byte `AA` means "received correctly", and XOR with `AA` gives the bits in error.
- **The fit:** `scripts/fit_crawdad.py` fits a two-state Gilbert-Elliott hidden Markov model to every trace by Baum-Welch (a numba kernel; checked on synthetic data, where it recovers the true parameters to within a few %).
- **What the data shows:** errors come in short bursts. The channel leaves the bad state with probability about 0.25 per bit, so a burst lasts about 4 bits, and bits inside a burst are wrong about half the time. That is consistent with one 802.15.4 DSSS symbol carrying 4 bits.
- **How the channel uses it:** every simulated frame draws one measured trace, weighted by its frame count, and uses that trace's burst timing. The bad-state noise rise is the BPSK-equivalent of the fitted good and bad error probabilities (3–20 dB). `results/crawdad_traces.png` compares measured and modelled burst lengths and errors per frame. The model reproduces short bursts closely but still underestimates the rare frames with 25+ errors.
- **Setting it up:** the 802.15.4 traces are in `data/crawdad/`. Check the dataset's terms on IEEE DataPort before making this repository public. To refit from scratch, download them (free IEEE account) into `data/crawdad/Traces_802.15.4/` and run `python scripts/fit_crawdad.py`. The 802.11 traces in the same dataset are pcap files of about 1 GB each and are not used.

```
python scripts/fit_crawdad.py                       # -> models/crawdad_ge.json, results/crawdad_traces.*
python -m guardcode.evaluate --profiles crawdad     # benchmark on the measured channel
```

In the dashboard, pick the **Real 802.15.4 (CRAWDAD)** channel.

## CRC-8 and CRC-16

Every part of the project accepts `CRC-8` (x⁸ + x² + x + 1, the default) or `CRC-16` (x¹⁶ + x¹² + x⁵ + 1, CRC-16/CCITT as in HDLC, X.25 and Bluetooth):

| | CRC-8 | CRC-16 |
|---|---|---|
| Frame | 32 payload + 8 CRC = 40 bits | 32 payload + 16 CRC = 48 bits |
| Coded symbols | 70 (rate 32/70) | 84 (rate 32/84) |
| Guard sum per frame | 256 terms | 65,536 terms (numba kernel, about 7,000 frames/s on 4 cores) |
| Files | `base_awgn.npz`, `thresholds_awgn.txt`, `benchmark.csv` | `base_awgn_crc16.npz`, `thresholds_awgn_crc16.txt`, `benchmark_crc16.csv` |

```
python scripts/train_all.py --crc CRC-16                     # neural codes for 48-bit frames
python -m guardcode.evaluate --crc CRC-16 --profiles awgn,burst,crawdad
python scripts/populate_db.py --crc CRC-8,CRC-16             # database with both
```

The dashboard has a **CRC-8 / CRC-16** switch. The database records the CRC of every frame and benchmark row, and the explorer has a "CRC-8 vs CRC-16" preset.

### CRC-16 results

These are undetected frames pooled over 0–7 dB, with 300,000 frames per point (2.4 million per row), from `results/benchmark_crc16.csv`:

| Receiver | AWGN | Burst | Real 802.15.4 (CRAWDAD) |
|---|---|---|---|
| Uncoded + CRC-16 | 49 | 65 | 51 |
| Hamming + CRC-16 → + SSG | 6 → 0 | 11 → 2 | 6 → 3 |
| Neural + CRC-16 | 5 | 17 | 13 |
| Neural + CRC-16 + LLR test | 5 | 16 | 13 |
| **GuardCode (CRC-16)** | **0** | **8** | **3** |

- **What CRC-16 does:** it removes about 99% of the undetected errors CRC-8 lets through. Neural+CRC drops from 1,559 to 5 on AWGN and from 2,154 to 13 on the real channel.
- **What the guard adds:** it still lowers the remainder. The counts are single-digit, though, so they show the direction rather than a precise percentage; measuring that would take hundreds of millions of frames.
- **The cost:** the 8 extra CRC bits lower goodput from 3 dB upward, from 0.40 to 0.29 on AWGN for the neural code. CRC-8 with GuardCode-ML keeps the higher goodput, at 324 undetected frames instead of about 5. Which is better depends on how much one undetected error costs the application.

**No ML gate for CRC-16.** Only about 2 in 600,000 CRC-16 frames passed the check while still being wrong, so even 6 million training frames would give about 20 positive examples, far too few to train a classifier honestly. CRC-16 runs with the exact Soft-Syndrome Guard only.
