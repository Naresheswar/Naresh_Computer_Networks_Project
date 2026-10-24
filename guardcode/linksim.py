"""
Monte-Carlo link simulator: one call = push N frames of one scheme through one channel.

Outcome per frame (exactly one of):
    OK          decoded correctly and accepted
    DETECTED    wrong and rejected (CRC or gate) -> ARQ retransmits
    UNDETECTED  wrong but accepted -> corrupted data delivered to the application (the
                dangerous case GuardCode is designed to suppress)
    FALSE_REJ   decoded correctly but rejected by a gate (costs one retransmission)
"""
import numpy as np

from .channels import transmit_np

OK, DETECTED, UNDETECTED, FALSE_REJ = 0, 1, 2, 3


def run_batch(scheme, crc, ebno_db, cfg, B, rng, k_info=32):
    msgs = rng.integers(0, 2, (B, k_info)).astype(np.uint8)
    frames = crc.encode(msgs)
    x = scheme.encode(frames.astype(np.float64))
    y, h, _ = transmit_np(x, ebno_db, scheme.rate, cfg, rng)
    bits, accept, crc_ok, sc = scheme.decide(y, h, ebno_db)
    wrong = (bits != frames).any(axis=1)
    bit_err = (bits[:, :k_info] != msgs).sum(axis=1)
    outcome = np.where(wrong, np.where(accept, UNDETECTED, DETECTED), np.where(accept, OK, FALSE_REJ))
    return dict(wrong=wrong, bit_err=bit_err, crc_ok=crc_ok, accept=accept, outcome=outcome,
                risk=sc["risk"], llr=sc["llr"], p1=sc["p1"], ml=sc.get("ml"), y=y, h=h, bits=bits)


def simulate(scheme, crc, ebno_db, cfg, n_frames, batch=20000, seed=0):
    rng = np.random.default_rng(seed)
    tot = dict(frames=0, bit_errors=0, frame_errors=0, detected=0, undetected=0, false_rej=0,
               crc_undetected=0, est_undetected=0.0)
    left = n_frames
    while left > 0:
        B = min(batch, left)
        r = run_batch(scheme, crc, ebno_db, cfg, B, rng)
        tot["frames"] += B
        tot["bit_errors"] += int(r["bit_err"].sum())
        tot["frame_errors"] += int(r["wrong"].sum())
        tot["detected"] += int((r["outcome"] == DETECTED).sum())
        tot["undetected"] += int((r["outcome"] == UNDETECTED).sum())
        tot["crc_undetected"] += int((r["wrong"] & r["crc_ok"]).sum())
        tot["false_rej"] += int((r["outcome"] == FALSE_REJ).sum())
        # SSG self-estimate of undetected errors among accepted frames (no ground truth needed)
        tot["est_undetected"] += float(r["risk"][r["accept"]].sum())
        left -= B
    f = tot["frames"]
    tot.update(
        ber=tot["bit_errors"] / (f * 32),
        fer=tot["frame_errors"] / f,
        uer=tot["undetected"] / f,
        reject_rate=(tot["detected"] + tot["false_rej"]) / f,
        # selective-repeat ARQ goodput: correctly delivered info bits per channel use
        goodput=(32 * (f - tot["detected"] - tot["false_rej"] - tot["undetected"])) / (f * scheme.n),
    )
    return tot
