"""
Train a neural code end-to-end.

    python -m guardcode.train --name guard_awgn --channel awgn --lam 2.0
    python -m guardcode.train --name base_awgn  --channel awgn --lam 0      (ablation: no blind loss)

The *only* difference between GuardCode and the neural baseline is the lam * L_blind
term, so any change in the undetected-error rate is attributable to the invention.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .channels import ChannelConfig, transmit_torch
from .crc import CRC
from .losses import SoftCRC, guard_loss
from .model import Decoder, Encoder, export_npz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(name, channel="awgn", lam=2.0, k=32, crc_name="CRC-8", n_sym=70, steps=6000,
          batch=1024, lr=1e-3, ebno_lo=1.0, ebno_hi=6.0, seed=7, log_every=500, out_dir=None,
          init=None, warmup=True, objective="bit"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    crc = CRC(k, crc_name)
    n_bits = crc.n
    rate = k / n_sym
    cfg = ChannelConfig(channel)
    interleave = channel == "burst"
    enc, dec = Encoder(n_bits, n_sym, interleave=interleave), Decoder(n_sym, n_bits)
    pow2 = 2 ** torch.arange(7, -1, -1)
    if init:  # fine-tune from a checkpoint (used for the controlled ablation)
        ck = torch.load(init)
        enc.load_state_dict(ck["enc"])
        dec.load_state_dict(ck["dec"])
    soft = SoftCRC(crc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        msgs = np.random.randint(0, 2, (batch, k))
        frames = torch.as_tensor(crc.encode(msgs), dtype=torch.float32)
        ebno = torch.empty(batch).uniform_(ebno_lo, ebno_hi)
        x = enc(frames)
        y, h = transmit_torch(x, ebno, rate, cfg)
        logp, logits = dec(y * h)
        sym_idx = (frames.view(batch, -1, 8).long() * pow2).sum(-1)
        # warm-up: let the code learn to communicate first, then switch on the guard term
        lam_t = lam * (min(1.0, max(0.0, (step - 0.3 * steps) / (0.2 * steps))) if warmup else min(1.0, step / (0.1 * steps)))
        loss, bce, blind, pue = guard_loss(logp, logits, frames, sym_idx, soft, lam_t, objective)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                hard = (logits > 0).float()  # noqa
                fer = (hard != frames).any(dim=1).float().mean().item()
            rec = dict(step=step, loss=float(loss), bce=float(bce), blind=float(blind), fer=fer,
                       lam=lam_t, sec=round(time.time() - t0, 1))
            hist.append(rec)
            print(json.dumps(rec), flush=True)
    meta = dict(name=name, channel=cfg.to_dict(), lam=lam, k=k, crc=crc_name, n_bits=n_bits,
                n_sym=n_sym, rate=rate, steps=steps, interleave=interleave, arch="bytecode-mlp", objective=objective, decision="bit", ebno_train=[ebno_lo, ebno_hi], seed=seed, init=os.path.basename(init) if init else None)
    out_dir = out_dir or os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    export_npz(enc, dec, os.path.join(out_dir, f"{name}.npz"), meta)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "meta": meta},
               os.path.join(out_dir, f"{name}.pt"))
    with open(os.path.join(out_dir, f"{name}_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return enc, dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--channel", default="awgn", choices=["awgn", "burst", "fading"])
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ebno_lo", type=float, default=1.0)
    ap.add_argument("--ebno_hi", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init", default=None, help="checkpoint (.pt) to fine-tune from")
    a = ap.parse_args()
    train(a.name, a.channel, a.lam, steps=a.steps, batch=a.batch, ebno_lo=a.ebno_lo,
          ebno_hi=a.ebno_hi, seed=a.seed, init=a.init, warmup=a.init is None)
