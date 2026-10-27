"""
Live link simulator for the dashboard.

A background thread transmits frames continuously through the selected channel with
every scheme side by side, writes EVERY frame to the SQLite database, and keeps
rolling statistics that the web page receives over Server-Sent Events.
"""
import collections
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardcode.adaptive import AdaptiveController                       # noqa: E402
from guardcode.channels import ChannelConfig, transmit_np                # noqa: E402
from guardcode.crawdad import FIT_PATH                                   # noqa: E402
from guardcode.db import Database                                        # noqa: E402
from guardcode.linksim import DETECTED, FALSE_REJ, OK, UNDETECTED        # noqa: E402
from guardcode.mlguard import MLGuard, model_path                         # noqa: E402
from guardcode.schemes import (CRC_CHOICES, HammingScheme, NeuralScheme,  # noqa: E402
                               Uncoded, crc, neural_path, thresholds)

HISTORY = 240          # chart points kept in memory
WINDOW = 20            # ticks in the rolling window


class LiveEngine(threading.Thread):
    def __init__(self, db: Database, frames_per_tick=100, tick=0.5, max_rows=10_000_000):
        super().__init__(daemon=True)
        self.db = db
        self.frames_per_tick = frames_per_tick
        self.tick = tick
        self.max_rows = max_rows
        self.ebno = 4.0
        self.channel = "awgn"
        self.paused = False
        self.storm_until = 0.0
        self.rng = np.random.default_rng()
        self.lock = threading.RLock()
        self.build("CRC-8")
        self.reset_stats()
        self.con = db.connect()
        self.run_id = db.new_run("live", "dashboard session", self.con)
        self.seq = 0
        self.rows_total = self.con.execute("SELECT COALESCE(MAX(id),0) FROM frames").fetchone()[0]
        self.insert_rate = 0.0
        self.last_bidx = 0.0

    def build(self, crc_name):
        """(Re)build the line-up for a CRC. CRC-16 rows need the *_crc16 models (train_all.py --crc CRC-16)."""
        self.crc_name = crc_name
        self.crc = crc(crc_name)
        ta, tb = thresholds("awgn", crc_name), thresholds("burst", crc_name)
        ham = HammingScheme(interleave=True, name="Hamming+CRC", crc_name=crc_name)
        self.schemes = [
            Uncoded(crc_name),
            ham,
            ham.with_gate("ssg", ta["ssg_hamming"], "Hamming+CRC+SSG"),
        ]
        self.adaptive = self.adaptive_ml = None
        pa, pb = neural_path("awgn", crc_name), neural_path("burst", crc_name)
        if os.path.exists(pa):
            nn_awgn = NeuralScheme(pa, "Neural+CRC")
            self.schemes += [nn_awgn.with_gate("llr", ta["llr"], "Neural+CRC+LLR"),
                             nn_awgn.with_gate("ssg", ta["ssg"], "GuardCode")]
            ml_a, ml_b = model_path("awgn", crc_name), model_path("burst", crc_name)
            if os.path.exists(ml_a) and "ml" in ta:
                self.schemes.append(nn_awgn.with_gate("ml", ta["ml"], "GuardCode-ML", ml=MLGuard.load(ml_a)))
            if os.path.exists(pb):
                nn_burst = NeuralScheme(pb, "Neural+CRC")
                self.adaptive = AdaptiveController({
                    "awgn": nn_awgn.with_gate("ssg", ta["ssg"], "GuardCode-Adaptive"),
                    "burst": nn_burst.with_gate("ssg", tb["ssg"], "GuardCode-Adaptive"),
                })
                if os.path.exists(ml_a) and os.path.exists(ml_b) and "ml" in ta and "ml" in tb:
                    self.adaptive_ml = AdaptiveController({
                        m: nn.with_gate("ml", t["ml"], "GuardCode-ML-Adaptive", ml=MLGuard.load(p))
                        for m, nn, t, p in (("awgn", nn_awgn, ta, ml_a), ("burst", nn_burst, tb, ml_b))})
        self.names = [s.name for s in self.schemes] + (["GuardCode-Adaptive"] if self.adaptive else []) + (
            ["GuardCode-ML-Adaptive"] if self.adaptive_ml else [])

    # ------------------------------------------------------------------ control
    def reset_stats(self):
        with self.lock:
            z = lambda: dict(frames=0, ok=0, detected=0, undetected=0, false_rej=0, bit_err=0, est_ue=0.0)  # noqa: E731
            self.cum = {n: z() for n in self.names}
            self.win = {n: collections.deque(maxlen=WINDOW) for n in self.names}
            self.series = collections.deque(maxlen=HISTORY)
            self.events = collections.deque(maxlen=30)
            self.t_start = time.time()

    def control(self, **kw):
        with self.lock:
            if "ebno" in kw:
                self.ebno = float(np.clip(float(kw["ebno"]), -2, 10))
            if kw.get("channel") in ("awgn", "burst", "fading") or (
                    kw.get("channel") == "crawdad" and os.path.exists(FIT_PATH)):
                self.channel = kw["channel"]
            if "paused" in kw:
                self.paused = bool(kw["paused"])
            if kw.get("storm"):
                self.storm_until = time.time() + 20
            if "rate" in kw:
                self.frames_per_tick = int(np.clip(int(kw["rate"]), 10, 2000))
            new_crc = kw.get("crc") in CRC_CHOICES and kw["crc"] != self.crc_name
            if new_crc:
                self.build(kw["crc"])
        if kw.get("reset") or new_crc:
            self.reset_stats()

    def current_channel(self):
        return "burst" if time.time() < self.storm_until else self.channel

    # ------------------------------------------------------------------ main loop
    def step(self):
        B = self.frames_per_tick
        chan = self.current_channel()
        cfg = ChannelConfig(chan)
        now = time.time()
        msgs = self.rng.integers(0, 2, (B, 32)).astype(np.uint8)
        fb = self.crc.encode(msgs)
        rows = []
        active = self.adaptive.active if self.adaptive else "-"
        ctls = [c for c in (self.adaptive, self.adaptive_ml) if c]
        for s in self.schemes + ctls:
            adaptive = isinstance(s, AdaptiveController)
            ctl = s
            if adaptive:
                s = ctl.scheme          # profile chosen from previous frames (ACK feedback)
            x = s.encode(fb.astype(np.float64))
            y, h, _ = transmit_np(x, self.ebno, s.rate, cfg, self.rng)
            bits, accept, crc_ok, sc = s.decide(y, h, self.ebno)
            if adaptive:
                bidx, act = ctl.observe(y, bits, h if chan == "fading" else None)
                if ctl is self.adaptive:
                    self.last_bidx, active = bidx, act
            name = s.name
            wrong = (bits != fb).any(axis=1)
            out = np.where(wrong, np.where(accept, UNDETECTED, DETECTED), np.where(accept, OK, FALSE_REJ))
            be = (bits[:, :32] != msgs).sum(axis=1)
            score = np.nan_to_num(sc["risk"] if s.gate != "llr" else sc["llr"], nan=-1.0)   # -1: not computed
            st = dict(frames=B, ok=int((out == OK).sum()), detected=int((out == DETECTED).sum()),
                      undetected=int((out == UNDETECTED).sum()), false_rej=int((out == FALSE_REJ).sum()),
                      bit_err=int(be.sum()), est_ue=float(np.nansum(sc["risk"][accept])))
            c = self.cum[name]
            for k in c:
                c[k] += st[k]
            self.win[name].append(st)
            if st["undetected"]:
                self.events.appendleft(dict(ts=now, scheme=name, channel=chan, ebno=round(self.ebno, 2),
                                            count=st["undetected"], bit_errors=int(be[out == UNDETECTED].sum())))
            seq0 = self.seq
            rows.extend((self.run_id, now, seq0 + i, name, chan, self.ebno, int(be[i]), int(out[i]),
                         float(score[i]), 1) for i in range(B))
        self.seq += B
        t0 = time.time()
        self.db.insert_frames(rows, self.con, crc=self.crc_name)
        self.con.execute("INSERT INTO adaptive_log(ts, run_id, burst_index, chosen_profile, true_channel)"
                         " VALUES (?,?,?,?,?)", (now, self.run_id, self.last_bidx, active, chan))
        self.con.commit()
        dt = time.time() - t0
        self.rows_total += len(rows)
        self.insert_rate = len(rows) / self.tick
        self.series.append(dict(
            t=round(now - self.t_start, 1), channel=chan, ebno=self.ebno,
            undetected={n: self.cum[n]["undetected"] for n in self.names},
            fer={n: self._rolling(n, "fer") for n in self.names},
            goodput={n: self._rolling(n, "goodput") for n in self.names},
            burst_index=round(self.last_bidx, 3), profile=active, db_ms=round(dt * 1000, 1)))
        if self.rows_total > self.max_rows and (self.seq // B) % 200 == 0:
            self.prune()

    def _n_sym(self, name):
        return next((s.n for s in self.schemes if s.name == name), 70)

    def _rolling(self, name, metric):
        w = list(self.win[name])
        f = sum(x["frames"] for x in w) or 1
        if metric == "fer":
            return (sum(x["detected"] + x["undetected"] for x in w)) / f
        return 32 * sum(x["ok"] for x in w) / (f * self._n_sym(name))

    def prune(self):
        """Retention policy: keep the newest max_rows frames."""
        cut = self.rows_total - self.max_rows
        self.con.execute("DELETE FROM frames WHERE id <= ?", (cut,))
        self.con.commit()

    def run(self):
        while True:
            t = time.time()
            if not self.paused:
                try:
                    with self.lock:
                        self.step()
                except Exception as e:  # keep the live loop alive
                    print("engine error:", repr(e), flush=True)
            time.sleep(max(0.0, self.tick - (time.time() - t)))

    # ------------------------------------------------------------------ snapshot
    def snapshot(self):
        with self.lock:
            kpis = {}
            for n in self.names:
                c = self.cum[n]
                f = c["frames"] or 1
                fe = c["detected"] + c["undetected"]
                kpis[n] = dict(frames=c["frames"], fer=fe / f, ber=c["bit_err"] / (32 * f),
                               undetected=c["undetected"], detected=c["detected"], false_rej=c["false_rej"],
                               undetected_share=c["undetected"] / max(1, fe), est_ue=round(c["est_ue"], 2),
                               goodput=self._rolling(n, "goodput"))
            return dict(running=not self.paused, ebno=self.ebno, channel=self.channel,
                        effective_channel=self.current_channel(),
                        storm_left=max(0, round(self.storm_until - time.time())),
                        rate=int(self.frames_per_tick / self.tick), names=self.names, kpis=kpis,
                        point=self.series[-1] if self.series else None, events=list(self.events)[:12],
                        crc=self.crc_name,
                        adaptive=dict(profile=self.adaptive.active if self.adaptive else "-",
                                      switches=self.adaptive.switches if self.adaptive else 0,
                                      ewma=None if not self.adaptive or self.adaptive.ewma is None
                                      else round(self.adaptive.ewma, 3)),
                        db=dict(rows=self.rows_total, insert_rate=self.insert_rate, run_id=self.run_id,
                                size_mb=round(os.path.getsize(self.db.path) / 1e6, 1)))

    def history(self):
        with self.lock:
            return list(self.series)
