"""
SQLite storage for GuardCode (file: data/guardcode.db, WAL mode so the live simulator
can write while the dashboard reads).

Tables
  runs         one row per simulation run (live session, bulk population, benchmark)
  frames       one row per transmitted frame  <- the large table (millions of rows)
  benchmark    aggregated Monte-Carlo sweep results
  adaptive_log decisions of the channel-adaptive profile controller
"""
import os
import sqlite3
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "guardcode.db")

OUTCOMES = {0: "OK", 1: "DETECTED", 2: "UNDETECTED", 3: "FALSE_REJECT"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    mode        TEXT NOT NULL,           -- live | bulk | benchmark
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    ts          REAL NOT NULL,           -- unix time
    seq         INTEGER NOT NULL,        -- frame sequence number within run+scheme
    scheme      TEXT NOT NULL,
    channel     TEXT NOT NULL,           -- awgn | burst | fading
    ebno_db     REAL NOT NULL,
    bit_errors  INTEGER NOT NULL,        -- payload bit errors after decoding
    outcome     INTEGER NOT NULL,        -- 0 OK, 1 DETECTED, 2 UNDETECTED, 3 FALSE_REJECT
    reliability REAL NOT NULL,           -- SSG residual risk R(y) (min |LLR| for the LLR-gated scheme)
    attempt     INTEGER NOT NULL DEFAULT 1,  -- ARQ transmission attempt number
    crc         TEXT NOT NULL DEFAULT 'CRC-8'  -- CRC-8 | CRC-16
);
CREATE INDEX IF NOT EXISTS ix_frames_run    ON frames(run_id);
CREATE INDEX IF NOT EXISTS ix_frames_ts     ON frames(ts);
CREATE INDEX IF NOT EXISTS ix_frames_lookup ON frames(scheme, channel, ebno_db, outcome);
CREATE TABLE IF NOT EXISTS benchmark (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL, profile TEXT, scheme TEXT,
    ebno_db REAL, n_sym INTEGER, frames INTEGER, bit_errors INTEGER, frame_errors INTEGER,
    detected INTEGER, undetected INTEGER, false_rej INTEGER, ber REAL, fer REAL, uer REAL,
    goodput REAL, crc TEXT NOT NULL DEFAULT 'CRC-8'
);
CREATE TABLE IF NOT EXISTS adaptive_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id INTEGER, burst_index REAL,
    chosen_profile TEXT, true_channel TEXT
);
CREATE VIEW IF NOT EXISTS v_summary AS
SELECT scheme, channel, ROUND(ebno_db, 1) AS ebno_db, COUNT(*) AS frames,
       SUM(outcome = 0) AS ok, SUM(outcome = 1) AS detected, SUM(outcome = 2) AS undetected,
       SUM(outcome = 3) AS false_reject, SUM(bit_errors) AS bit_errors
FROM frames GROUP BY scheme, channel, ROUND(ebno_db, 1);
"""


class Database:
    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        con = self.connect()
        con.executescript(SCHEMA)
        for table in ("frames", "benchmark"):      # databases created before CRC-16 support
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if "crc" not in cols:
                con.execute(f"ALTER TABLE {table} ADD COLUMN crc TEXT NOT NULL DEFAULT 'CRC-8'")
        con.commit()
        con.close()

    def connect(self):
        con = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def new_run(self, mode, notes="", con=None):
        own = con is None
        con = con or self.connect()
        cur = con.execute("INSERT INTO runs(started_at, mode, notes) VALUES (?,?,?)", (time.time(), mode, notes))
        con.commit()
        rid = cur.lastrowid
        if own:
            con.close()
        return rid

    def insert_frames(self, rows, con=None, crc="CRC-8"):
        """rows: iterable of (run_id, ts, seq, scheme, channel, ebno_db, bit_errors, outcome, reliability, attempt)"""
        own = con is None
        con = con or self.connect()
        with self.lock:
            con.executemany("INSERT INTO frames(run_id, ts, seq, scheme, channel, ebno_db, bit_errors, outcome,"
                            " reliability, attempt, crc) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r + (crc,) for r in rows))
            con.commit()
        if own:
            con.close()

    def insert_benchmark(self, rows):
        con = self.connect()
        now = time.time()
        con.executemany(
            "INSERT INTO benchmark(created_at, profile, scheme, ebno_db, n_sym, frames, bit_errors, frame_errors,"
            " detected, undetected, false_rej, ber, fer, uer, goodput, crc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(now, r["profile"], r["scheme"], r["ebno_db"], r["n_sym"], r["frames"], r["bit_errors"],
              r["frame_errors"], r["detected"], r["undetected"], r["false_rej"], r["ber"], r["fer"],
              r["uer"], r["goodput"], r.get("crc", "CRC-8")) for r in rows])
        con.commit()
        con.close()

    def stats(self):
        con = self.connect()
        n = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        con.close()
        size = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0
        return dict(frames=n, runs=runs, size_mb=round(size, 1))
