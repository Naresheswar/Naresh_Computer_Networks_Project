"""
GuardCode live dashboard.

    python dashboard/app.py            -> http://127.0.0.1:5000

Pages
  /            live link monitor (Server-Sent Events, ~2 updates/s)
  /explorer    read-only SQL explorer over the frames database
API
  GET  /stream             SSE snapshots
  GET  /api/history        chart history
  POST /api/control        {ebno, channel, crc, paused, storm, rate, reset}
  GET  /api/benchmark      offline benchmark table (from python -m guardcode.evaluate)
  GET  /api/dbstats        table sizes
  POST /api/query          {"sql": "SELECT ..."}  read-only, 500-row limit
"""
import json
import os
import sqlite3
import sys
import time

from flask import Flask, Response, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from guardcode.db import DB_PATH, Database   # noqa: E402
from engine import LiveEngine               # noqa: E402

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"), static_folder=os.path.join(HERE, "static"))
db = Database()
engine = LiveEngine(db)

PRESETS = {
    "Undetected errors by scheme": """SELECT scheme, COUNT(*) AS frames, SUM(outcome=2) AS undetected,
  ROUND(1.0*SUM(outcome=2)/COUNT(*), 7) AS uer, ROUND(1.0*SUM(outcome IN (1,2))/COUNT(*), 5) AS fer
FROM frames GROUP BY scheme ORDER BY uer DESC""",
    "Share of errors that slipped past the CRC": """SELECT scheme, channel, SUM(outcome IN (1,2)) AS frame_errors, SUM(outcome=2) AS undetected,
  ROUND(100.0*SUM(outcome=2)/MAX(1,SUM(outcome IN (1,2))), 4) AS pct_undetected
FROM frames GROUP BY scheme, channel ORDER BY channel, pct_undetected DESC""",
    "CRC-8 vs CRC-16 (undetected errors)": """SELECT crc, scheme, COUNT(*) AS frames, SUM(outcome IN (1,2)) AS frame_errors, SUM(outcome=2) AS undetected,
  ROUND(100.0*SUM(outcome=2)/MAX(1,SUM(outcome IN (1,2))), 5) AS pct_undetected
FROM frames GROUP BY crc, scheme ORDER BY crc, pct_undetected DESC""",
    "Undetected error rate vs Eb/N0": """SELECT scheme, channel, ROUND(ebno_db) AS ebno, COUNT(*) AS frames, SUM(outcome=2) AS undetected
FROM frames GROUP BY scheme, channel, ROUND(ebno_db) ORDER BY channel, scheme, ebno""",
    "Last 25 undetected frames": """SELECT id, datetime(ts,'unixepoch','localtime') AS time, scheme, channel, ROUND(ebno_db,2) AS ebno,
  bit_errors, ROUND(reliability,3) AS reliability FROM frames WHERE outcome=2 ORDER BY id DESC LIMIT 25""",
    "SSG residual risk R(y) by outcome": """SELECT scheme, CASE outcome WHEN 0 THEN 'OK' WHEN 1 THEN 'DETECTED' WHEN 2 THEN 'UNDETECTED' ELSE 'FALSE_REJECT' END AS outcome,
  COUNT(*) AS n, ROUND(AVG(reliability),5) AS avg_risk, ROUND(MAX(reliability),5) AS max_risk
FROM frames WHERE scheme LIKE 'GuardCode%' GROUP BY scheme, outcome""",
    "Adaptive controller decisions": """SELECT true_channel, chosen_profile, COUNT(*) AS ticks, ROUND(AVG(burst_index),3) AS avg_burst_index
FROM adaptive_log GROUP BY true_channel, chosen_profile""",
    "Runs": "SELECT id, datetime(started_at,'unixepoch','localtime') AS started, mode, notes FROM runs ORDER BY id DESC LIMIT 50",
    "Offline benchmark (GuardCode vs baselines)": """SELECT profile, crc, scheme, ebno_db, frames, fer, uer, undetected, ROUND(goodput,4) AS goodput
FROM benchmark WHERE id IN (SELECT MAX(id) FROM benchmark GROUP BY profile, crc, scheme, ebno_db) ORDER BY profile, crc, ebno_db, scheme""",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/explorer")
def explorer():
    return render_template("explorer.html", presets=PRESETS)


@app.route("/stream")
def stream():
    def gen():
        while True:
            yield f"data: {json.dumps(engine.snapshot())}\n\n"
            time.sleep(engine.tick)
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/api/snapshot")
def snapshot():
    return jsonify(engine.snapshot())


@app.route("/api/history")
def history():
    return jsonify(engine.history())


@app.route("/api/control", methods=["POST"])
def control():
    engine.control(**(request.get_json(silent=True) or {}))
    return jsonify(engine.snapshot())


@app.route("/api/benchmark")
def benchmark():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM benchmark WHERE id IN (SELECT MAX(id) FROM benchmark GROUP BY profile, crc, scheme, ebno_db)"
                       " ORDER BY profile, scheme, ebno_db").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/dbstats")
def dbstats():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    out = {}
    for t in ("runs", "frames", "benchmark", "adaptive_log"):
        out[t] = con.execute(f"SELECT COALESCE(MAX(rowid),0) FROM {t}").fetchone()[0]
    con.close()
    out["size_mb"] = round(os.path.getsize(DB_PATH) / 1e6, 1)
    return jsonify(out)


@app.route("/api/query", methods=["POST"])
def query():
    sql = (request.get_json(silent=True) or {}).get("sql", "").strip().rstrip(";")
    low = sql.lower()
    if not (low.startswith("select") or low.startswith("with")) or ";" in sql:
        return jsonify(error="Only a single SELECT statement is allowed."), 400
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
        t0 = time.time()
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(500)
        con.close()
        return jsonify(columns=cols, rows=rows, ms=round((time.time() - t0) * 1000, 1))
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    engine.start()
    port = int(os.environ.get("PORT", 5000))
    print(f"GuardCode dashboard running on http://127.0.0.1:{port}  (database: {DB_PATH})")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
