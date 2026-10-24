"""Collect the headline numbers from results/benchmark.csv + results/analysis.json into
results/summary.json (used by the report / patent generators)."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guardcode.schemes import MODELS, ROOT, crc, thresholds  # noqa: E402

RES = os.path.join(ROOT, "results")
rows = list(csv.DictReader(open(os.path.join(RES, "benchmark.csv"))))
for r in rows:
    for k in r:
        if k not in ("profile", "scheme"):
            r[k] = float(r[k])
an = json.load(open(os.path.join(RES, "analysis.json")))


def get(prof, sch, e):
    return next(r for r in rows if r["profile"] == prof and r["scheme"] == sch and r["ebno_db"] == e)


def pooled(prof, sch):
    sub = [r for r in rows if r["profile"] == prof and r["scheme"] == sch]
    return dict(frames=int(sum(r["frames"] for r in sub)), fe=int(sum(r["frame_errors"] for r in sub)),
                ue=int(sum(r["undetected"] for r in sub)), est=sum(r["est_undetected"] for r in sub),
                fr=int(sum(r["false_rej"] for r in sub)))


schemes = ["Uncoded+CRC", "Hamming+CRC", "Hamming+CRC+SSG", "Neural+CRC", "Neural+CRC+LLR", "GuardCode", "GuardCode-ML"]
S = dict(k=32, crc_dmin=crc().min_distance(), bench_frames=int(rows[0]["frames"]),
         ebnos=sorted({r["ebno_db"] for r in rows}))
S["pooled"] = {p: {s: pooled(p, s) for s in schemes} for p in ("awgn", "burst")}
S["thresholds"] = {p: thresholds(p) for p in ("awgn", "burst")}
# point tables
S["points"] = {p: {s: {str(int(r["ebno_db"])): dict(fer=r["fer"], ber=r["ber"], uer=r["uer"], ue=int(r["undetected"]),
                                                     goodput=r["goodput"])
                       for r in rows if r["profile"] == p and r["scheme"] == s} for s in schemes} for p in ("awgn", "burst")}


def red(a, b):
    return 100.0 * (1 - a / b) if b else 0.0


P = S["pooled"]
S["reduction"] = {p: dict(
    guard_vs_llr=red(P[p]["GuardCode"]["ue"], P[p]["Neural+CRC+LLR"]["ue"]),
    guard_vs_plain=red(P[p]["GuardCode"]["ue"], P[p]["Neural+CRC"]["ue"]),
    hamssg_vs_ham=red(P[p]["Hamming+CRC+SSG"]["ue"], P[p]["Hamming+CRC"]["ue"]),
    guard_vs_ham=red(P[p]["GuardCode"]["ue"], P[p]["Hamming+CRC"]["ue"]),
    ml_vs_guard=red(P[p]["GuardCode-ML"]["ue"], P[p]["GuardCode"]["ue"]),
    ml_vs_llr=red(P[p]["GuardCode-ML"]["ue"], P[p]["Neural+CRC+LLR"]["ue"])) for p in ("awgn", "burst")}
S["fer_gain"] = {e: dict(neural=get("awgn", "Neural+CRC", e)["fer"], hamming=get("awgn", "Hamming+CRC", e)["fer"]) for e in (3.0, 4.0, 5.0, 6.0)}
S["analysis"] = an
S["patent_table"] = [["Scheme (gate)", "AWGN: undetected", "AWGN: % of errors", "Burst: undetected", "Burst: % of errors"]] + [
    [s, f"{P['awgn'][s]['ue']:,}", f"{100 * P['awgn'][s]['ue'] / max(1, P['awgn'][s]['fe']):.3f}",
     f"{P['burst'][s]['ue']:,}", f"{100 * P['burst'][s]['ue'] / max(1, P['burst'][s]['fe']):.3f}"] for s in schemes]
R = S["reduction"]
S["patent_text"] = (
    f"At the same retransmission budget, the soft-syndrome guard reduced the number of undetected frame errors of the neural decoder "
    f"by {R['awgn']['guard_vs_llr']:.0f} % on the AWGN channel and {R['burst']['guard_vs_llr']:.0f} % on the burst channel relative to the conventional "
    f"minimum-|LLR| threshold test, and by {R['awgn']['guard_vs_plain']:.0f} % / {R['burst']['guard_vs_plain']:.0f} % relative to plain CRC acceptance. "
    f"Applied unchanged to a classical Hamming(7,4) soft decoder it reduced undetected errors by {R['awgn']['hamssg_vs_ham']:.0f} % (AWGN) and "
    f"{R['burst']['hamssg_vs_ham']:.0f} % (burst), demonstrating that the guard is independent of the decoder. Because both gates were held to the same "
    f"retransmission budget, the ARQ goodput of the guarded link stayed within about 1 % of the unguarded link from 3 dB upward, the "
    f"usable operating range, and within a few percent at 0-2 dB, where goodput is already very low. "
    f"Replacing the analytic risk by a learned gradient-boosted risk classifier built on the same inputs lowered the undetected count by a further "
    f"{R['awgn']['ml_vs_guard']:.0f} % (AWGN) and {R['burst']['ml_vs_guard']:.0f} % (burst) at the same retransmission budget.")
json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps(S["reduction"], indent=1))
print(json.dumps({p: {s: (v["ue"], v["fe"], round(v["est"], 1)) for s, v in P[p].items()} for p in P}, indent=1))
