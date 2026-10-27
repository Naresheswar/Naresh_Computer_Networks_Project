// Project report (BCSE308L / BCSE308P), structured to the review rubric:
// research gap, objectives, workflow, code, output implementation with comparison tables.
// Run after scripts/summarize.py (and scripts/draw_workflow.py for Figure 1).
const path = require("path");
const fs = require("fs");
const { build } = require("./docgen");
const ROOT = path.resolve(__dirname, "..", "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "summary.json"), "utf8"));
const R = (f) => path.join(ROOT, "results", f);
const A = S.analysis;
const pct = (x) => `${x.toFixed(0)} %`;
const e = (x) => (x === 0 ? "0" : x.toExponential(2));
const n = (x) => x.toLocaleString("en-IN");
const P = S.pooled, RD = S.reduction, P16 = S.pooled_crc16, CW = S.crawdad;
const cal = A.calibration;
const ad = A.adaptive;

// source excerpt from startMarker up to (not including) endMarker
function snippet(file, start, end) {
  const src = fs.readFileSync(path.join(ROOT, file), "utf8");
  const i = src.indexOf(start);
  const j = end ? src.indexOf(end, i + start.length) : src.length;
  if (i < 0 || j < 0) throw new Error(`snippet markers not found in ${file}`);
  return src.slice(i, j).replace(/\s+$/, "");
}

const sch = ["Uncoded+CRC", "Hamming+CRC", "Hamming+CRC+SSG", "Neural+CRC", "Neural+CRC+LLR", "GuardCode", "GuardCode-ML"];
function pointTable(prof, metric, fmt) {
  const eb = ["0", "1", "2", "3", "4", "5", "6", "7"];
  return [["Scheme", ...eb.map((x) => `${x} dB`)], ...sch.map((s) => [s, ...eb.map((x) => fmt(S.points[prof][s][x][metric]))])];
}
const ueTable = (prof) => [["Scheme", "Frames", "Frame errors", "Undetected", "% of errors undetected", "Ledger estimate (accepted)"],
  ...sch.map((s) => { const v = P[prof][s]; return [s, n(v.frames), n(v.fe), n(v.ue), (100 * v.ue / Math.max(1, v.fe)).toFixed(3), s.includes("SSG") || s === "GuardCode" ? v.est.toFixed(0) : "-"]; })];
const tr = (prof, lab) => A[`tradeoff_${prof}`][lab];
const trRow = (prof, lab) => {
  const t = tr(prof, lab); const idx = [3, 6];
  return [`${lab} (${prof.toUpperCase()})`, e(t.uer_llr[0]), ...idx.flatMap((i) => [e(t.uer_llr[i]), e(t.uer_ssg[i])])];
};
const CH = ["awgn", "burst", "crawdad"];
const allChannels = [["Receiver", "AWGN", "Burst", "Measured 802.15.4", "Goodput from 3 dB (AWGN)"],
  ...sch.map((s) => [s, ...CH.map((p) => n(P[p][s].ue)), P.awgn[s].goodput_3db.toFixed(3)])];
const methodTable = [["Method", "Extra bits", "Changes CRC", "Works with any decoder", "Undetected AWGN", "Undetected burst", "Undetected measured"],
  ["Plain CRC acceptance (standard receiver)", "0", "No", "Yes", n(P.awgn["Neural+CRC"].ue), n(P.burst["Neural+CRC"].ue), n(P.crawdad["Neural+CRC"].ue)],
  ["Min-|LLR| confidence threshold (prior art)", "0", "No", "Yes", n(P.awgn["Neural+CRC+LLR"].ue), n(P.burst["Neural+CRC+LLR"].ue), n(P.crawdad["Neural+CRC+LLR"].ue)],
  ["Longer CRC (CRC-16 instead of CRC-8)", "+8 per frame", "Yes", "Yes", n(P16.awgn["Neural+CRC"].ue), n(P16.burst["Neural+CRC"].ue), n(P16.crawdad["Neural+CRC"].ue)],
  ["Soft-Syndrome Guard (GuardCode, this work)", "0", "No", "Yes", n(P.awgn.GuardCode.ue), n(P.burst.GuardCode.ue), n(P.crawdad.GuardCode.ue)],
  ["Learned gate (GuardCode-ML, this work)", "0", "No", "Needs training", n(P.awgn["GuardCode-ML"].ue), n(P.burst["GuardCode-ML"].ue), n(P.crawdad["GuardCode-ML"].ue)]];
const reductionTable = [["Comparison", "AWGN", "Burst", "Measured 802.15.4"],
  ["GuardCode vs plain CRC acceptance", ...CH.map((p) => pct(RD[p].guard_vs_plain))],
  ["GuardCode vs min-|LLR| test", ...CH.map((p) => pct(RD[p].guard_vs_llr))],
  ["GuardCode-ML vs GuardCode", ...CH.map((p) => pct(RD[p].ml_vs_guard))],
  ["GuardCode-ML vs min-|LLR| test", ...CH.map((p) => pct(RD[p].ml_vs_llr))],
  ["Hamming + SSG vs Hamming (decoder-agnostic)", ...CH.map((p) => pct(RD[p].hamssg_vs_ham))]];
const crc16Table = [["Receiver", "CRC-8 AWGN", "CRC-16 AWGN", "CRC-8 burst", "CRC-16 burst", "CRC-8 measured", "CRC-16 measured"],
  ...["Uncoded+CRC", "Hamming+CRC", "Hamming+CRC+SSG", "Neural+CRC", "Neural+CRC+LLR", "GuardCode"].map((s) =>
    [s, ...CH.flatMap((p) => [n(P[p][s].ue), n(P16[p][s].ue)])])];

const blocks = [
  // ---------------- cover
  { space: 500 },
  { center: "GuardCode: Suppressing Undetected Frame Errors", bold: true, size: 36, after: 60 },
  { center: "with a Soft-Syndrome Guard and a Learned Gate on CRC-Protected Links", bold: true, size: 32, after: 500 },
  { center: "Project Report", bold: true, size: 26, after: 60 },
  { center: "BCSE308L - Computer Networks", bold: true, size: 26, after: 60 },
  { center: "BCSE308P - Computer Networks Lab", bold: true, size: 24, after: 500 },
  { center: "Submitted by", italics: true, size: 24, after: 60 },
  { center: "S Naresh Nireenjan (24BCE2353)", bold: true, size: 26, after: 40 },
  { center: "Dharshan J (24BCE2335)", bold: true, size: 26, after: 40 },
  { center: "Sachin Saravanan S (24BCE2321)", bold: true, size: 26, after: 500 },
  { center: "Under the guidance of", italics: true, size: 24, after: 60 },
  { center: "Dr. Anusha N", bold: true, size: 26, after: 900 },
  { center: "School of Computer Science and Engineering", bold: true, size: 26, after: 60 },
  { center: "Vellore Institute of Technology, Vellore", bold: true, size: 26, after: 60 },
  { center: "October, 2026", bold: true, size: 26 },
  { pagebreak: true },

  // ---------------- contents
  { h1: "Contents" },
  { table: [["Chapter", "Covers"],
    ["Abstract, 1. Introduction", "Context and problem statement"],
    ["2. Research Gap", "Related work and the gaps this project fills"],
    ["3. Objectives", "Objectives, each mapped to a gap and to where it is achieved"],
    ["4. Workflow and Methodology", "Workflow diagram, system architecture, methods"],
    ["5. Code", "Software modules, key code listings, database design"],
    ["6. Output Implementation", "The running system (dashboard screenshots) and generated outputs"],
    ["7. Comparison Tables", "Receivers, channels, CRC-8 vs CRC-16, existing methods"],
    ["8. Conclusion and Future Work, References, Appendix", "Summary and how to run"]], widths: [3, 5], left: true },
  { pagebreak: true },

  // ---------------- abstract
  { h1: "Abstract" },
  { p: `Every frame on a data link carries a cyclic redundancy check (CRC). The receiver decodes the frame, recomputes the CRC, and either delivers the frame or asks for a retransmission. The weak spot is the **undetected error**: a frame that is decoded wrongly but still satisfies the CRC, so corrupted data is delivered silently. This project builds **GuardCode**, a receiver that removes much of that gap without adding a bit to the frame and without changing the CRC.` },
  { p: `Its core, the **Soft-Syndrome Guard (SSG)**, uses the decoder's soft output and the exact algebra of the standard CRC to compute, for every frame that passes the CRC, the probability that it is still wrong; risky frames are retransmitted. A gradient-boosted machine-learning gate (**GuardCode-ML**) refines that probability. With ${n(S.bench_frames)} simulated frames per operating point and the same retransmission budget for every receiver, SSG removed ${pct(RD.awgn.guard_vs_llr)} (AWGN) and ${pct(RD.burst.guard_vs_llr)} (burst) of the undetected errors left by the classic confidence test, and GuardCode-ML removed a further ${pct(RD.awgn.ml_vs_guard)} and ${pct(RD.burst.ml_vs_guard)}. On a channel built from ${CW.n_traces} **real measured 802.15.4 error traces** (CRAWDAD niit/bit_errors), with nothing re-tuned, SSG still left ${pct(RD.crawdad.guard_vs_llr)} fewer undetected errors than the classic test. Everything works with CRC-8 and CRC-16. A live web dashboard and a SQLite database that logs every frame show the system running.` },
  { p: "**Keywords:** CRC, undetected error, ARQ, forward error correction, neural channel code, machine learning, Hamming code, burst errors, data link layer." },

  // ---------------- 1
  { h1: "1. Introduction" },
  { h2: "1.1 Background" },
  { p: "Module 3 of the course covers error detection and correction (parity, checksum, CRC, Hamming code) and flow and error control (Stop-and-Wait, Go-Back-N and Selective-Repeat ARQ). In a real link these work together: a forward error correction (FEC) code repairs most channel errors, the CRC decides whether the repaired frame can be trusted, and ARQ retransmits the frames the CRC rejects." },
  { p: "A CRC with r check bits misses about one in 2^r of the error patterns that reach it. For the short CRCs used in IoT, sensor and control traffic (8 to 16 bits) this is not negligible: one undetected error per few thousand frame errors can corrupt a sensor stream or a control command." },
  { h2: "1.2 Problem statement" },
  { p: "Can a receiver lower the undetected-error rate of a CRC-protected link, at a fixed retransmission budget, without extra redundancy and without changing the CRC or the frame format?" },
  { h2: "1.3 Mapping to the syllabus" },
  { table: [["Course topic", "Where it appears in this project"],
    ["Module 3 - CRC (polynomial division)", "guardcode/crc.py: CRC-8 and CRC-16, generator and parity-check matrices"],
    ["Module 3 - Hamming code", "guardcode/hamming.py: Hamming(7,4), soft decoding, block interleaver"],
    ["Module 3 - Burst errors, interleaving", "Gilbert-Elliott burst channel; real 802.15.4 burst traces; interleaved profiles"],
    ["Module 3 - ARQ (Selective Repeat)", "Goodput model; NAK on CRC or guard rejection"],
    ["Module 1 - Transmission impairments", "AWGN, burst noise, Rayleigh fading and measured channels"],
    ["Module 6 - QoS, reliability", "Undetected error rate as a QoS metric; risk ledger"],
    ["Lab - socket / web programming", "Flask web server, Server-Sent Events, REST API, SQL explorer"]], widths: [3, 5], left: true, caption: "Table 1 - Syllabus mapping" },

  // ---------------- 2 research gap
  { h1: "2. Research Gap" },
  { h2: "2.1 Related work" },
  { table: [["Work", "Idea", "Limitation"],
    ["CRC polynomial design for a given inner code (Lou, Wesel et al., 2015)", "Choose g(x) to minimise undetected errors for a convolutional code", "Requires changing the standard polynomial"],
    ["CRC-aided list decoding (polar CA-SCL; US 2023/0198672 A1)", "CRC selects among decoder candidates", "Decoder-specific; no estimate of the accepted frame's residual risk"],
    ["Neural channel codes (Turbo Autoencoder; US 12,155,482 B2)", "Codes learned end-to-end", "Use parity or LLR checks only as runtime flags"],
    ["Erasure / reliability tests (Forney 1968; ROVA 1998; polar UER thresholds, 2023)", "Reject when decoder confidence is low", "Confidence is blind to which errors the CRC would catch"],
    ["Gallager's parity-check probability (LDPC, 1963)", "Probability that a parity check is satisfied", "Used inside iterative decoders, not as a CRC acceptance test"],
    ["Wireless error measurements (CRAWDAD niit/bit_errors, 2008)", "Real residual bit-error traces of 802.15.4 and 802.11", "Characterise errors; not used to test acceptance rules"]],
    widths: [3, 3, 3], left: true, caption: "Table 2 - Related work" },
  { h2: "2.2 Gaps addressed by this project" },
  { table: [["Gap", "Description", "How this project addresses it"],
    ["G1", "Receivers treat the CRC as a pass/fail bit; no method gives the probability that a CRC-passing frame is wrong from the standard CRC alone", "Soft-Syndrome Guard: exact CRC-conditioned posterior via the dual-code expansion"],
    ["G2", "Confidence thresholds waste the retransmission budget on errors the CRC would catch anyway", "Gate only on the risk that the CRC cannot see, at an equal retransmission budget"],
    ["G3", "Lowering undetected errors usually needs more CRC bits or a non-standard polynomial", "No extra bits; unmodified CRC-8 / CRC-16"],
    ["G4", "Neural-decoder probabilities are not perfectly calibrated, so an exact formula alone is not optimal", "GuardCode-ML: a learned gate that uses the exact risk as one input"],
    ["G5", "Acceptance rules are evaluated on idealised simulated channels only", "Validation on a channel fitted to 22 real measured 802.15.4 error traces"],
    ["G6", "Receivers cannot tell how many corrupted frames they delivered", "Risk ledger: running estimate without ground truth"]],
    widths: [1, 5, 5], left: true, caption: "Table 3 - Research gaps" },
  { p: "A search of Google Patents, Justia and arXiv found no document that computes the CRC-conditioned posterior probability of an accepted frame being wrong through the dual-code expansion of a standard CRC and uses it as an ARQ acceptance gate." },

  // ---------------- 3 objectives
  { h1: "3. Objectives" },
  { table: [["#", "Objective", "Gap", "Achieved in"],
    ["O1", "Build a link simulator with CRC-8 / CRC-16 framing, Hamming(7,4) and learned neural FEC, AWGN, burst, fading and measured channels, and ARQ accounting", "-", "Ch. 4, 5"],
    ["O2", "Design a receiver test that computes the exact residual risk of each CRC-passing frame (Soft-Syndrome Guard)", "G1, G3", "Sec. 4.4, 5.2"],
    ["O3", "Compare it fairly with plain CRC acceptance and the min-|LLR| test at equal retransmission cost", "G2", "Ch. 7"],
    ["O4", "Add a learned (ML) acceptance gate and test whether it improves on the analytic guard", "G4", "Sec. 4.6, 7.1"],
    ["O5", "Validate on real measured error traces and support both CRC-8 and CRC-16", "G5", "Sec. 4.9, 7.1, 7.4"],
    ["O6", "Add a risk ledger, channel-adaptive profiles, a live dashboard and a frame database", "G6", "Sec. 4.7-4.8, Ch. 6"]],
    widths: [1, 6, 1, 2], left: true, caption: "Table 4 - Objectives" },

  // ---------------- 4 workflow
  { h1: "4. Workflow and Methodology" },
  { h2: "4.1 Workflow" },
  { img: R("workflow.png"), width: 600, ratio: 0.615, caption: "Figure 1 - End-to-end workflow" },
  { numbers: [
    "Generate a random 32-bit payload and append the CRC (CRC-8: 40-bit frame; CRC-16: 48-bit frame).",
    "Encode with the FEC code: the learned neural code (70 or 84 symbols) or Hamming(7,4).",
    "Pass through the channel: AWGN, Gilbert-Elliott burst, Rayleigh fading, or the channel fitted to real CRAWDAD traces.",
    "Decode with a soft-output decoder, which gives a probability distribution for every byte or block.",
    "Check the CRC on the hard decision. If it fails, request a retransmission.",
    "If it passes, compute the Soft-Syndrome Guard risk R(y) and, optionally, the GuardCode-ML score.",
    "Deliver the frame only if the score is below a threshold set for a 1 % false-reject budget; otherwise send a NAK.",
    "Record the outcome (OK, detected, undetected, false reject) in the database, aggregate benchmark results, and build the comparison tables and the live dashboard.",
  ], ref: "num" },
  { h2: "4.2 System architecture" },
  { img: R("architecture.png"), width: 600, ratio: 0.43, caption: "Figure 2 - System architecture" },
  { h2: "4.3 Frame format and CRC as a linear code" },
  { p: `A frame carries 32 payload bits and an r-bit CRC: CRC-8 with g(x) = x^8 + x^2 + x + 1, or CRC-16 with g(x) = x^16 + x^12 + x^5 + 1 (CRC-16/CCITT, as in HDLC, X.25 and Bluetooth). The CRC is linear, so crc(m) = m G (mod 2), where row i of G is the CRC of the i-th unit vector, which is exactly the long division done in the lab. Every valid frame b satisfies H b = 0 with H = [G^T | I_r]. Exhaustive search gives a minimum distance of ${S.crc_dmin} for the 40-bit CRC-8 frame and ${S.crc16_dmin} for the 48-bit CRC-16 frame.` },
  { h2: "4.4 Soft-Syndrome Guard" },
  { p: "The decoder outputs a posterior over each **group** of frame bits: 256 values per byte for the neural decoder, 16 messages per 4-bit Hamming block, or one bit for uncoded BPSK. The hard decision x takes the most likely value of every group. Since the transmitted frame must be a CRC codeword, Bayes' rule gives:" },
  { eq: "P(b = x | y, Hb = 0) = ∏_c P_c(x_c) / P(Hb = 0 | y)" },
  { p: "The denominator is computed exactly by summing over the 2^r words u of the dual code, where each group contributes one Walsh-Hadamard transform (WHT) coefficient of its posterior:" },
  { eq: "P(Hb = 0 | y) = 2^(-r) · Σ_u ∏_c F_c(s_c(u)),   F_c(s) = Σ_v P_c(v) (-1)^popcount(s AND v)" },
  { p: "The residual risk is R(y) = 1 − P(b = x | y, Hb = 0). A frame is delivered only if the CRC passes and R(y) ≤ τ. CRC-8 needs 256 terms per frame; CRC-16 needs 65,536, computed by a compiled multi-core kernel at about 7,000 frames per second. The formula was checked against Monte-Carlo estimates: 0.003671 analytical vs 0.003663 simulated over 400,000 trials." },
  { p: "**Worked example.** Three bits with an even-parity rule; the decoder says each bit is 1 with probability 0.9, 0.8 and 0.3. The decision 110 passes and has probability 0.504. All parity-valid words together have probability 0.596, so R = 1 − 0.504/0.596 = 15.4 % and the frame is retransmitted. With 0.99, 0.98 and 0.01 the risk is 0.05 % and the frame is accepted." },
  { h2: "4.5 Learned neural channel code" },
  { bullets: [
    "**Encoder:** each byte (256 values) is mapped by a trainable table to 14 real symbols: 70 symbols per CRC-8 frame, 84 per CRC-16 frame. The burst profile adds a fixed interleaver.",
    "**Decoder:** a 3-layer MLP (512 ELU units per layer) sees all received symbols and outputs a 256-way posterior for every byte.",
    "**Training:** 5,000 Adam steps, batch 1,024, Eb/N0 uniform in 1-6 dB, through a differentiable channel in PyTorch; weights are exported so inference runs on numpy.",
  ] },
  { h2: "4.6 Learned risk gate (GuardCode-ML)" },
  { p: "SSG is an exact posterior only if the decoder's probabilities are calibrated, and a neural decoder is only approximately calibrated. GuardCode-ML is a gradient-boosted decision-tree classifier (scikit-learn HistGradientBoostingClassifier) trained on 6 million simulated frames per channel profile to predict whether a CRC-passing frame is wrong. Its inputs are the SSG risk, the minimum |LLR|, sorted per-byte confidences and runner-up gaps, the residual noise energy after re-encoding the decision, and a burst index. Its threshold uses the same 1 % budget; training and test seeds are disjoint. No ML gate is trained for CRC-16: only about 2 in 600,000 CRC-16 frames are wrong but pass the CRC, too few examples to learn from." },
  { h2: "4.7 Channel-adaptive profile controller" },
  { p: "Two profiles are trained (AWGN and burst). The receiver re-encodes its decision, measures residual energy in windows of 7 symbols, and smooths log10(max/median) of those energies (EWMA 0.02). With hysteresis (switch up at 0.44, down at 0.40) it selects the profile and feeds the choice back as 1 bit on the ACK." },
  { h2: "4.8 Risk ledger" },
  { p: "Summing R(y) over delivered frames gives the receiver its own estimate of how many undetected errors it delivered, without knowing the transmitted data." },
  { h2: "4.9 Real-world channel from measured traces (CRAWDAD)" },
  { p: `The burst model was fitted to the public **CRAWDAD niit/bit_errors** dataset (IEEE DataPort, DOI 10.15783/C7S88K): residual bit-error traces of IEEE 802.15.4 MicaZ motes, 20-byte frames, 5-12 m from a base station in four indoor setups. The ${CW.n_traces} traces hold ${n(CW.frames)} frames (${n(CW.bits)} payload bits). In a trace, a payload byte 0xAA means "correct" and an XOR with 0xAA gives the bits in error. A two-state Gilbert-Elliott hidden Markov model is fitted to every trace by the Baum-Welch algorithm (checked on synthetic data, where it recovers the planted parameters to within a few percent). Errors come in short bursts: the channel leaves the Bad state with probability about ${CW.pooled.p_bg.toFixed(2)} per bit, so a burst lasts about ${(1 / CW.pooled.p_bg).toFixed(0)} bits, and bits inside a burst are essentially random, matching one wrong 4-bit 802.15.4 symbol. Every simulated frame draws one measured trace's model. The codes, thresholds and ML model are taken unchanged from the burst profile, so this is a genuine test of generalisation.` },

  // ---------------- 5 code
  { h1: "5. Code" },
  { h2: "5.1 Software modules" },
  { table: [["Component", "File", "Technology"],
    ["CRC-8 / CRC-16, parity-check and dual-code matrices", "guardcode/crc.py", "Python, numpy"],
    ["Channels (AWGN, Gilbert-Elliott, Rayleigh, CRAWDAD)", "guardcode/channels.py", "numpy; torch for training"],
    ["Hamming(7,4) soft decoder, interleaver", "guardcode/hamming.py", "numpy"],
    ["Neural code: training / inference", "guardcode/model.py, train.py", "PyTorch / numpy"],
    ["Soft-Syndrome Guard", "guardcode/guard.py", "numpy; numba kernel for CRC-16"],
    ["Receiver line-up and acceptance rules", "guardcode/schemes.py", "numpy"],
    ["Learned risk gate (GuardCode-ML)", "guardcode/mlguard.py, scripts/train_mlguard.py", "scikit-learn"],
    ["CRAWDAD loader, Baum-Welch fit", "guardcode/crawdad.py, scripts/fit_crawdad.py", "numpy, numba"],
    ["Adaptive controller", "guardcode/adaptive.py", "numpy"],
    ["Link simulator, benchmark, plots", "guardcode/linksim.py, evaluate.py", "numpy, matplotlib"],
    ["Database layer", "guardcode/db.py", "SQLite (WAL mode)"],
    ["Live dashboard and SQL explorer", "dashboard/app.py, engine.py, templates", "Flask, Server-Sent Events, Chart.js"]], widths: [3, 3, 2], left: true, caption: "Table 5 - Software modules" },
  { h2: "5.2 Key code" },
  { h3: "CRC encoding and checking (guardcode/crc.py)" },
  { code: snippet("guardcode/crc.py", "    def encode(self, msgs):", "    # ---------- soft syndrome support") },
  { h3: "Soft-Syndrome Guard risk (guardcode/guard.py)" },
  { code: snippet("guardcode/guard.py", "    def risk_groups(self, P, rows=None):", "    def _idx32(self, m):") },
  { h3: "Receiver decision: CRC, guard and gates (guardcode/schemes.py)" },
  { code: snippet("guardcode/schemes.py", "    def decide(self, y, h, ebno_db):", "class Uncoded(Scheme):") },
  { h3: "GuardCode-ML features (guardcode/mlguard.py)" },
  { code: snippet("guardcode/mlguard.py", "def features(scheme, P, risk, llr, y, h, bits):", "class MLGuard:") },
  { h3: "CRAWDAD trace loader (guardcode/crawdad.py)" },
  { code: snippet("guardcode/crawdad.py", "def load_trace(path):", "def environment(path):") },
  { p: "The complete source code is in the project repository; Appendix A lists the commands to run it." },
  { h2: "5.3 Database design" },
  { p: "Every transmitted frame, from bulk runs and from live operation, is stored as one row. WAL (write-ahead logging) mode lets the live simulator write over 10,000 rows per second while the dashboard reads. Setup populates 1,000,000 frames; a retention policy keeps the newest 10 million rows during live use." },
  { table: [["Table", "Key columns", "Purpose"],
    ["frames", "run_id, ts, seq, scheme, channel, ebno_db, bit_errors, outcome, reliability, attempt, crc", "One row per frame (outcome: 0 OK, 1 detected, 2 undetected, 3 false reject)"],
    ["runs", "id, started_at, mode, notes", "Live sessions, bulk populations, benchmarks"],
    ["benchmark", "profile, crc, scheme, ebno_db, frames, fer, uer, goodput ...", "Aggregated Monte-Carlo results"],
    ["adaptive_log", "ts, burst_index, chosen_profile, true_channel", "Decisions of the adaptive controller"],
    ["v_summary (view)", "scheme, channel, ebno_db, counts", "Per-scheme summary"]], widths: [2, 5, 4], left: true, caption: "Table 6 - Database schema (data/guardcode.db)" },

  // ---------------- 6 output
  { h1: "6. Output Implementation" },
  { h2: "6.1 Simulation setup" },
  { table: [["Parameter", "Value"],
    ["Payload / CRC / frame", "32 bits / CRC-8 / 40 bits; or CRC-16 / 48 bits"],
    ["Channel symbols per frame", "CRC-8: 70 coded, 40 uncoded; CRC-16: 84 coded, 48 uncoded"],
    ["Eb/N0 range", "0 to 7 dB (energy per information bit)"],
    ["Burst channel", "Gilbert-Elliott: P(G->B) = 0.03, P(B->G) = 0.25, +8 dB noise in Bad state"],
    ["Measured channel", "Per-trace Gilbert-Elliott mixture fitted to 22 real 802.15.4 traces"],
    ["Frames per operating point", n(S.bench_frames)],
    ["Gate calibration", "Every gate rejects 1 % of correctly decoded frames at 3 dB (equal ARQ cost)"],
    ["ARQ model", "Selective repeat; goodput = correctly delivered info bits per channel use"]], widths: [3, 5], left: true, caption: "Table 7 - Simulation parameters" },
  { h2: "6.2 Live dashboard" },
  { p: "The running system is a Flask web application. A background thread transmits frames continuously through the selected channel with all eight receivers side by side, writes every frame to the database and pushes a snapshot to the browser twice a second. The operator controls Eb/N0, the channel (AWGN, burst, fading, measured 802.15.4), the CRC (CRC-8 / CRC-16), the frame rate and a 20-second burst storm." },
  { img: R("dashboard_top.png"), width: 600, ratio: 0.625, caption: "Figure 3 - Live dashboard during a burst storm at Eb/N0 = 2 dB (117,000 frames per receiver)" },
  { p: "In the run shown in Figure 3, the classic confidence test (Neural+CRC+LLR) let 123 corrupted frames through, GuardCode 53 and GuardCode-ML 32, while all neural receivers had about the same frame error rate and goodput. The adaptive receivers switched to the burst profile during the storm (burst index 0.453 above the 0.44 threshold)." },
  { img: R("dashboard_explorer.png"), width: 600, ratio: 0.625, caption: "Figure 4 - SQL explorer: undetected errors per receiver, queried from about one million logged frames" },
  { h2: "6.3 Benchmark outputs" },
  { img: R("benchmark_awgn.png"), width: 520, ratio: 0.73, caption: "Figure 5 - AWGN channel: BER, FER, undetected FER and goodput (CRC-8)" },
  { img: R("benchmark_burst.png"), width: 520, ratio: 0.73, caption: "Figure 6 - Gilbert-Elliott burst channel (CRC-8)" },
  { img: R("crawdad_traces.png"), width: 600, ratio: 0.29, caption: "Figure 7 - The 22 measured CRAWDAD traces: error rate per trace, burst lengths and errors per frame, measured vs fitted model" },
  { img: R("benchmark_crawdad.png"), width: 520, ratio: 0.73, caption: "Figure 8 - Measured 802.15.4 channel (CRC-8)" },
  { img: R("tradeoff_awgn.png"), width: 400, ratio: 0.71, caption: "Figure 9 - Undetected error rate vs retransmission cost (AWGN, 2-4 dB)" },
  { img: R("calibration.png"), width: 360, ratio: 0.83, caption: "Figure 10 - Predicted risk vs observed error frequency" },
  { p: `Calibration: the Hamming receiver's posteriors are exact on AWGN, so its points lie on the diagonal (${cal["Hamming decoder"].total_pred.toFixed(0)} undetected errors predicted vs ${cal["Hamming decoder"].total_obs} observed between 1 and 3 dB). The neural receiver is over-confident by a factor of about 2-4 (${cal["Neural decoder"].total_pred.toFixed(0)} vs ${cal["Neural decoder"].total_obs}) but keeps the ordering of frames by risk, which is what the gate needs; this miscalibration is also why the learned gate helps.` },
  { img: R("adaptive.png"), width: 480, ratio: 0.67, caption: "Figure 11 - Profile switching when the channel changes AWGN -> burst -> AWGN (4 dB)" },
  { p: `The controller switched ${ad.switches} times and had the matching profile active in ${ad.profile_correct_pct.toFixed(0)} % of the blocks; burst-phase goodput was ${ad.goodput_burst_adaptive.toFixed(3)} adaptive vs ${ad.goodput_burst_fixed.toFixed(3)} with the fixed AWGN profile.` },
  { p: "**Negative result.** A training-time version of the idea, a loss that penalises CRC-blind errors during training, did not change the share of undetected errors (0.106 %, 0.101 % and 0.099 % for λ = 0, 5 and 50). It is therefore not claimed." },

  // ---------------- 7 comparison tables
  { h1: "7. Comparison Tables" },
  { p: "All tables count undetected frames pooled over Eb/N0 = 0-7 dB, with every gate held to the same retransmission budget. Lower is better." },
  { h2: "7.1 All receivers on all channels (CRC-8)" },
  { table: allChannels, widths: [3, 2, 2, 2, 3], caption: "Table 8 - Undetected frames per receiver and channel, CRC-8" },
  { table: reductionTable, widths: [4, 2, 2, 2], caption: "Table 9 - Reduction in undetected frames" },
  { h2: "7.2 This work vs existing approaches" },
  { table: methodTable, widths: [4, 2, 2, 2, 2, 2, 2], caption: "Table 10 - Comparison with existing approaches (neural decoder)" },
  { p: `Only the longer CRC does better than GuardCode, and it pays with 8 extra bits on every frame: goodput from 3 dB upward falls from ${P.awgn.GuardCode.goodput_3db.toFixed(2)} to ${P16.awgn.GuardCode.goodput_3db.toFixed(2)} information bits per channel use on AWGN. GuardCode and GuardCode-ML keep the frame and the CRC unchanged.` },
  { h2: "7.3 Detailed results per channel" },
  { table: ueTable("awgn"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 11 - AWGN channel" },
  { table: ueTable("burst"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 12 - Burst channel" },
  { table: ueTable("crawdad"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 13 - Measured 802.15.4 channel" },
  { table: pointTable("awgn", "fer", (x) => x.toFixed(4)), widths: [3, 1, 1, 1, 1, 1, 1, 1, 1], caption: "Table 14 - Frame error rate per Eb/N0, AWGN" },
  { p: `The neural code beats Hamming(7,4) at the same rate: at 4 dB its FER is ${S.fer_gain["4.0"].neural.toFixed(3)} vs ${S.fer_gain["4.0"].hamming.toFixed(3)}, a gain of about 1 dB.` },
  { h2: "7.4 CRC-8 vs CRC-16" },
  { table: crc16Table, widths: [3, 1, 1, 1, 1, 1, 1], caption: "Table 15 - Undetected frames with CRC-8 and CRC-16" },
  { p: `CRC-16 removes about 99 % of CRC-8's undetected errors. The guard still lowers the CRC-16 remainder (burst ${P16.burst["Neural+CRC"].ue} to ${P16.burst.GuardCode.ue}, measured ${P16.crawdad["Neural+CRC"].ue} to ${P16.crawdad.GuardCode.ue}), but these single-digit counts show the direction rather than a precise percentage.` },
  { h2: "7.5 Trade-off at different retransmission budgets" },
  { table: [["Decoder (channel)", "No gate", "LLR @1 %", "SSG @1 %", "LLR @5 %", "SSG @5 %"], trRow("awgn", "Neural"), trRow("awgn", "Hamming"), trRow("burst", "Neural"), trRow("burst", "Hamming")],
    widths: [3, 2, 2, 2, 2, 2], caption: "Table 16 - Undetected frame error rate at 1 % and 5 % false-reject budgets" },

  // ---------------- 8 conclusion
  { h1: "8. Conclusion and Future Work" },
  { p: "GuardCode shows that the CRC carries more information than a pass/fail bit. Combined with the decoder's soft output, the CRC's exact structure tells the receiver how likely a passing frame is to be wrong, and acting on that estimate removes about half the undetected errors left by the classic test, at no extra bandwidth and with the standard CRC unchanged. A learned gate removes more, and the result holds on a channel built from real 802.15.4 traces. The work also produced a neural code that beats Hamming(7,4) by about 1 dB, a live dashboard, a database of over a million frames, CRC-8 and CRC-16 support, and a draft patent with VIT invention disclosure forms." },
  { p: "Future work:" },
  { bullets: [
    "CRC-24/32, where the exact 2^r-term sum must be replaced by a codeword-side sum or a list of the most likely alternative frames;",
    "LDPC, polar and convolutional decoders, and HARQ soft combining;",
    "the 802.11 traces of the CRAWDAD dataset, and slow fading within traces;",
    "an implementation on a software-defined-radio testbed (TRL 4).",
  ] },

  // ---------------- refs
  { h1: "References" },
  { numbers: [
    "B. A. Forouzan, Data Communications and Networking, 5th ed., McGraw-Hill.",
    "C.-Y. Lou, B. Daneshrad, R. D. Wesel, \"Convolutional-code-specific CRC code design,\" IEEE Trans. Communications, 2015.",
    "\"Autoencoders with list decoding for reliable data transmission over noisy channels,\" US Patent Application 2023/0198672 A1.",
    "\"Hierarchical deep channel coding,\" US Patent 12,155,482 B2.",
    "Y. Jiang et al., \"Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels,\" NeurIPS 2019.",
    "A. R. Raghavan, C. W. Baum, \"A reliability output Viterbi algorithm with applications to hybrid ARQ,\" IEEE Trans. Information Theory, 1998.",
    "G. D. Forney, \"Exponential error bounds for erasure, list, and decision feedback schemes,\" IEEE Trans. Information Theory, 1968.",
    "R. G. Gallager, Low-Density Parity-Check Codes, MIT Press, 1963.",
    "E. N. Gilbert, \"Capacity of a burst-noise channel,\" BSTJ, 1960; E. O. Elliott, \"Estimates of error rates for codes on burst-noise channels,\" BSTJ, 1963.",
    "\"Error detection strategies for CRC-concatenated polar codes under successive cancellation list decoding,\" DLR, 2023.",
    "A. Iqbal, K. Shahzad, S. A. Khayam, Y. Cho, \"CRAWDAD dataset niit/bit_errors (v. 2008-07-08),\" IEEE DataPort, DOI 10.15783/C7S88K.",
    "L. R. Rabiner, \"A tutorial on hidden Markov models and selected applications in speech recognition,\" Proceedings of the IEEE, 1989.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, 2011.",
  ], ref: "num2" },

  // ---------------- appendix
  { h1: "Appendix A - How to Run" },
  { code: "Naresh_Computer_Networks_Project\\\n  setup.bat            one-click: venv, dependencies, benchmark import, 1M-frame database\n  run_dashboard.bat    live dashboard at http://127.0.0.1:5000  (explorer: /explorer)\n\nManual (inside .venv):\n  python -m guardcode.evaluate --frames 300000                        CRC-8, AWGN + burst\n  python -m guardcode.evaluate --profiles crawdad                     measured 802.15.4 channel\n  python -m guardcode.evaluate --crc CRC-16 --profiles awgn,burst,crawdad\n  python scripts\\fit_crawdad.py                                       refit the CRAWDAD traces\n  python scripts\\train_mlguard.py --frames 6000000                    retrain GuardCode-ML\n  python scripts\\train_all.py --crc CRC-16                            CRC-16 neural codes\n  python scripts\\analysis.py ; python scripts\\summarize.py\n  python scripts\\populate_db.py --frames 1000000\n  python dashboard\\app.py" },
];

build(blocks, path.join(ROOT, "docs", "GuardCode_Project_Report.docx"));
