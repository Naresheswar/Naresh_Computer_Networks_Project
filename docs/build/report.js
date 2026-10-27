// Project report (BCSE308L). Run after scripts/summarize.py.
const path = require("path");
const fs = require("fs");
const { build } = require("./docgen");
const ROOT = path.resolve(__dirname, "..", "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "summary.json"), "utf8"));
const R = (f) => path.join(ROOT, "results", f);
const A = S.analysis;
const pct = (x) => `${x.toFixed(0)} %`;
const e = (x) => (x === 0 ? "0" : x.toExponential(2));
const P = S.pooled, RD = S.reduction;

const sch = ["Uncoded+CRC", "Hamming+CRC", "Hamming+CRC+SSG", "Neural+CRC", "Neural+CRC+LLR", "GuardCode", "GuardCode-ML"];
const P16 = S.pooled_crc16, CW = S.crawdad;
const n = (x) => x.toLocaleString("en-IN");
const crc16Table = [["Receiver", "AWGN", "Burst", "Real 802.15.4 (CRAWDAD)"],
  ...["Uncoded+CRC", "Hamming+CRC", "Hamming+CRC+SSG", "Neural+CRC", "Neural+CRC+LLR", "GuardCode"].map((s) => [s, ...["awgn", "burst", "crawdad"].map((p) => n(P16[p][s].ue))])];
function pointTable(prof, metric, fmt) {
  const eb = ["0", "1", "2", "3", "4", "5", "6", "7"];
  return [["Scheme", ...eb.map((x) => `${x} dB`)], ...sch.map((s) => [s, ...eb.map((x) => fmt(S.points[prof][s][x][metric]))])];
}
const ueTable = (prof) => [["Scheme", "Frames", "Frame errors", "Undetected", "% of errors undetected", "Ledger estimate (accepted)"],
  ...sch.map((s) => { const v = P[prof][s]; return [s, v.frames.toLocaleString("en-IN"), v.fe.toLocaleString("en-IN"), v.ue.toLocaleString("en-IN"), (100 * v.ue / Math.max(1, v.fe)).toFixed(3), s.includes("SSG") || s === "GuardCode" ? v.est.toFixed(0) : "-"]; })];
const tr = (prof, lab) => A[`tradeoff_${prof}`][lab];
const trRow = (prof, lab) => {
  const t = tr(prof, lab); const idx = [3, 6].map((i) => i);
  return [`${lab} (${prof.toUpperCase()})`, e(t.uer_llr[0]), ...idx.flatMap((i) => [e(t.uer_llr[i]), e(t.uer_ssg[i])])];
};
const cal = A.calibration;
const ad = A.adaptive;

const blocks = [
  // ---------------- cover
  { space: 600 },
  { center: "GuardCode: Suppressing Undetected Frame Errors", bold: true, size: 36, after: 60 },
  { center: "with a Soft-Syndrome Guard on CRC-Protected Neural FEC Links", bold: true, size: 32, after: 600 },
  { center: "Project Report", bold: true, size: 26, after: 60 },
  { center: "BCSE308L - Computer Networks", bold: true, size: 26, after: 60 },
  { center: "BCSE308P - Computer Networks Lab", bold: true, size: 24, after: 600 },
  { center: "Submitted by", italics: true, size: 24, after: 60 },
  { center: "S Naresh Nireenjan (24BCE2353)", bold: true, size: 26, after: 40 },
  { center: "Dharshan J (24BCE2335)", bold: true, size: 26, after: 40 },
  { center: "Sachin Saravanan S (24BCE2321)", bold: true, size: 26, after: 600 },
  { center: "Under the guidance of", italics: true, size: 24, after: 60 },
  { center: "Dr. Anusha N", bold: true, size: 26, after: 1000 },
  { center: "School of Computer Science and Engineering", bold: true, size: 26, after: 60 },
  { center: "Vellore Institute of Technology, Vellore", bold: true, size: 26, after: 60 },
  { center: "October, 2026", bold: true, size: 26 },
  { pagebreak: true },

  // ---------------- abstract
  { h1: "Abstract" },
  { p: `Every frame on a data link carries a cyclic redundancy check (CRC). The receiver decodes the frame, recomputes the CRC and either delivers the frame or asks for a retransmission. The weak spot of this design is the **undetected error**: a frame that is decoded wrongly but still happens to satisfy the CRC, so corrupted data is delivered silently. This project builds and evaluates **GuardCode**, a receiver that closes much of that gap without adding a single bit to the frame and without changing the CRC polynomial.` },
  { p: `The key component, the **Soft-Syndrome Guard (SSG)**, uses the decoder's soft output and the exact algebra of the standard CRC to compute, for every frame that passes the CRC, the probability that it is still wrong. Frames whose residual risk is too high are retransmitted. A learned neural channel code (32-bit payload, CRC-8, 70 channel symbols) is trained in PyTorch and gains about 1 dB over Hamming(7,4) at the same rate. The system is evaluated by Monte-Carlo simulation on AWGN and Gilbert-Elliott burst channels with ${S.bench_frames.toLocaleString("en-IN")} frames per operating point. At an identical retransmission budget, SSG removes ${pct(RD.awgn.guard_vs_llr)} (AWGN) and ${pct(RD.burst.guard_vs_llr)} (burst) of the undetected errors left by the classic confidence-threshold test, and ${pct(RD.awgn.hamssg_vs_ham)} / ${pct(RD.burst.hamssg_vs_ham)} of those of a plain Hamming receiver. A gradient-boosted machine-learning gate (GuardCode-ML) built on the same receiver-side information removes a further ${pct(RD.awgn.ml_vs_guard)} (AWGN) and ${pct(RD.burst.ml_vs_guard)} (burst). The system was also tested on a channel fitted to **real measured error traces**: ${CW.n_traces} traces (${n(CW.frames)} frames) from IEEE 802.15.4 radios in the public CRAWDAD niit/bit_errors dataset. Without any re-tuning, GuardCode left ${pct(RD.crawdad.guard_vs_llr)} fewer undetected errors there than the classic test. Every part of the system supports both CRC-8 and CRC-16. A channel-adaptive controller, a live web dashboard and a SQLite database that logs every simulated frame complete the system. A provisional patent specification is drafted from the work.` },
  { p: "**Keywords:** CRC, undetected error probability, ARQ, forward error correction, neural channel codes, Hamming code, burst errors, data link layer." },

  // ---------------- 1
  { h1: "1. Introduction" },
  { h2: "1.1 Background" },
  { p: "Module 3 of the course covers error detection and correction (parity, checksum, CRC, Hamming code) and flow and error control (Stop-and-Wait, Go-Back-N and Selective-Repeat ARQ). In a real link these mechanisms work together. A forward error correction (FEC) code repairs most channel errors. The CRC then decides whether the repaired frame can be trusted. ARQ retransmits the frames the CRC rejects." },
  { p: "A CRC with r check bits misses roughly one in 2^r of the error patterns that reach it. For short CRCs used in low-latency, IoT and control traffic (CRC-8 to CRC-16), this residual is not negligible: one undetected error per few thousand frame errors is enough to corrupt a sensor stream or a control command. The usual remedies have real costs:" },
  { bullets: [
    "a longer CRC costs payload on every frame;",
    "a redesigned polynomial breaks compatibility with standards;",
    "simple confidence thresholds reject many good frames.",
  ] },
  { h2: "1.2 Problem statement" },
  { p: "Can a receiver lower the undetected-error rate of a CRC-protected link, at a fixed retransmission budget, without extra redundancy and without changing the CRC or the frame format?" },
  { h2: "1.3 Objectives" },
  { numbers: [
    "Build a link-level simulator: CRC-8 framing, Hamming(7,4) and learned neural FEC, AWGN, burst and fading channels, and ARQ accounting.",
    "Design a receiver test that uses the CRC's algebraic structure and the decoder's soft output to estimate the residual risk of each frame (the Soft-Syndrome Guard).",
    "Compare it fairly with plain CRC acceptance and with the classic minimum-|LLR| threshold test, at equal retransmission cost.",
    "Add a learned (machine-learning) acceptance gate and check whether it improves on the analytic guard.",
    "Validate the receiver on a channel built from real measured error traces, and support both CRC-8 and CRC-16.",
    "Add a channel-adaptive profile controller, a live monitoring dashboard and a large frame database.",
    "Assess patentability and draft a provisional patent specification.",
  ], ref: "num" },
  { h2: "1.4 Mapping to the syllabus" },
  { table: [["Course topic", "Where it appears in this project"],
    ["Module 3 - CRC (polynomial division)", "guardcode/crc.py: bit-serial CRC register, generator matrix, parity-check matrix"],
    ["Module 3 - Hamming code", "guardcode/hamming.py: Hamming(7,4), soft ML decoding, block interleaver"],
    ["Module 3 - Burst errors, interleaving", "Gilbert-Elliott burst channel; interleaved Hamming and neural profiles"],
    ["Module 3 - ARQ (Selective Repeat)", "Goodput model in linksim.py; NAK on CRC or guard rejection"],
    ["Module 1 - Transmission impairments", "AWGN, burst noise and Rayleigh fading channel models"],
    ["Module 6 - QoS, reliability", "Residual (undetected) error rate as a QoS metric; risk ledger"],
    ["Lab - socket/web programming", "Flask web server with Server-Sent Events, REST API, SQL explorer"]], widths: [3, 5], left: true },

  // ---------------- 2
  { h1: "2. Literature Survey and Prior Art" },
  { table: [["Work", "Idea", "Limitation addressed by GuardCode"],
    ["CRC polynomial design for a given inner code (Lou, Wesel et al., UCLA)", "Choose g(x) to minimise undetected errors for a convolutional code", "Requires changing the standard polynomial; GuardCode keeps g(x)"],
    ["CRC-aided list decoding (polar CA-SCL; US 2023/0198672 A1, list autoencoders)", "CRC selects among decoder candidates; CRC treated as data bits during training", "Decoder-specific; does not estimate residual risk of the accepted frame"],
    ["Turbo Autoencoder, DeepJSCC, hierarchical deep channel coding (US 12155482 B2)", "Neural channel codes trained end-to-end", "Use parity/LLR checks only as runtime flags"],
    ["Forney erasure decoding; ROVA (Raghavan and Baum); threshold tests for polar UER", "Reject when decoder confidence is low", "Confidence is blind to which errors the CRC would catch"],
    ["Gallager's parity-check probability (LDPC)", "Probability that a parity check is satisfied from bit probabilities", "Used inside iterative decoders, not as a CRC acceptance test"]],
    widths: [3, 3, 3], left: true, caption: "Table 1 - Related work" },
  { p: "A search of Google Patents, Justia and arXiv found no document that computes the CRC-conditioned posterior probability of an accepted frame being wrong through the dual-code expansion of a standard CRC and uses it as an ARQ acceptance gate. The closest patent, US 2023/0198672 A1, explicitly treats CRC bits as information bits during training and uses the CRC only to check list candidates. Section 7 discusses what this means for patentability." },

  // ---------------- 3
  { h1: "3. Proposed System" },
  { img: R("architecture.png"), width: 600, ratio: 0.43, caption: "Figure 1 - GuardCode system architecture" },
  { h2: "3.1 Frame format and CRC as a linear code" },
  { p: `A frame carries k = 32 payload bits and r = 8 CRC bits with g(x) = x^8 + x^2 + x + 1. Since the CRC (zero initial value) is linear, crc(m) = m G (mod 2), where row i of G is the CRC of the i-th unit vector, which is exactly the long division done in the lab. Every valid 40-bit frame b satisfies H b = 0 with H = [G^T | I_8]. Exhaustive search shows the minimum distance of this frame code is ${S.crc_dmin}: every pattern of up to ${S.crc_dmin - 1} bit errors is always detected.` },
  { p: `The whole system also runs with **CRC-16** (g(x) = x^16 + x^12 + x^5 + 1, CRC-16/CCITT as used by HDLC, X.25 and Bluetooth). The frame then has 48 bits (32 payload + 16 CRC) and is coded into 84 channel symbols, keeping 14 symbols per byte. The dual-code sum of Section 3.2 then has 2^16 = 65,536 terms per frame. A compiled, multi-core implementation (numba) evaluates it at about 7,000 frames per second on a 4-core laptop CPU, roughly 40 times faster than plain numpy. The risk is computed only for frames that pass the CRC, because only those can be delivered. The minimum distance of the 48-bit CRC-16 frame code is ${S.crc16_dmin}.` },
  { h2: "3.2 Soft-Syndrome Guard" },
  { p: "The decoder outputs a posterior over each **group** of frame bits: the 256 values of a byte for the neural decoder, the 16 messages of a 4-bit Hamming block, or one bit for uncoded BPSK. The hard decision x takes the most likely value of every group. Knowing that the transmitted frame must be a CRC codeword, Bayes' rule gives the probability that the decision is right:" },
  { eq: "P(b = x | y, Hb = 0) = \u220F_c P_c(x_c) / P(Hb = 0 | y)" },
  { p: "The denominator is the probability that a frame drawn from the decoder's posterior satisfies the CRC. It is computed exactly by summing over the 2^r = 256 words u of the dual code. Each group contributes one value of the Walsh-Hadamard transform (WHT) of its posterior:" },
  { eq: "P(Hb = 0 | y) = 2^(-8) \u00B7 \u03A3_u \u220F_c F_c(s_c(u)),   F_c(s) = \u03A3_v P_c(v) (-1)^popcount(s AND v)" },
  { p: "Here s_c(u) is the part of the row u^T H that falls inside group c. The residual risk is R(y) = 1 \u2212 P(b = x | y, Hb = 0). A frame is delivered only if the CRC passes **and** R(y) \u2264 \u03C4; otherwise a NAK triggers retransmission. The cost per frame is one 256-point WHT per byte plus a 256 x 5 gather-and-product, which is negligible next to the decoder. Two numerical checks confirmed the formula:" },
  { bullets: [
    "against a 400,000-trial Monte-Carlo estimate: 0.003671 analytical vs 0.003663 simulated;",
    "for correlated byte posteriors around a CRC-valid decision: 0.3236 analytical vs 0.3246 simulated (1,000,000 trials).",
  ] },
  { p: "**Why groups matter.** A simpler variant treats the 40 bits as independent. That ignores the fact that bits inside the same byte or Hamming block fail together, so its risk estimate is badly miscalibrated. The group/WHT form keeps that correlation exactly; Section 6.3 compares the two." },
  { p: "**Why SSG beats a confidence threshold.** A minimum-|LLR| test rejects a frame if any single bit is uncertain, even when that uncertainty is confined to patterns the CRC would certainly catch. SSG only counts error patterns that the CRC cannot see, so it spends the retransmission budget on the frames that are actually dangerous." },
  { h2: "3.3 Learned neural channel code" },
  { bullets: [
    "**Encoder:** the 40-bit frame is cut into 5 bytes. A trainable table maps each byte (256 values) to 14 real symbols, giving 70 symbols (rate 32/70, the same as Hamming(7,4)). The frame is normalised to unit energy per symbol. The burst profile adds a fixed interleaver.",
    "**Decoder:** a 3-layer MLP (512 ELU units per layer) sees all 70 received symbols and outputs 256-way log-posteriors for each byte. Bit posteriors are their marginals, which is what SSG needs.",
    "**Training:** 5,000 Adam steps, batch 1,024, Eb/N0 uniform in 1-6 dB, through a differentiable channel (PyTorch). Weights are exported to .npz so the simulator and dashboard run on numpy only.",
  ] },
  { h2: "3.4 Channel-adaptive profile controller" },
  { p: "Two profiles are trained, one for AWGN and one for Gilbert-Elliott bursts. After decoding, the receiver re-encodes the decision and measures residual energy in windows of 7 symbols. The burst index is log10(max/median) of those window energies. An EWMA (factor 0.02) with hysteresis (switch up at 0.44, down at 0.40) selects the profile, which is fed back in the ACK. Each profile has its own calibrated \u03C4." },
  { h2: "3.5 Risk ledger" },
  { p: "Summing R(y) over frames gives the receiver its own estimate of how many undetected errors it has delivered, without knowing the transmitted data. Section 6.4 checks how far this estimate can be trusted." },

  { h2: "3.6 Learned risk classifier (GuardCode-ML)" },
  { p: "The Soft-Syndrome Guard is an exact posterior only when the decoder's posteriors are calibrated, and a neural decoder is only approximately calibrated. GuardCode-ML therefore adds a second, learned opinion. A gradient-boosted decision-tree classifier (scikit-learn HistGradientBoostingClassifier, open source) is trained on CRC-passing frames to predict whether the frame is nevertheless wrong. Its inputs are all available at the receiver: the SSG risk, the smallest |LLR|, the per-byte posterior confidence and runner-up gap, the residual energy after re-encoding the decoded bits, and the burst index. It needs no extra bits and leaves the CRC polynomial unchanged. Its acceptance threshold is calibrated to the same retransmission budget as the other gates (1 % of correct frames rejected at 3 dB). Training and benchmark random seeds are disjoint. Each profile is trained on 6 million simulated frames. For CRC-16 no ML gate is trained: only about 2 in 600,000 CRC-16 frames are wrong but pass the CRC, too few positive examples to train a classifier honestly." },
  { h2: "3.7 Real-world channel from measured traces (CRAWDAD)" },
  { p: `Simulated channels are only as realistic as their parameters. To test GuardCode on real error behaviour, the burst model was fitted to the public **CRAWDAD niit/bit_errors** dataset (IEEE DataPort, DOI 10.15783/C7S88K): residual bit-error traces from IEEE 802.15.4 (ZigBee) MicaZ motes, with 20-byte frames sent 5-12 m to a base station in four indoor setups (OutRoom, PhdLab, Stair and Up_Floor). The ${CW.n_traces} traces hold ${n(CW.frames)} frames and ${n(CW.bits)} payload bits. Each line of a trace is one received frame; a payload byte 0xAA means "correct", and an XOR with 0xAA gives the bits in error.` },
  { p: "A two-state Gilbert-Elliott hidden Markov model (Good/Bad states, per-bit error probabilities e_G and e_B, transition probabilities p_GB and p_BG) is fitted to every trace by the Baum-Welch algorithm. On synthetic data the fit recovers the planted parameters to within a few percent. The measured traces show that:" },
  { bullets: [
    `errors arrive in short bursts: the channel leaves the Bad state with probability about ${CW.pooled.p_bg.toFixed(2)} per bit, so a burst lasts about ${(1 / CW.pooled.p_bg).toFixed(0)} bits on average;`,
    "inside a burst the bits are essentially random (e_B close to 0.5), which fits 802.15.4 physics, where each spread-spectrum symbol carries 4 bits and one wrong symbol scrambles a 4-bit chunk;",
    "outside bursts the channel is almost error-free;",
    "the traces differ strongly: measured bit error rates range from about 5 x 10^-7 to 3.6 x 10^-2.",
  ] },
  { p: "Because the traces differ so much, the project's **crawdad** channel uses a mixture: every simulated frame draws one measured trace, weighted by its frame count, and uses that trace's burst timing. The Bad-state noise rise is the BPSK-equivalent of the fitted error probabilities. Figure 9 (Section 6.8) shows that this reproduces the measured burst lengths closely. It still underestimates the rare frames with 25 or more bit errors, which a two-state model cannot fully capture." },
  { h1: "4. Implementation" },
  { table: [["Component", "File", "Technology"],
    ["CRC, parity-check and dual-code matrices", "guardcode/crc.py", "Python, numpy"],
    ["Channels (AWGN, Gilbert-Elliott, Rayleigh)", "guardcode/channels.py", "numpy + differentiable torch version"],
    ["Hamming(7,4) soft decoder, interleaver", "guardcode/hamming.py", "numpy"],
    ["Neural code: training / inference", "guardcode/model.py, train.py", "PyTorch / numpy"],
    ["Soft-Syndrome Guard (CRC-8 and CRC-16)", "guardcode/guard.py", "numpy; numba kernel for CRC-16"],
    ["Learned risk gate (GuardCode-ML)", "guardcode/mlguard.py, scripts/train_mlguard.py", "scikit-learn (gradient-boosted trees)"],
    ["CRAWDAD trace loader and burst-model fit", "guardcode/crawdad.py, scripts/fit_crawdad.py", "numpy, numba (Baum-Welch)"],
    ["Adaptive controller", "guardcode/adaptive.py", "numpy"],
    ["Link simulator, benchmark, plots", "guardcode/linksim.py, evaluate.py", "numpy, matplotlib"],
    ["Database layer", "guardcode/db.py", "SQLite (WAL mode)"],
    ["Live dashboard and SQL explorer", "dashboard/app.py, engine.py, templates", "Flask, Server-Sent Events, Chart.js"],
    ["Bulk database population, analyses", "scripts/populate_db.py, analysis.py", "Python"]], widths: [3, 3, 3], left: true, caption: "Table 2 - Software modules" },
  { h2: "4.1 Database design" },
  { p: "Every transmitted frame, from bulk runs and from live operation, is stored as one row. The database runs in WAL (write-ahead logging) mode, so the live simulator writes about 1,200 rows per second while the dashboard reads. Setup populates 1,000,000 frames. A retention policy keeps the newest 10 million rows during live use." },
  { table: [["Table", "Key columns", "Purpose"],
    ["frames", "run_id, ts, seq, scheme, channel, ebno_db, bit_errors, outcome, reliability, attempt, crc", "One row per frame (outcome: 0 OK, 1 detected, 2 undetected, 3 false reject)"],
    ["runs", "id, started_at, mode, notes", "Live sessions, bulk populations, benchmarks"],
    ["benchmark", "profile, crc, scheme, ebno_db, frames, fer, uer, goodput ...", "Aggregated Monte-Carlo results"],
    ["adaptive_log", "ts, burst_index, chosen_profile, true_channel", "Decisions of the adaptive controller"],
    ["v_summary (view)", "scheme, channel, ebno_db, counts", "Per-scheme summary for queries"]], widths: [2, 5, 4], left: true, caption: "Table 3 - Database schema (data/guardcode.db)" },
  { h2: "4.2 Live dashboard" },
  { p: "A background thread transmits frames continuously through the selected channel with all eight receivers side by side, including GuardCode-ML and the two adaptive receivers. Each frame is written to the database, and a snapshot is pushed to the browser twice a second. The operator can change Eb/N0, switch between AWGN, burst, fading and the measured 802.15.4 (CRAWDAD) channel, switch between CRC-8 and CRC-16, trigger a 20-second burst storm, and change the frame rate. The page shows:" },
  { bullets: [
    "a scoreboard with FER, BER, detected and undetected frames and goodput;",
    "live charts of cumulative undetected errors, rolling FER and goodput;",
    "the adaptive controller's burst index;",
    "a feed of undetected-error events.",
  ] },
  { p: "The /explorer page runs read-only SQL with preset queries." },

  // ---------------- 5
  { h1: "5. Simulation Setup" },
  { table: [["Parameter", "Value"],
    ["Payload / CRC / frame", "32 bits / CRC-8 (x^8+x^2+x+1) / 40 bits; or CRC-16 (x^16+x^12+x^5+1) / 48 bits"],
    ["Channel symbols per frame", "CRC-8: 70 for all coded schemes (rate 32/70), 40 uncoded; CRC-16: 84 coded, 48 uncoded"],
    ["Modulation reference", "Real-valued unit-energy symbols (BPSK-equivalent for Hamming/uncoded)"],
    ["Eb/N0 range", "0 to 7 dB (energy per information bit)"],
    ["Burst channel", "Gilbert-Elliott: P(G->B) = 0.03, P(B->G) = 0.25, +8 dB noise in Bad state"],
    ["Measured channel (CRAWDAD)", "Per-trace Gilbert-Elliott mixture fitted to 22 real 802.15.4 traces; tested with the burst profile's codes and thresholds, no re-tuning"],
    ["Frames per operating point", S.bench_frames.toLocaleString("en-IN")],
    ["Gate calibration", "Both gates reject 1 % of correctly decoded frames at 3 dB (equal ARQ cost)"],
    ["ARQ model", "Selective repeat; goodput = correctly delivered info bits per channel use"]], widths: [3, 5], left: true, caption: "Table 4 - Simulation parameters" },

  // ---------------- 6
  { h1: "6. Results" },
  { h2: "6.1 Error-rate performance" },
  { img: R("benchmark_awgn.png"), width: 560, ratio: 0.73, caption: "Figure 2 - AWGN channel: BER, FER, undetected FER and goodput" },
  { p: `The learned code beats Hamming(7,4) at the same rate. At 4 dB its FER is ${S.fer_gain["4.0"].neural.toFixed(3)} vs ${S.fer_gain["4.0"].hamming.toFixed(3)}, and at 6 dB it is ${S.fer_gain["6.0"].neural.toFixed(4)} vs ${S.fer_gain["6.0"].hamming.toFixed(4)}, a gain of roughly 1 dB. This raises ARQ goodput. Its BER is higher at very low SNR because a wrong byte decision costs several bits; the frame-level metrics matter more for ARQ.` },
  { table: pointTable("awgn", "fer", (x) => x.toFixed(4)), widths: [3, 1, 1, 1, 1, 1, 1, 1, 1], caption: "Table 5a - Frame error rate, AWGN" },
  { img: R("benchmark_burst.png"), width: 560, ratio: 0.73, caption: "Figure 3 - Gilbert-Elliott burst channel" },
  { h2: "6.2 Undetected errors at equal retransmission cost" },
  { table: ueTable("awgn"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 5b - Pooled over 0-7 dB, AWGN" },
  { table: ueTable("burst"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 5c - Pooled over 0-7 dB, burst channel" },
  { p: `On AWGN, GuardCode delivers ${P.awgn.GuardCode.ue} undetected frames against ${P.awgn["Neural+CRC+LLR"].ue} for the same decoder with the classic threshold test (${pct(RD.awgn.guard_vs_llr)} fewer) and ${P.awgn["Neural+CRC"].ue} with plain CRC acceptance (${pct(RD.awgn.guard_vs_plain)} fewer). On the burst channel the reductions are ${pct(RD.burst.guard_vs_llr)} and ${pct(RD.burst.guard_vs_plain)}. Applying the same guard to the Hamming receiver removes ${pct(RD.awgn.hamssg_vs_ham)} (AWGN) and ${pct(RD.burst.hamssg_vs_ham)} (burst) of its undetected errors, which confirms that the method does not depend on the decoder. The price is small because the gates were calibrated to the same false-rejection budget. From 3 dB upward, where the link is usable, ARQ goodput stays within about 1 % of the unguarded receiver. At 0-2 dB the cost rises to a few percent, but goodput is already very low there.` },
  { img: R("undetected_share.png"), width: 520, ratio: 0.45, caption: "Figure 4 - Share of frame errors delivered undetected" },
  { h2: "6.3 Trade-off curves" },
  { p: "Figures 5 and 6 sweep the gate threshold. The x-axis is the percentage of good frames sent back (extra retransmissions); the y-axis is the remaining undetected error rate. Each decoder has three curves:" },
  { bullets: [
    "the classic min-|LLR| test (dashed);",
    "the bit-wise SSG variant (dotted);",
    "the group/WHT SSG used in GuardCode (solid).",
  ] },
  { p: "The group form is consistently the lowest, which shows the gain comes from using the code's structure correctly, not from the threshold alone." },
  { img: R("tradeoff_awgn.png"), width: 430, ratio: 0.71, caption: "Figure 5 - Undetected error rate vs retransmission cost (AWGN, 2-4 dB)" },
  { img: R("tradeoff_burst.png"), width: 430, ratio: 0.71, caption: "Figure 6 - Same, burst channel" },
  { table: [["Decoder (channel)", "No gate", "LLR @1 %", "SSG @1 %", "LLR @5 %", "SSG @5 %"], trRow("awgn", "Neural"), trRow("awgn", "Hamming"), trRow("burst", "Neural"), trRow("burst", "Hamming")],
    widths: [3, 2, 2, 2, 2, 2], caption: "Table 6 - Undetected frame error rate at 1 % and 5 % false-reject budgets" },
  { h2: "6.4 Calibration and the risk ledger" },
  { img: R("calibration.png"), width: 380, ratio: 0.83, caption: "Figure 7 - Predicted risk vs observed error frequency" },
  { p: `Frames are binned by predicted risk R(y), and each bin's observed fraction of wrong frames is plotted.` },
  { bullets: [
    `**Hamming receiver:** its posteriors are exact on AWGN and its points lie on the diagonal. Summed over all CRC-passing frames between 1 and 3 dB, it predicted ${cal["Hamming decoder"].total_pred.toFixed(0)} undetected errors and ${cal["Hamming decoder"].total_obs} occurred, so R(y) is a true probability here.`,
    `**Neural receiver:** its curve runs parallel to the diagonal but a factor of about 2-4 higher (${cal["Neural decoder"].total_pred.toFixed(0)} predicted vs ${cal["Neural decoder"].total_obs} observed). The network is somewhat over-confident, so the absolute risk is under-stated. The ordering of frames by risk is preserved, which is what the gate needs; this is why the guard still works well with it.`,
  ] },
  { p: `The ledger is only as good as the decoder's posteriors. Among frames the guard **accepted** (last column of Tables 5b and 5c), the Hamming receiver, whose posteriors are exact on AWGN, predicted ${P.awgn["Hamming+CRC+SSG"].est.toFixed(0)} undetected errors against ${P.awgn["Hamming+CRC+SSG"].ue} observed. The neural decoder under-counts on AWGN (${P.awgn.GuardCode.est.toFixed(0)} predicted vs ${P.awgn.GuardCode.ue} observed): the errors that survive the gate are confidently-wrong decisions, so its posteriors are over-confident in that tail. The Hamming receiver also under-counts on the burst channel, where its AWGN noise model is wrong. Better calibration of the neural decoder is listed as future work.` },
  { h2: "6.5 Channel-adaptive operation" },
  { img: R("adaptive.png"), width: 520, ratio: 0.67, caption: "Figure 8 - Profile switching when the channel changes AWGN -> burst -> AWGN (4 dB)" },
  { p: `The burst index separates the channel classes. The controller switched ${ad.switches} times in the three-phase scenario and had the matching profile active in ${ad.profile_correct_pct.toFixed(0)} % of the blocks. Goodput during the burst phase was ${ad.goodput_burst_adaptive.toFixed(3)} adaptive vs ${ad.goodput_burst_fixed.toFixed(3)} with the fixed AWGN profile; in the AWGN phases it was ${ad.goodput_awgn_adaptive.toFixed(3)} vs ${ad.goodput_awgn_fixed.toFixed(3)}. The gain from switching profiles is modest with the current burst-trained model; the main benefit is that each profile carries its own calibrated threshold.` },
  { h2: "6.6 Negative result: training-time guard loss" },
  { p: "A training-time version of the same idea was also tried: a differentiable loss that penalises the probability of CRC-blind errors during training (guardcode/losses.py). Fine-tuning with weights \u03BB = 5 and 50 did not change the measured share of undetected errors (0.106 %, 0.101 % and 0.099 % for \u03BB = 0, 5 and 50 at 2-4 dB). With byte-level outputs, a single wrong byte is always caught by CRC-8, and the multi-byte coincidences that escape cannot be shaped by per-byte posteriors. This variant is therefore not claimed. The models are kept in models/ablation/ for reproducibility." },

  // ---------------- 7
  { h2: "6.7 Learned gate: GuardCode-ML" },
  { p: `Pooled over Eb/N0 = 0-7 dB at the same retransmission budget, GuardCode-ML left ${P.awgn["GuardCode-ML"].ue.toLocaleString("en-IN")} undetected frames on AWGN against ${P.awgn.GuardCode.ue.toLocaleString("en-IN")} for the Soft-Syndrome Guard (${pct(RD.awgn.ml_vs_guard)} fewer) and ${P.burst["GuardCode-ML"].ue.toLocaleString("en-IN")} against ${P.burst.GuardCode.ue.toLocaleString("en-IN")} on the burst channel (${pct(RD.burst.ml_vs_guard)} fewer). Relative to the classic min-|LLR| test the reductions are ${pct(RD.awgn.ml_vs_llr)} (AWGN) and ${pct(RD.burst.ml_vs_llr)} (burst). Goodput is unchanged to within about 0.1 % from 3 dB upward. The classifier is trained on a limited number of wrong frames, so the gain should be confirmed with larger runs before it is claimed as general.` },
  { h2: "6.8 Validation on the measured 802.15.4 channel (CRAWDAD)" },
  { img: R("crawdad_traces.png"), width: 600, ratio: 0.29, caption: "Figure 9 - The 22 measured CRAWDAD traces: bit error rate per trace, burst lengths and errors per frame, measured vs fitted model" },
  { table: ueTable("crawdad"), widths: [3, 2, 2, 2, 2, 2], caption: "Table 7 - Pooled over 0-7 dB, measured 802.15.4 channel (CRC-8)" },
  { p: `On the channel built from the real traces, with codes, thresholds and the ML model taken unchanged from the simulated burst profile, GuardCode delivered ${n(P.crawdad.GuardCode.ue)} undetected frames against ${n(P.crawdad["Neural+CRC+LLR"].ue)} for the classic min-|LLR| test (${pct(RD.crawdad.guard_vs_llr)} fewer) and ${n(P.crawdad["Neural+CRC"].ue)} for plain CRC acceptance (${pct(RD.crawdad.guard_vs_plain)} fewer). GuardCode-ML lowered this to ${n(P.crawdad["GuardCode-ML"].ue)} (${pct(RD.crawdad.ml_vs_guard)} fewer than GuardCode). The guard cut the Hamming receiver's undetected errors from ${n(P.crawdad["Hamming+CRC"].ue)} to ${n(P.crawdad["Hamming+CRC+SSG"].ue)} (${pct(RD.crawdad.hamssg_vs_ham)} fewer). Because nothing was tuned on this channel, the result shows that the method carries over to measured burst statistics.` },
  { img: R("benchmark_crawdad.png"), width: 560, ratio: 0.73, caption: "Figure 10 - Measured 802.15.4 channel: BER, FER, undetected FER and goodput (CRC-8)" },
  { h2: "6.9 CRC-8 versus CRC-16" },
  { table: crc16Table, widths: [3, 2, 2, 3], caption: "Table 8 - Undetected frames with CRC-16, pooled over 0-7 dB (300,000 frames per point)" },
  { p: `CRC-16 removes about 99 % of the undetected errors that CRC-8 lets through: the plain neural receiver drops from ${n(P.awgn["Neural+CRC"].ue)} to ${n(P16.awgn["Neural+CRC"].ue)} on AWGN and from ${n(P.crawdad["Neural+CRC"].ue)} to ${n(P16.crawdad["Neural+CRC"].ue)} on the measured channel. The guard still lowers the remainder (for example ${P16.burst["Neural+CRC"].ue} to ${P16.burst.GuardCode.ue} on the burst channel and ${P16.crawdad["Neural+CRC"].ue} to ${P16.crawdad.GuardCode.ue} on the measured channel). These counts are single-digit, so they show the direction rather than a precise percentage; measuring the CRC-16 gain precisely would need hundreds of millions of frames. The 8 extra CRC bits cost goodput: from 3 dB upward the neural receiver reaches ${P16.awgn.GuardCode.goodput_3db.toFixed(2)} information bits per channel use with CRC-16 against ${P.awgn.GuardCode.goodput_3db.toFixed(2)} with CRC-8 on AWGN. CRC-8 with GuardCode-ML therefore keeps the higher throughput at the cost of more undetected errors. Which choice is better depends on how much a single undetected error costs the application.` },
  { h1: "7. Patentability Assessment" },
  { table: [["Criterion", "Assessment"],
    ["Novelty", "No document found that uses the dual-code expansion of a standard CRC to compute the posterior probability that a CRC-passing frame is wrong, and uses it as an ARQ acceptance gate with a risk ledger and per-profile thresholds."],
    ["Inventive step", "Main risk. An examiner may combine reliability-output decoding (ROVA, Forney) with Gallager's parity-probability formula. Counter-arguments: neither applies the formula to an outer CRC to condition acceptance; there is a measured technical effect at equal retransmission cost; and the method is decoder-agnostic."],
    ["Industrial applicability", "Strong. Drops into existing receivers (Wi-Fi, BLE, 5G control channels, industrial IoT) without changing the standard frame or polynomial."],
    ["Section 3(k), Indian Patents Act", "Claims are drafted as a receiver method and apparatus with a technical effect (fewer corrupted frames delivered), not as an algorithm per se."]],
    widths: [2, 6], left: true, caption: "Table 9 - Patentability" },
  { p: "A draft provisional specification with 12 claims is provided in docs/GuardCode_Provisional_Patent_Draft.docx. Recommended next steps:" },
  { bullets: [
    "Have the VIT IPR cell or a registered patent agent run a professional search on InPASS and Espacenet, using CPC classes H03M13/09, H04L1/00 and H04L1/18.",
    "File a provisional application to secure the priority date.",
    "Complete the specification within 12 months.",
  ] },
  { p: "A granted patent cannot be promised. Examination in India typically takes 2-4 years, and expedited examination is available to students and start-ups." },

  // ---------------- 8
  { h1: "8. Conclusion and Future Work" },
  { p: "GuardCode shows that the CRC on a link carries more information than a pass/fail bit. Combined with soft decoder output, the CRC's exact algebraic structure tells the receiver how likely a passing frame is to be wrong, and acting on that estimate removes a large share of undetected errors at no extra bandwidth. The work also produced:" },
  { bullets: [
    "a neural channel code that beats Hamming(7,4) by about 1 dB;",
    "a live monitoring dashboard;",
    "a frame-level database with more than a million records;",
    "a validation on a channel built from 22 real 802.15.4 error traces;",
    "full support for both CRC-8 and CRC-16;",
    "a draft patent.",
  ] },
  { p: "Future work:" },
  { bullets: [
    "extend from CRC-8 and CRC-16 to CRC-24/32, where the exact 2^r-term sum must be replaced by a codeword-side sum or a list of the most likely alternative frames;",
    "use the 802.11 (Wi-Fi) traces of the CRAWDAD dataset, and model the slow fading that makes some measured frames carry 25 or more bit errors;",
    "apply SSG to LDPC/polar decoders and to HARQ soft combining;",
    "make the decoder SNR-aware so that posteriors stay calibrated on burst channels;",
    "implement the guard on an SDR testbed.",
  ] },

  // ---------------- refs
  { h1: "References" },
  { numbers: [
    "B. A. Forouzan, Data Communications and Networking, 5th ed., McGraw-Hill (course text: error detection/correction and ARQ).",
    "C.-Y. Lou, B. Daneshrad, R. D. Wesel, \"Convolutional-code-specific CRC code design,\" IEEE Trans. Communications, 2015 (arXiv:1506.02990).",
    "\"Autoencoders with list decoding for reliable data transmission over noisy channels,\" US Patent Application 2023/0198672 A1.",
    "\"Hierarchical deep channel coding,\" US Patent 12,155,482 B2.",
    "Y. Jiang, H. Kim, H. Asnani, S. Kannan, S. Oh, P. Viswanath, \"Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels,\" NeurIPS 2019.",
    "A. R. Raghavan, C. W. Baum, \"A reliability output Viterbi algorithm with applications to hybrid ARQ,\" IEEE Trans. Information Theory, 1998.",
    "G. D. Forney, \"Exponential error bounds for erasure, list, and decision feedback schemes,\" IEEE Trans. Information Theory, 1968.",
    "R. G. Gallager, Low-Density Parity-Check Codes, MIT Press, 1963.",
    "E. N. Gilbert, \"Capacity of a burst-noise channel,\" Bell System Technical Journal, 1960; E. O. Elliott, \"Estimates of error rates for codes on burst-noise channels,\" BSTJ, 1963.",
    "\"Error detection strategies for CRC-concatenated polar codes under successive cancellation list decoding,\" DLR, 2023.",
    "A. Iqbal, K. Shahzad, S. A. Khayam, Y. Cho, \"CRAWDAD dataset niit/bit_errors (v. 2008-07-08),\" IEEE DataPort, DOI 10.15783/C7S88K.",
    "L. R. Rabiner, \"A tutorial on hidden Markov models and selected applications in speech recognition,\" Proceedings of the IEEE, 1989 (Baum-Welch algorithm).",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, 2011.",
  ], ref: "num2" },

  // ---------------- appendix
  { h1: "Appendix A - How to Run" },
  { code: "Naresh_Computer_Networks_Project\\\n  setup.bat            one-click: venv, dependencies, benchmark import, 1M-frame database\n  run_dashboard.bat    live dashboard at http://127.0.0.1:5000  (explorer: /explorer)\n  run_experiments.bat  full benchmark + analyses\n  retrain_models.bat   retrain neural profiles (PyTorch)\n\nManual (inside .venv):\n  python -m guardcode.evaluate --frames 300000                        CRC-8, AWGN + burst\n  python -m guardcode.evaluate --profiles crawdad                     measured 802.15.4 channel\n  python -m guardcode.evaluate --crc CRC-16 --profiles awgn,burst,crawdad\n  python scripts\\fit_crawdad.py                                       refit the CRAWDAD traces\n  python scripts\\train_mlguard.py --frames 6000000                    retrain GuardCode-ML\n  python scripts\\train_all.py --crc CRC-16                            CRC-16 neural codes\n  python scripts\\analysis.py  and  python scripts\\summarize.py\n  python scripts\\populate_db.py --frames 1000000\n  python dashboard\\app.py" },
  { h1: "Appendix B - Core Code: Soft-Syndrome Guard" },
  { code: (() => { const g = fs.readFileSync(path.join(ROOT, "guardcode", "guard.py"), "utf8"); return g.slice(g.indexOf("class SoftSyndromeGuard")); })() },
];

build(blocks, path.join(ROOT, "docs", "GuardCode_Project_Report.docx"));
