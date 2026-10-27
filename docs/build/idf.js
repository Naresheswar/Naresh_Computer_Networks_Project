// VIT IPR&TT Cell Invention Disclosure Formats IDF-A (inventors) and IDF-B (invention), filled
// in from the project results. Run after scripts/summarize.py.
const path = require("path");
const fs = require("fs");
const { build, AlignmentType } = require("./docgen");
const ROOT = path.resolve(__dirname, "..", "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "summary.json"), "utf8"));
const R = (f) => path.join(ROOT, "results", f);
const P = S.pooled, RD = S.reduction, P16 = S.pooled_crc16, CW = S.crawdad;
const pct = (x) => `${x.toFixed(0)} %`;
const n = (x) => x.toLocaleString("en-IN");
const FILL = "[to be filled]";

const TITLE = "Soft-Syndrome Guard: a receiver method and system for suppressing undetected frame errors on cyclic-redundancy-check protected communication links using the exact CRC-conditioned posterior, a learned risk gate and channel-adaptive decoding";

const header = (form, docNo) => ({
  table: [["©VIT IPR&TTCELL", "Document No.", docNo],
    [`Invention Disclosure Format (IDF)-${form}`, "Issue No/Date", "2/01.02.2024"],
    ["", "Amd. No/Date", "0/00.00.0000"]],
  widths: [5, 2, 2], left: true,
});

// ------------------------------------------------------------------ IDF-A
const VIT = "Vellore Institute of Technology, Vellore - 632014, Tamil Nadu, India";
const students = [
  ["S Naresh Nireenjan", "24BCE2353"],
  ["Dharshan J", "24BCE2335"],
  ["Sachin Saravanan S", "24BCE2321"],
];
const inventorTable = (label, rows) => ({ table: [["(" + label + ")", ""], ...rows], widths: [3, 5], left: true });
const faculty = inventorTable("i", [
  ["Full Name", "Dr. Anusha N"],
  ["VIT Employee ID", FILL],
  ["Gender", FILL],
  ["Nationality", FILL],
  ["Position", FILL],
  ["Department", FILL],
  ["School/Center", FILL],
  ["Phone & E-Mail (Office)", FILL],
  ["Complete Address (Office)", `${FILL}, ${VIT}`],
]);
const studentTables = students.map(([name, reg], i) => inventorTable(["ii", "iii", "iv"][i], [
  ["Full Name", name],
  ["VIT Employee ID", `Not applicable (student) - Registration No. ${reg}`],
  ["Gender", FILL],
  ["Nationality", FILL],
  ["Position", "B.Tech student, Computer Science and Engineering"],
  ["Department", "Computer Science and Engineering"],
  ["School/Center", "School of Computer Science and Engineering (SCOPE)"],
  ["Phone & E-Mail (Office)", FILL],
  ["Complete Address (Office)", `School of Computer Science and Engineering, ${VIT}`],
]));

const idfA = [
  header("A", "02-IPR-R002"),
  { space: 120 },
  { h1: "1. Inventor(s) information" },
  faculty, { space: 160 },
  ...studentTables.flatMap((t) => [t, { space: 160 }]),
  { p: "NOTE: In case of co-inventors from external organization (Academic institution / Industry / R&D lab / Funding agency with specific IP sharing agreement) contact IPR&TT Cell before submitting the IDF. All inventors listed above are from VIT; there are no external co-inventors.", style: { italics: true } },
  { h2: "Declaration" },
  { p: "I declare that the information provided in this document is complete and true to the best of my knowledge and based on records." },
  { p: "Date: ____________________" },
  { table: [["Inventor Name", "Signature"],
    ["Dr. Anusha N", ""],
    ["S Naresh Nireenjan (24BCE2353)", ""],
    ["Dharshan J (24BCE2335)", ""],
    ["Sachin Saravanan S (24BCE2321)", ""]], widths: [4, 4], left: true },
  { space: 200 },
  { center: "---------------------END OF THE DOCUMENT-----------------------------", size: 20 },
];

// ------------------------------------------------------------------ IDF-B
const resultsTable = [["Receiver (same 1 % retransmission budget)", "AWGN", "Burst", "Measured 802.15.4 (CRAWDAD)"],
  ...[["Neural decoder + CRC-8 only", "Neural+CRC"], ["Neural + classic min-|LLR| test (prior art)", "Neural+CRC+LLR"],
    ["Neural + Soft-Syndrome Guard (invention)", "GuardCode"], ["Neural + learned risk gate (invention, ML)", "GuardCode-ML"],
    ["Hamming(7,4) + CRC-8 only", "Hamming+CRC"], ["Hamming(7,4) + Soft-Syndrome Guard", "Hamming+CRC+SSG"]]
    .map(([lab, s]) => [lab, ...["awgn", "burst", "crawdad"].map((p) => n(P[p][s].ue))])];

const idfB = [
  header("B", "02-IPR-R003"),
  { space: 120 },
  { h1: "1. Title of the invention" },
  { p: TITLE },

  { h1: "2. Field / Area of invention" },
  { p: "Digital communication networks: error control at the data-link and physical layers, more specifically receiver-side acceptance of frames protected by a cyclic redundancy check (CRC) together with forward error correction (FEC) and automatic repeat request (ARQ). Related areas: channel coding, neural-network channel codes, machine learning for communication receivers, wireless sensor and IoT links (for example IEEE 802.15.4, Bluetooth Low Energy, industrial control links)." },

  { h1: "3. Prior Patents and Publications from literature" },
  { table: [["Prior art", "What it does", "Gap addressed by this invention"],
    ["Lou, Daneshrad, Wesel, \"Convolutional-code-specific CRC code design\", IEEE Trans. Commun., 2015", "Chooses the CRC polynomial jointly with a convolutional code to minimise undetected errors", "Requires changing the standard CRC polynomial; the invention keeps the unmodified standard CRC"],
    ["US 2023/0198672 A1, \"Autoencoders with list decoding for reliable data transmission\"", "CRC selects among list-decoder candidates; CRC bits treated as data bits in training", "Decoder-specific; gives no probability that the accepted frame is still wrong"],
    ["US 12,155,482 B2, \"Hierarchical deep channel coding\"; Jiang et al., Turbo Autoencoder (NeurIPS 2019)", "Neural channel codes trained end-to-end", "Use parity or LLR checks only as runtime flags; no CRC-conditioned acceptance"],
    ["Forney, erasure decoding (IEEE Trans. IT, 1968); Raghavan and Baum, ROVA (IEEE Trans. IT, 1998); threshold tests for polar-code undetected error rate (DLR, 2023)", "Reject a frame when decoder confidence is low", "Confidence measures are blind to which error patterns the CRC would catch, so they waste the retransmission budget"],
    ["Gallager, Low-Density Parity-Check Codes (MIT Press, 1963)", "Probability that a parity check is satisfied, computed from bit probabilities", "Used inside iterative decoders, never as a CRC acceptance test with grouped posteriors"]],
    widths: [3, 3, 3], left: true, caption: "Table 1 - Summary of prior art" },
  { p: "A search of Google Patents, Justia and arXiv found no document that computes the CRC-conditioned posterior probability that an accepted frame is wrong through the dual-code expansion of an unmodified standard CRC, and uses it as an ARQ acceptance gate. A professional search on InPASS and Espacenet (CPC H03M13/09, H04L1/00, H04L1/18) by the IPR&TT Cell is recommended." },

  { h1: "4. Summary and background of the invention (gap / novelty)" },
  { p: "**Background.** Every frame on a data link carries a CRC. After the FEC decoder repairs channel errors, the receiver delivers the frame if its CRC is satisfied and otherwise requests a retransmission. A CRC with r check bits nevertheless accepts about one in 2^r of the error patterns that reach it. These **undetected errors** deliver corrupted data silently, and they matter most for the short CRCs (8-16 bits) used in IoT, sensor and control traffic." },
  { p: "**Gap.** The known remedies all cost something: a longer CRC costs payload on every frame, a redesigned polynomial breaks standards compatibility, and confidence thresholds reject many good frames because they cannot tell whether an uncertain pattern would be caught by the CRC anyway." },
  { p: "**Novelty.** The invention computes, for every frame that passes the CRC, the **exact posterior probability that it is still wrong**, using only the decoder's soft output and the algebra of the unmodified CRC. The normalising probability that a frame drawn from the decoder's posterior satisfies the CRC is evaluated exactly by a dual-code (Walsh-Hadamard) expansion over 2^r terms, in which each byte or block of the frame contributes the Walsh-Hadamard transform of its own posterior, so correlation inside a byte is kept exactly. Frames whose residual risk exceeds a threshold calibrated to a fixed retransmission budget are sent back. No bits are added and the CRC is not changed. Further novel elements are a learned (gradient-boosted) risk gate that uses this exact risk as one of its inputs, a running estimate of the number of undetected errors delivered (risk ledger), and pilot-free channel-adaptive profile selection with 1-bit feedback on the ACK." },

  { h1: "5. Objective(s) of invention" },
  { numbers: [
    "To lower the number of corrupted frames delivered to higher layers, at a fixed retransmission budget, without lengthening the CRC and without sending extra bits.",
    "To give the receiver the exact probability that a CRC-passing frame is wrong, for any soft-output decoder (classical or neural), and for standard CRCs such as CRC-8 and CRC-16.",
    "To refine that probability with a learned risk classifier using only information available at the receiver.",
    "To give a ground-truth-free running estimate of delivered undetected errors.",
    "To adapt the code profile to the channel (thermal noise or bursts) without pilot symbols.",
  ], ref: "num" },

  { h1: "6. Working principle of the invention (in brief)" },
  { numbers: [
    "The transmitter sends a standard frame: payload plus an unmodified CRC (CRC-8 or CRC-16), protected by an FEC code (here a learned neural code or Hamming(7,4)).",
    "The receiver's soft-output decoder gives, for each byte (or code block) c of the frame, a probability distribution P_c(v) over its possible values. The hard decision x takes the most likely value of every byte.",
    "If the CRC of x fails, the frame is retransmitted as usual.",
    "If the CRC passes, the Soft-Syndrome Guard computes P(code) = 2^-r · Σ_u ∏_c F_c(s_c(u)), where F_c is the Walsh-Hadamard transform of P_c and s_c(u) is the part of the dual-code word u^T H that falls in byte c. The residual risk is R(y) = 1 − ∏_c P_c(x_c) / P(code).",
    "The frame is delivered only if R(y) ≤ τ, where τ is set so that a fixed share (for example 1 %) of correctly decoded frames is rejected. Otherwise a NAK requests retransmission.",
    "Optionally, a gradient-boosted classifier takes R(y), the minimum bit reliability, per-byte confidence, residual noise energy and a burst index, and outputs a refined probability of error that is used in the same way.",
  ], ref: "num2" },
  { p: "**Worked example.** Three bits with an even-parity rule, decoder confidences that each bit is 1 of 0.9, 0.8 and 0.3. The decision 110 passes the check and has probability 0.504. All parity-valid words together have probability 0.596, so R = 1 − 0.504/0.596 = 15.4 %, and the frame is retransmitted. With confidences 0.99, 0.98 and 0.01 the risk is 0.05 % and the frame is accepted. Both frames pass the CRC, but only the second is safe." },

  { h1: "7. Description of the invention in detail" },
  { img: R("architecture.png"), width: 600, ratio: 0.43, caption: "Figure 1 - Block diagram: transmitter (top), receiver with Soft-Syndrome Guard (middle), adaptive loop (bottom)" },
  { h3: "7.1 Frame and CRC" },
  { p: `A frame carries 32 payload bits and an r-bit CRC: CRC-8 (x^8 + x^2 + x + 1, 40-bit frame) or CRC-16 (x^16 + x^12 + x^5 + 1, 48-bit frame). Because the CRC is linear, every valid frame b satisfies H b = 0 with H = [G^T | I_r], where row i of G is the CRC of the i-th unit vector. The minimum distance of the frame code is ${S.crc_dmin} for CRC-8 and ${S.crc16_dmin} for CRC-16.` },
  { h3: "7.2 Soft-output decoders" },
  { bullets: [
    "**Neural code:** each byte is mapped by a learned table to 14 real symbols (70 symbols per CRC-8 frame, 84 per CRC-16 frame). A 3-layer neural network sees all received symbols and outputs a 256-way posterior for every byte.",
    "**Hamming(7,4):** exact soft decoding gives a 16-way posterior for every 4-bit block, which shows that the guard is independent of the decoder.",
  ] },
  { h3: "7.3 Soft-Syndrome Guard unit" },
  { p: "For every CRC-passing frame the unit transforms each group posterior with a 2^m-point Walsh-Hadamard transform, gathers the coefficient selected by each of the 2^r dual-code words, multiplies across groups and averages. A table of the gather indices is precomputed once per CRC and frame length. CRC-8 needs 256 terms per frame. CRC-16 needs 65,536 terms; a compiled multi-core kernel computes this at about 7,000 frames per second on a laptop CPU." },
  { h3: "7.4 Acceptance and ARQ" },
  { p: "A frame is delivered only if the CRC passes and R(y) ≤ τ. τ is calibrated at a reference operating point so that a fixed fraction (1 %) of correctly decoded frames is rejected, which fixes the retransmission overhead and makes comparisons with other gates fair." },
  { h3: "7.5 Learned risk gate (GuardCode-ML)" },
  { p: "A gradient-boosted decision-tree classifier (scikit-learn HistGradientBoostingClassifier) is trained on 6 million simulated frames per channel profile (using the frames that pass the CRC) to predict whether such a frame is wrong. Its inputs are the exact risk R(y), the minimum |LLR|, the sorted per-byte confidences and runner-up gaps, the residual noise energy after re-encoding the decision, and the burst index. It is calibrated to the same 1 % budget." },
  { h3: "7.6 Risk ledger and channel-adaptive profiles" },
  { p: "Summing R(y) over delivered frames estimates how many undetected errors were delivered, without knowing the transmitted data. Two code profiles (thermal noise and bursts) are kept. The receiver re-encodes its decision, measures residual noise energy in windows of 7 symbols, smooths the ratio of the largest to the median window energy, and switches profile with hysteresis; the choice is fed back as 1 bit on the ACK." },
  { h3: "7.7 Implementation" },
  { p: "The complete system is implemented in Python: a link simulator, training code (PyTorch), numpy inference, a SQLite database that logs every frame, and a live Flask web dashboard with a CRC-8/CRC-16 switch and AWGN, burst, fading and measured 802.15.4 channels. The source code is in the project repository." },

  { h1: "8. Experimental validation results" },
  { p: `Monte-Carlo simulation used ${n(S.bench_frames)} frames per operating point, Eb/N0 = 0-7 dB, and every gate rejected the same 1 % of correctly decoded frames. Besides simulated thermal-noise (AWGN) and Gilbert-Elliott burst channels, the receiver was validated on a channel fitted to **real measured error traces**: ${CW.n_traces} traces (${n(CW.frames)} frames) from IEEE 802.15.4 MicaZ radios in the public CRAWDAD niit/bit_errors dataset (IEEE DataPort, DOI 10.15783/C7S88K). Nothing was re-tuned for this channel.` },
  { table: resultsTable, widths: [4, 2, 2, 3], caption: "Table 2 - Undetected frames delivered with CRC-8, pooled over 0-7 dB (lower is better)" },
  { bullets: [
    `**Soft-Syndrome Guard vs the classic confidence test:** ${pct(RD.awgn.guard_vs_llr)} fewer undetected errors on AWGN, ${pct(RD.burst.guard_vs_llr)} on the burst channel and ${pct(RD.crawdad.guard_vs_llr)} on the measured 802.15.4 channel.`,
    `**Learned gate:** a further ${pct(RD.awgn.ml_vs_guard)} (AWGN), ${pct(RD.burst.ml_vs_guard)} (burst) and ${pct(RD.crawdad.ml_vs_guard)} (measured channel) fewer than the guard alone.`,
    `**Decoder-agnostic:** applied unchanged to Hamming(7,4) the guard removed ${pct(RD.awgn.hamssg_vs_ham)} (AWGN), ${pct(RD.burst.hamssg_vs_ham)} (burst) and ${pct(RD.crawdad.hamssg_vs_ham)} (measured) of its undetected errors.`,
    "**Cost:** ARQ goodput from 3 dB upward stays within about 1 % of the unguarded receiver, because all gates spend the same retransmission budget.",
    `**CRC-16:** CRC-16 alone removes about 99 % of CRC-8's undetected errors; the guard still lowered the remaining single-digit counts (burst ${P16.burst["Neural+CRC"].ue} to ${P16.burst.GuardCode.ue}, measured channel ${P16.crawdad["Neural+CRC"].ue} to ${P16.crawdad.GuardCode.ue}). The 8 extra CRC bits cost throughput (${P16.awgn.GuardCode.goodput_3db.toFixed(2)} vs ${P.awgn.GuardCode.goodput_3db.toFixed(2)} information bits per channel use on AWGN from 3 dB up).`,
    "**Accuracy of the formula:** checked against Monte-Carlo estimates (0.003671 analytical vs 0.003663 simulated over 400,000 trials).",
  ] },
  { img: R("benchmark_awgn.png"), width: 520, ratio: 0.73, caption: "Figure 2 - AWGN channel: bit, frame and undetected frame error rates and ARQ goodput for all receivers" },
  { img: R("crawdad_traces.png"), width: 600, ratio: 0.29, caption: "Figure 3 - The 22 measured CRAWDAD 802.15.4 traces and the fitted burst model" },

  { h1: "9. What aspect(s) of the invention need(s) protection?" },
  { numbers: [
    "The receiver method of computing, for a frame that passes an unmodified linear error-detecting code (such as a standard CRC), the exact posterior probability that the frame is nevertheless wrong, from the decoder's grouped soft output, by a dual-code expansion in which each group contributes the Walsh-Hadamard transform of its posterior.",
    "Using that residual risk as an ARQ acceptance gate whose threshold is calibrated to a fixed retransmission budget.",
    "The risk ledger: accumulating the residual risk over delivered frames to estimate delivered undetected errors without ground truth, and reporting it or using it to adjust the threshold.",
    "A learned risk gate that takes the exact residual risk together with other receiver-side features (minimum bit reliability, per-group confidence, residual noise energy, burst index) and outputs a refined acceptance probability.",
    "Pilot-free channel-adaptive selection of learned code profiles from a residual-energy burst index, fed back as one bit on the acknowledgement, with a calibrated threshold per profile.",
    "The receiver apparatus (decoder, check unit, soft-syndrome guard unit, acceptance unit) and software implementing the above.",
  ], ref: "num3" },
  { p: "A draft provisional specification with claims (Form 2, Indian Patent Office) is attached separately as GuardCode_Provisional_Patent_Draft.docx." },

  { h1: "10. Technology readiness level" },
  { table: [["TRL", "Stage", "Description", "Selected"],
    ...[["TRL 1", "Research", "Basic principles observed"], ["TRL 2", "Research", "Technology concept formulated"],
      ["TRL 3", "Research", "Experimental proof of concept"], ["TRL 4", "Development", "Technology validated in a lab"],
      ["TRL 5", "Development", "Technology validated in a relevant environment"], ["TRL 6", "Development", "Technology demonstrated in a relevant environment"],
      ["TRL 7", "Deployment", "System prototype demonstration in an operational environment"], ["TRL 8", "Deployment", "System complete and qualified"],
      ["TRL 9", "Deployment", "Actual system proven in an operational environment"]].map((r) => [...r, r[0] === "TRL 3" ? "\u2714 Selected" : ""])],
    widths: [1, 2, 5, 2] },
  { p: "**TRL 3 - experimental proof of concept.** The method is fully implemented in software and validated by large-scale simulation, including a channel built from real measured 802.15.4 error traces. It has not yet run on radio hardware. Implementing the guard on a software-defined-radio (SDR) testbed would move it to TRL 4." },
  { space: 200 },
  { center: "----------------------END OF THE DOCUMENT-----------------------------", size: 20 },
];

Promise.all([
  build(idfA, path.join(ROOT, "docs", "IDF_A_GuardCode.docx")),
  build(idfB, path.join(ROOT, "docs", "IDF_B_GuardCode.docx")),
]);
