// Provisional patent specification draft (Indian Patent Office, Form 2 style).
const path = require("path");
const fs = require("fs");
const { build, AlignmentType } = require("./docgen");
const ROOT = path.resolve(__dirname, "..", "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "summary.json"), "utf8"));
const R = (f) => path.join(ROOT, "results", f);

const blocks = [
  { center: "FORM 2", bold: true, size: 26, after: 40 },
  { center: "THE PATENTS ACT, 1970 (39 of 1970)", after: 40 },
  { center: "and", after: 40 },
  { center: "THE PATENTS RULES, 2003", after: 240 },
  { center: "PROVISIONAL SPECIFICATION", bold: true, size: 26, after: 40 },
  { center: "(See section 10 and rule 13)", after: 360 },
  { center: "TITLE OF THE INVENTION", bold: true, after: 80 },
  { center: "SYSTEM AND METHOD FOR SUPPRESSING UNDETECTED FRAME ERRORS IN CYCLIC-REDUNDANCY-CHECK PROTECTED COMMUNICATION LINKS USING AN EXACT SOFT-SYNDROME POSTERIOR AND CHANNEL-ADAPTIVE LEARNED DECODING", bold: true, after: 360 },
  { center: "APPLICANT", bold: true, after: 80 },
  { center: "Naresh Nireenjan, Indian national, B.Tech (Computer Science and Engineering), School of Computer Science and Engineering, Vellore Institute of Technology, Vellore - 632014, Tamil Nadu, India", after: 120 },
  { center: "[Co-applicant / assignee details to be completed with the VIT IPR cell]", italics: true, after: 360 },
  { center: "PREAMBLE TO THE DESCRIPTION", bold: true, after: 80 },
  { center: "The following specification describes the invention.", after: 120 },
  { center: "[DRAFT prepared for review by a registered patent agent before filing. Not legal advice.]", italics: true, after: 120 },
  { pagebreak: true },

  { h1: "FIELD OF THE INVENTION" },
  { p: "The present invention relates to error control in digital communication networks, and more particularly to a receiver-side method and system that estimates, for every received frame that passes a cyclic redundancy check (CRC), the exact posterior probability that the frame is nevertheless erroneous, and uses that probability to trigger automatic repeat request (ARQ) retransmissions, to account for residual errors delivered to higher layers, and to adapt a learned channel code to the prevailing channel condition, without modifying the CRC polynomial or the frame format and without transmitting additional redundancy." },

  { h1: "BACKGROUND OF THE INVENTION" },
  { p: "Data-link and physical-layer protocols (for example HDLC, IEEE 802.3, IEEE 802.11, Bluetooth, LTE and 5G NR) append a CRC to every frame. The receiver first corrects errors with a forward error correction (FEC) decoder and then recomputes the CRC; a frame whose CRC matches is delivered, otherwise a negative acknowledgement requests retransmission. A frame that is wrong but still satisfies the CRC is an **undetected error**: corrupted data is silently delivered to the application. For an r-bit CRC the share of decoding errors that escape detection is of the order of 2^-r, which is significant for the short CRCs used in low-latency and IoT links (r = 8 to 16) and for safety-related machine-type communication." },
  { p: "Known approaches to lower the undetected-error probability include: (a) lengthening the CRC, which costs payload efficiency on every frame; (b) designing the CRC polynomial jointly with a particular inner code (for example convolutional-code-specific CRC design), which requires changing standardised polynomials; (c) CRC-aided list decoding, in which the CRC is used to select among decoder candidates (for example US 2023/0198672 A1, autoencoders with list decoding, where the CRC is appended to a neural code and checked on list candidates at test time); and (d) reliability or threshold tests on the decoder output, such as Forney-type erasure decoding, the reliability-output Viterbi algorithm (ROVA) of Raghavan and Baum, and minimum-|LLR| or likelihood-ratio threshold tests applied before or after CRC checking. Separately, the probability that a single parity check is satisfied given bit probabilities is classical in low-density parity-check decoding (Gallager), where it is used inside iterative decoders rather than as an acceptance test for an outer error-detecting code." },
  { p: "Threshold tests of type (d) use a per-bit or per-path confidence that is blind to the structure of the outer CRC: they cannot distinguish between low-confidence bit patterns that the CRC would certainly catch and those that form a CRC codeword. Approaches (b) and (c) use the CRC structure but either modify the code design or change the decoder. There is therefore a need for a decoder-agnostic receiver test that uses the exact algebraic structure of a standard, unmodified CRC together with soft decoder output to estimate the probability that a CRC-passing frame is wrong." },
  { p: "Learned (neural-network) channel codes such as Turbo Autoencoder and deep joint source-channel coding provide soft outputs but are typically trained for a single channel model; when the channel changes, for example from thermal noise to bursty interference, their posteriors become mis-calibrated, which makes any reliability test less trustworthy." },

  { h1: "OBJECTS OF THE INVENTION" },
  { numbers: [
    "To provide a method for computing, for each received frame, the exact posterior probability that a frame which passes a standard CRC is erroneous, using the decoder's per-bit soft output and the parity-check structure of the unmodified CRC.",
    "To reduce the rate of undetected frame errors delivered to higher layers for a given retransmission budget, without lengthening the CRC and without transmitting additional bits.",
    "To provide a receiver test that works with any soft-output decoder, including classical block-code decoders and neural-network decoders.",
    "To provide a running, ground-truth-free estimate of the number of undetected errors delivered (a risk ledger) that can be reported to higher layers or used to tune the acceptance threshold.",
    "To provide channel-adaptive selection among a plurality of learned encoder/decoder profiles, driven by residual statistics obtained by re-encoding decoded frames, with the selection fed back in the acknowledgement.",
  ], ref: "num" },

  { h1: "SUMMARY OF THE INVENTION" },
  { p: "A frame of k payload bits is extended with an r-bit CRC computed with a generator polynomial g(x), giving n = k + r frame bits, and transmitted after channel encoding. Because a CRC with zero initial value is a linear code, every valid frame b satisfies H b = 0 (mod 2), where H = [G^T | I_r] and the i-th row of G is the CRC of the i-th unit vector. A soft-output decoder produces, for each group c of m consecutive frame bits (m = 1 for bit-wise decoders, m = 4 for a Hamming(7,4) block, m = 8 for a byte-wise neural decoder), a posterior distribution P_c(v) over the 2^m values of the group. The hard decision x takes the most probable value of each group and is CRC-checked in the usual way." },
  { p: "For a frame that passes the CRC, the invention computes the CRC-conditioned posterior probability that the decision is correct," },
  { eq: "P(b = x | y, Hb = 0)  =  ∏_c P_c(x_c)  /  P(Hb = 0 | y)" },
  { p: "where the normaliser is evaluated exactly by a dual-code expansion over all 2^r vectors u, in which every group contributes one coefficient of the Walsh-Hadamard transform of its posterior," },
  { eq: "P(Hb = 0 | y)  =  2^(-r) · Σ_u  ∏_c F_c(s_c(u)),     F_c(s) = Σ_v P_c(v) (−1)^popcount(s AND v)" },
  { p: "with s_c(u) the m bits of u^T H that fall in group c. For m = 1 this reduces to the product of (1 − 2 p_i)^((u^T H)_i)." },
  { p: "and defines the residual risk R(y) = 1 − P(b = x | y, Hb = 0). The frame is accepted only if the CRC passes and R(y) ≤ τ; otherwise it is discarded and a retransmission is requested. The threshold τ is set from a retransmission budget (the fraction of correctly decoded frames that may be rejected) or from a target residual error rate. The sum of R(y) over frames provides an estimate, under the decoder's posterior, of the number of undetected errors delivered; its accuracy depends on the calibration of the decoder output." },
  { p: "In a preferred embodiment the channel code is a learned code: each m-bit group of the frame is mapped by a trained table to a point in d-dimensional real space and a neural network decoder that observes the whole frame outputs 2^m-ary posteriors per group, from which the bit posteriors are obtained by marginalisation. A plurality of such profiles is trained for different channel classes; the receiver re-encodes each decoded frame, computes the residual energy in windows of w symbols, forms a burst index b = log10(max window energy / median window energy), smooths it with an exponentially weighted moving average and selects a profile with hysteresis; the selected profile and its own threshold τ_k are signalled to the transmitter in the acknowledgement." },

  { h1: "BRIEF DESCRIPTION OF THE DRAWINGS" },
  { bullets: [
    "**Figure 1** is a block diagram of the system: transmitter (101-103), channel (104), soft-output decoder (105), standard CRC check (106), soft-syndrome guard (107), acceptance logic (108), delivery/ARQ unit (109), residual burst-index estimator (110), profile selector (111) and risk ledger (112).",
    "**Figure 2** shows the undetected-frame-error rate versus the fraction of correctly decoded frames rejected, for the soft-syndrome guard (group form), its bit-wise variant and a conventional minimum-|LLR| threshold test, on an AWGN channel.",
    "**Figure 3** shows the same comparison on a Gilbert-Elliott burst-noise channel.",
    "**Figure 4** shows the calibration of the residual risk R(y): predicted risk versus observed error frequency of CRC-passing frames.",
    "**Figure 5** shows the channel-adaptive profile selection over time when the channel switches between AWGN and burst noise.",
  ] },
  { img: R("architecture.png"), width: 600, ratio: 0.43, caption: "Figure 1 - System block diagram" },

  { h1: "DETAILED DESCRIPTION OF THE INVENTION" },
  { h2: "1. Frame format and CRC parity-check matrix" },
  { p: `In the embodiment used for the experiments, k = ${S.k} payload bits are protected by the standard CRC-8 with g(x) = x^8 + x^2 + x + 1 (r = 8), giving n = 40 frame bits. The generator rows G_i are obtained by running the conventional bit-serial CRC register on unit vectors, and H = [G^T | I_8]. At this frame length the CRC has minimum Hamming distance ${S.crc_dmin}, i.e. every pattern of fewer than ${S.crc_dmin} bit errors is detected. The matrix W whose rows are u^T H (mod 2) for all 256 vectors u is precomputed once per CRC and frame length (W is the codebook of the dual code).` },
  { h2: "2. Soft-syndrome guard (107)" },
  { p: "For every group c the guard computes the fast Walsh-Hadamard transform F_c of the group posterior P_c (2^m values, m · 2^m additions). An index table M of size 2^r x G, precomputed once per CRC and frame length, gives for every dual-code word u and group c the m-bit pattern s_c(u). The normaliser is the average over u of the products ∏_c F_c(M[u, c]); for CRC-8 and five byte groups this is 256 x 5 multiplications per frame, negligible compared with the decoder. The numerator is ∏_c P_c(x_c), and R(y) follows directly. The computation is exact whenever the groups are conditionally independent given y. Correlations inside a group (for example between the bits of one byte or of one Hamming block) are preserved exactly, which a bit-wise computation cannot do." },
  { p: "Correctness was verified numerically: (i) for independent bits, the formula and a 400 000-trial Monte-Carlo estimate agree (0.003671 vs 0.003663); (ii) for correlated random byte posteriors around a CRC-valid decision, the exact risk 0.3236 matches a 1 000 000-trial Monte-Carlo estimate of 0.3246 (three further cases agree within Monte-Carlo error). A bit-wise variant that ignores within-group correlation is markedly less effective (Figures 2 and 3)." },
  { h2: "3. Acceptance logic (108) and threshold selection" },
  { p: "A frame is delivered if (i) the CRC computed on the hard decision is zero and (ii) R(y) ≤ τ. The threshold is calibrated off-line or on-line so that a chosen fraction ε of correctly decoded frames (for example ε = 1 %) is rejected at a reference operating point; this fixes the extra retransmission cost. Alternatively, when the decoder output is well calibrated (as for the classical decoder in Figure 4), τ may be set directly from a target residual error probability, because R(y) is then a calibrated probability." },
  { h2: "4. Risk ledger (112)" },
  { p: "The receiver accumulates L = Σ R(y) over all accepted frames. L estimates the number of undetected errors that have been delivered, without any knowledge of the transmitted data. L may be reported to higher layers (for example to trigger an end-to-end integrity check), logged for network management, or used in a feedback loop that lowers τ when L grows faster than a target rate." },
  { h2: "5. Learned channel code and decoder (103, 105)" },
  { p: "The 40-bit frame is divided into C = 5 bytes. A trained table of 256 x 14 real values maps each byte to 14 channel symbols; the 70 symbols are normalised to unit average energy and optionally interleaved. The decoder is a fully connected network (70 inputs, three hidden layers of 512 ELU units, 5 x 256 outputs) producing per-byte log-posteriors; bit posteriors are the marginals. Encoder and decoder are trained end-to-end through a differentiable channel model at Eb/N0 drawn uniformly from 1 to 6 dB with a cross-entropy objective. For deployment the weights are exported and inference runs without any deep-learning runtime." },
  { h2: "6. Channel-adaptive profile selection (110, 111)" },
  { p: "Profiles are trained for an additive white Gaussian noise channel and for a Gilbert-Elliott burst channel. After decoding, the receiver re-encodes the decoded bytes with the active profile, forms residuals r = y − x_hat, computes the mean residual energy in windows of 7 symbols, and the burst index b = log10(max/median). An EWMA of b (factor 0.02) is compared with hysteresis thresholds (0.44 up, 0.40 down) to select the profile, which is signalled in the next acknowledgement (one bit for two profiles). Each profile carries its own calibrated threshold τ_k, so the guard remains calibrated after the switch." },
  { h2: "7. Experimental example" },
  { p: `The system was simulated with ${S.bench_frames.toLocaleString("en-IN")} frames per operating point (Eb/N0 = 0 to 7 dB) on AWGN and burst channels, and every frame of a further bulk run and of live operation was logged to a database. All gates were calibrated to reject 1 % of correctly decoded frames at 3 dB, so that differences in undetected errors are not obtained by rejecting more frames.` },
  { table: S.patent_table, widths: [3, 2, 2, 2, 2], caption: "Table 1 - Undetected frame errors pooled over Eb/N0 = 0-7 dB (same retransmission budget for all gates)" },
  { p: S.patent_text },
  { img: R("tradeoff_awgn.png"), width: 430, ratio: 0.71, caption: "Figure 2 - Undetected error rate vs. retransmission cost, AWGN" },
  { img: R("tradeoff_burst.png"), width: 430, ratio: 0.71, caption: "Figure 3 - Undetected error rate vs. retransmission cost, burst channel" },
  { img: R("calibration.png"), width: 380, ratio: 0.83, caption: "Figure 4 - Calibration of the residual risk R(y)" },
  { img: R("adaptive.png"), width: 520, ratio: 0.67, caption: "Figure 5 - Channel-adaptive profile selection" },
  { h2: "8. Variants" },
  { bullets: [
    "Any linear error-detecting code (CRC-16, CRC-24, CRC-32, checksums expressible over GF(2)) may be used; for r > 16 the sum over 2^r vectors may be evaluated by a fast Walsh-Hadamard transform over the syndrome space or restricted to the dominant terms.",
    "The soft output may come from any decoder: Hamming, BCH, convolutional (BCJR), LDPC, polar, turbo, or neural decoders, including outputs of HARQ soft-combining.",
    "The guard may be applied per code block in multi-block transport blocks, and the per-block risks combined.",
    "The acceptance rule may grade the response: deliver, deliver-with-flag, or retransmit, using two thresholds.",
  ] },

  { h1: "CLAIMS" },
  { p: "[In a provisional application claims are optional; the following draft claims are included to define the scope for the complete specification.]", style: { italics: true } },
  { p: "We claim:" },
  { numbers: [
    "A method for suppressing undetected frame errors at a receiver of a communication link, the method comprising: receiving a channel output y for a frame comprising payload bits and r check bits generated by a linear error-detecting code having parity-check matrix H; obtaining, from a soft-output decoder, for each of a plurality of groups of one or more frame bits, a posterior distribution P_c over the values of the group; forming a hard decision x by selecting the most probable value of each group and checking H x = 0; for a frame satisfying the check, computing a residual risk R(y) = 1 − ∏_c P_c(x_c) / P(Hb = 0 | y), wherein P(Hb = 0 | y) is computed as the average, over all 2^r vectors u, of the product over groups of the Walsh-Hadamard transform coefficient of P_c at the pattern formed by the bits of u^T H falling in group c; and accepting the frame only if R(y) does not exceed a threshold τ, and otherwise discarding the frame and requesting its retransmission; wherein the linear error-detecting code and the frame format are not modified and no additional redundancy is transmitted.",
    "The method of claim 1, wherein the linear error-detecting code is a cyclic redundancy check with a standardised generator polynomial, and the rows of H are obtained by computing the cyclic redundancy check of unit vectors.",
    "The method of claim 1, wherein the groups are the bytes of the frame or the message blocks of an inner block code, so that statistical dependence between bits of the same group is preserved exactly, and wherein a table giving, for every vector u and group c, the index of the corresponding transform coefficient is precomputed once per error-detecting code and frame length.",
    "The method of claim 1, wherein τ is selected such that a predetermined fraction of correctly decoded frames is rejected at a reference operating point, thereby fixing the retransmission overhead.",
    "The method of claim 1, further comprising accumulating R(y) over accepted frames to obtain an estimate of the number of undetected errors delivered, and reporting said estimate to a higher protocol layer or adjusting τ in dependence on said estimate.",
    "The method of claim 1, wherein the soft-output decoder is a neural network that receives all channel symbols of the frame and outputs, for each group of m frame bits, a posterior distribution over 2^m values, and the p_i are obtained by marginalising said distributions.",
    "The method of claim 6, wherein the transmitter maps each group of m frame bits to a point in a d-dimensional real space by a table learned jointly with the neural network decoder.",
    "The method of claim 6, further comprising: re-encoding the decoded frame with the active encoder profile; computing residual energies of the received symbols with respect to the re-encoded symbols over windows of consecutive symbols; deriving a burst index from the ratio of the largest to the median window energy; smoothing the burst index over frames; selecting, with hysteresis, one of a plurality of encoder/decoder profiles each trained for a different channel class and each having its own threshold τ_k; and signalling the selected profile to the transmitter in an acknowledgement message.",
    "The method of claim 1, wherein the soft-output decoder is a classical soft-decision decoder of a block, convolutional, LDPC, polar or turbo code.",
    "A receiver for a communication link comprising a soft-output decoder, a check unit for a linear error-detecting code, a soft-syndrome guard unit configured to compute the residual risk R(y) as defined in claim 1, and an acceptance unit configured to deliver a frame only if the check is satisfied and R(y) does not exceed τ and otherwise to issue a retransmission request.",
    "The receiver of claim 10, further comprising a risk-ledger unit configured to accumulate R(y) over delivered frames, and a profile-selection unit configured as in claim 8.",
    "A non-transitory computer-readable medium storing instructions which, when executed by a processor, cause the processor to perform the method of any one of claims 1 to 9.",
  ], ref: "claims" },

  { pagebreak: true },
  { h1: "ABSTRACT" },
  { p: "A receiver method and system suppress undetected frame errors on links protected by a standard cyclic redundancy check (CRC). A soft-output decoder (neural or classical) gives posteriors over groups of frame bits; for each frame passing the CRC, a soft-syndrome guard computes the exact probability that the frame is still wrong, by dividing the posterior of the hard decision by the probability that the true frame is a CRC codeword, the latter evaluated exactly through a dual-code expansion of the unmodified CRC parity-check matrix using Walsh-Hadamard transforms of the group posteriors. Frames whose residual risk exceeds a threshold are retransmitted. The accumulated risk of delivered frames estimates the undetected errors delivered without ground truth. In an embodiment, learned encoder/decoder profiles are selected per channel class from burst statistics of re-encoded residuals, signalled in the acknowledgement. At an equal retransmission budget the guard delivers fewer undetected errors than conventional confidence-threshold tests, without extra bits or changes to the CRC." },
  { p: "(Figure 1)", align: AlignmentType.CENTER },
];

build(blocks, path.join(ROOT, "docs", "GuardCode_Provisional_Patent_Draft.docx"));
