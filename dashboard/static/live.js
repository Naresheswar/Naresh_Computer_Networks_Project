const COLORS = {
  "Uncoded+CRC": "#98a2b3",
  "Hamming+IL+CRC": "#1f77b4",
  "Neural+CRC": "#f79009",
  "GuardCode": "#d92d20",
  "GuardCode-ML": "#7a5af8",
  "GuardCode-Adaptive": "#079455",
  "GuardCode-ML-Adaptive": "#0e9384",
};
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x === 0 ? "0" : x < 1e-3 ? x.toExponential(2) : x.toFixed(d));
const MAXPTS = 240;

function lineChart(id, ylabel, logY = false) {
  return new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      animation: false, responsive: true, maintainAspectRatio: true, aspectRatio: 1.9,
      elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.15 } },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "time (s)" }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: true, text: ylabel }, type: logY ? "logarithmic" : "linear", beginAtZero: !logY },
      },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } },
    },
  });
}
const charts = { ue: lineChart("cUE", "undetected frames"), fer: lineChart("cFER", "FER"), gp: lineChart("cGP", "info bits / channel use") };
const cB = new Chart($("cB"), {
  type: "line",
  data: { labels: [], datasets: [
    { label: "burst index (EWMA)", data: [], borderColor: "#344054", borderWidth: 2, pointRadius: 0 },
    { label: "switch-to-burst threshold", data: [], borderColor: "#d92d20", borderDash: [5, 4], borderWidth: 1, pointRadius: 0 },
  ] },
  options: { animation: false, aspectRatio: 1.9, scales: { x: { title: { display: true, text: "time (s)" }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "log10(max/median residual energy)" } } },
    plugins: { legend: { labels: { boxWidth: 10, font: { size: 11 } } } } },
});

function ensureSets(names) {
  for (const key of ["ue", "fer", "gp"]) {
    const c = charts[key];
    if (c.data.datasets.length) continue;
    c.data.datasets = names.map((n) => ({ label: n, data: [], borderColor: COLORS[n] || "#000", backgroundColor: COLORS[n] || "#000" }));
  }
}

function pushPoint(p, names) {
  ensureSets(names);
  for (const [key, field] of [["ue", "undetected"], ["fer", "fer"], ["gp", "goodput"]]) {
    const c = charts[key];
    c.data.labels.push(p.t);
    c.data.datasets.forEach((ds) => ds.data.push(p[field][ds.label]));
    if (c.data.labels.length > MAXPTS) { c.data.labels.shift(); c.data.datasets.forEach((ds) => ds.data.shift()); }
  }
  cB.data.labels.push(p.t);
  cB.data.datasets[0].data.push(p.burst_index);
  cB.data.datasets[1].data.push(0.85);
  if (cB.data.labels.length > MAXPTS) { cB.data.labels.shift(); cB.data.datasets.forEach((d) => d.data.shift()); }
}

let lastT = null, frame = 0;
function render(s) {
  $("ebnoVal").textContent = s.ebno.toFixed(2);
  if (document.activeElement !== $("ebno")) $("ebno").value = s.ebno;
  document.querySelectorAll("#chanSeg button").forEach((b) => b.classList.toggle("on", b.dataset.ch === s.channel));
  const pill = $("runPill");
  pill.className = "pill" + (s.running ? (s.storm_left ? " storm" : "") : " off");
  pill.textContent = s.running ? (s.storm_left ? `burst storm ${s.storm_left}s` : "running") : "paused";
  $("pause").textContent = s.running ? "Pause" : "Resume";
  $("effChan").textContent = s.effective_channel.toUpperCase();
  $("profile").textContent = s.adaptive.profile + ` (${s.adaptive.switches} switches)`;
  $("bidx").textContent = s.adaptive.ewma ?? "-";
  $("dbRows").textContent = s.db.rows.toLocaleString();
  $("dbRate").textContent = Math.round(s.db.insert_rate).toLocaleString() + " rows/s";
  $("dbSize").textContent = s.db.size_mb + " MB";

  const tb = $("kpi").querySelector("tbody");
  tb.innerHTML = s.names.map((n) => {
    const k = s.kpis[n];
    const g = n.startsWith("GuardCode") ? "guard" : "";
    const uc = k.undetected === 0 ? "zero" : "bad";
    return `<tr class="${g}"><td><span class="swatch" style="background:${COLORS[n]}"></span>${n}</td>
      <td>${k.frames.toLocaleString()}</td><td>${fmt(k.fer)}</td><td>${fmt(k.ber, 4)}</td>
      <td>${k.detected.toLocaleString()}</td><td class="${uc}">${k.undetected.toLocaleString()}</td>
      <td class="${uc}">${(100 * k.undetected_share).toFixed(3)}%</td><td>${k.goodput.toFixed(3)}</td></tr>`;
  }).join("");

  if (s.point && s.point.t !== lastT) {
    lastT = s.point.t;
    pushPoint(s.point, s.names);
    if (++frame % 2 === 0) { Object.values(charts).forEach((c) => c.update()); cB.update(); }
  }
  const ev = $("events");
  ev.innerHTML = s.events.length ? s.events.map((e) =>
    `<li><b style="color:${COLORS[e.scheme]}">${e.scheme}</b> &middot; ${new Date(e.ts * 1000).toLocaleTimeString()} &middot; ${e.count} frame(s) &middot; ${e.bit_errors} corrupted payload bits &middot; ${e.channel.toUpperCase()} @ ${e.ebno} dB</li>`).join("")
    : '<li class="muted">none yet</li>';
}

function post(body) {
  return fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then((r) => r.json()).then(render);
}
$("ebno").addEventListener("input", (e) => ($("ebnoVal").textContent = (+e.target.value).toFixed(2)));
$("ebno").addEventListener("change", (e) => post({ ebno: +e.target.value }));
document.querySelectorAll("#chanSeg button").forEach((b) => b.addEventListener("click", () => post({ channel: b.dataset.ch })));
$("rate").addEventListener("change", (e) => post({ rate: +e.target.value }));
$("storm").addEventListener("click", () => post({ storm: true }));
$("pause").addEventListener("click", () => post({ paused: $("pause").textContent === "Pause" }));
$("reset").addEventListener("click", () => {
  for (const c of [...Object.values(charts), cB]) { c.data.labels = []; c.data.datasets.forEach((d) => (d.data = [])); c.update(); }
  post({ reset: true });
});

fetch("/api/history").then((r) => r.json()).then((h) => {
  fetch("/api/snapshot").then((r) => r.json()).then((s) => {
    h.forEach((p) => pushPoint(p, s.names));
    render(s);
    Object.values(charts).forEach((c) => c.update()); cB.update();
    const es = new EventSource("/stream");
    es.onmessage = (m) => render(JSON.parse(m.data));
  });
});

// keep explanatory text in sync even if an older cached template is served
(() => {
  const tag = document.querySelector(".tag");
  if (tag) tag.innerHTML = "Soft-Syndrome Guard on a CRC-8 protected neural FEC link &middot; 32-bit payload &middot; every frame logged to SQLite";
  const note = document.querySelector(".note");
  if (note) note.innerHTML = "An <b>undetected</b> frame is decoded wrongly but still passes the receiver checks, so corrupted data reaches the application. GuardCode's Soft-Syndrome Guard computes, for every frame that passes CRC-8, the probability that it is still wrong, and sends risky frames back for retransmission &mdash; no extra bits, unmodified CRC.";
})();
