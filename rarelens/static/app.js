/* RareLens front end — plain JS, no external libraries (runs fully offline). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const f2 = (x) => (x == null ? "—" : Number(x).toFixed(2));
const f3 = (x) => (x == null ? "—" : Number(x).toFixed(3));
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
const sd = (a) => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const SHORT = { baseline_no_aug: "Baseline", oversample_only: "Oversample", classic_aug: "Classic", smote_interp: "SMOTE", gan_aug: "GAN" };
const ORDER = ["baseline_no_aug", "oversample_only", "classic_aug", "smote_interp", "gan_aug"];

const S = { showGuess: false, meta: null, ev: null, pred: null, focus: "DF", thr: null, heatMode: true, example: null,
            gen: { rare: "DF", seed: 1042, sel: [0, 5], next: 0, checks: null, pair: 0 }, evRare: "DF" };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (_) {} throw new Error(m); }
  return r.json();
}

/* ------------------------------------------------------------------ tooltip */
const tip = $("#tooltip");
function bindTips(root) {
  $$("[data-tip]", root).forEach((el) => {
    el.addEventListener("mouseenter", () => { tip.innerHTML = el.dataset.tip; tip.classList.add("show"); });
    el.addEventListener("mousemove", (e) => {
      const w = tip.offsetWidth, h = tip.offsetHeight;
      tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 8) + "px";
      tip.style.top = Math.max(8, e.clientY - h - 12) + "px";
    });
    el.addEventListener("mouseleave", () => tip.classList.remove("show"));
    el.addEventListener("click", () => tip.classList.remove("show"));
  });
}

/* ------------------------------------------------------------------ routing */
function route() {
  const tab = (location.hash || "#examine").slice(1);
  const ok = ["examine", "synthesize", "evidence", "method"].includes(tab) ? tab : "examine";
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${ok}`));
  $$(".tabs a").forEach((a) => { if (a.dataset.tab === ok) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  tip.classList.remove("show");
  if (ok === "synthesize" && !S.gen.loaded) generate();
  window.scrollTo(0, 0);
}

/* ================================================================== EXAMINE */
function polar(c, r, deg) { const t = (deg * Math.PI) / 180; return [c + r * Math.sin(t), c - r * Math.cos(t)]; }
function arcPath(c, r, a0, a1) {
  const [x0, y0] = polar(c, r, a0), [x1, y1] = polar(c, r, a1);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
function drawRing(p, rare) {
  const c = 235, parts = [];
  for (let k = 0; k < 120; k++) {
    const big = k % 10 === 0, [x0, y0] = polar(c, big ? c - 12 : c - 6, k * 3), [x1, y1] = polar(c, c - 1, k * 3);
    parts.push(`<path d="M${x0.toFixed(1)} ${y0.toFixed(1)}L${x1.toFixed(1)} ${y1.toFixed(1)}" stroke="#A7A094" stroke-opacity="${big ? 0.9 : 0.45}" stroke-width="1"/>`);
  }
  const r = c - 24;
  parts.push(`<circle cx="${c}" cy="${c}" r="${r}" stroke="#1F2428" stroke-width="8" fill="none"/>`);
  if (p != null && p > 0.004) {
    const end = Math.min(359.9, 360 * p);
    parts.push(`<path d="${arcPath(c, r, 0, end)}" stroke="${rare ? "#E7A64A" : "#EDE8DE"}" stroke-width="8" stroke-linecap="round" fill="none"/>`);
  }
  parts.push(`<circle cx="${c}" cy="${c}" r="${r - 13}" stroke="#2B3136" stroke-width="1" fill="none"/>`);
  $("#ring").innerHTML = parts.join("");
}

function buildThumbs() {
  $("#thumbs").innerHTML = S.meta.examples.map((x) =>
    `<button type="button" data-id="${esc(x.id)}" aria-pressed="false" aria-label="${esc(x.name)} test image"
      data-tip="<b>${esc(x.name)}</b> · test set${x.note ? "<br>" + esc(x.note) : ""}"><img src="/${esc(x.src)}" alt=""></button>`).join("");
  $$("#thumbs button").forEach((b) => b.addEventListener("click", () => runExample(b.dataset.id)));
  bindTips($("#thumbs"));
}

function setBusy(on) { $("#lens-wrap").classList.toggle("busy", on); }

async function runExample(id) {
  const ex = S.meta.examples.find((x) => x.id === id);
  $$("#thumbs button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  S.example = ex;
  showImage(`/${ex.src}`, `Viewer · ${ex.id}`);
  setBusy(true);
  try { S.pred = await api(`/api/predict/example/${encodeURIComponent(id)}`); onPrediction(); }
  catch (e) { showError(e); } finally { setBusy(false); }
}

async function runUpload(file) {
  if (!file) return;
  $$("#thumbs button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  S.example = null;
  showImage(URL.createObjectURL(file), `Viewer · ${file.name}`);
  const fd = new FormData(); fd.append("file", file);
  setBusy(true);
  try { S.pred = await api("/api/predict", { method: "POST", body: fd }); onPrediction(); }
  catch (e) { showError(e); } finally { setBusy(false); }
}

function showImage(src, label) {
  const img = $("#lens-img"); img.src = src; img.hidden = false;
  $("#lens-empty").hidden = true; $("#lens-cam").hidden = true;
  $("#viewer-label").textContent = label;
}

function showError(e) {
  $("#flags").innerHTML = `<div class="flag">${esc(e.message || e)}</div>`;
}

function onPrediction() {
  const p = S.pred.probs;
  S.focus = (p.DF ?? 0) >= (p.VASC ?? 0) ? "DF" : "VASC";
  if (!S.meta.rares.includes(S.focus)) S.focus = S.meta.rares[0];
  S.thr = null; S.showGuess = false;
  const cam = $("#lens-cam"); cam.src = S.pred.cam; cam.hidden = false;
  applyHeat();
  renderReport();
}

function applyHeat() {
  const v = $("#heat").value / 100;
  $("#lens-cam").style.opacity = S.heatMode ? v : 0;
  $("#lens").classList.toggle("heat", S.heatMode && !!S.pred);
  $("#mode-heat").setAttribute("aria-pressed", String(S.heatMode));
  $("#mode-image").setAttribute("aria-pressed", String(!S.heatMode));
  $("#viewer-note").textContent = S.heatMode && S.pred
    ? "Warm areas influenced the decision most (Grad-CAM of the RareLens model, for the top class)." : "";
}

function decide(probs) {
  const codes = S.meta.classes;
  if (S.thr == null) return codes.reduce((a, c) => (probs[c] > probs[a] ? c : a), codes[0]);
  if (probs[S.focus] >= S.thr) return S.focus;
  return codes.filter((c) => c !== S.focus).reduce((a, c) => (probs[c] > probs[a] ? c : a), codes.find((c) => c !== S.focus));
}

function curvePoint(rare, t) {
  const cv = S.ev.diseases[rare]?.threshold_curve;
  if (!cv) return null;
  return cv.points.reduce((a, q) => (Math.abs(q.t - t) < Math.abs(a.t - t) ? q : a), cv.points[0]);
}

function renderReport() {
  const pr = S.pred; if (!pr) return;
  const probs = pr.probs, names = S.meta.class_names, counts = S.ev.dataset.counts, total = S.ev.dataset.images;
  const d = decide(probs), isRare = S.meta.rares.includes(d);
  const reject = (pr.flags || []).some((f) => f.kind === "unusual") && !S.showGuess;
  $("#report").classList.toggle("rejected", reject);
  if (reject) {
    $("#decision").textContent = "No reliable reading";
    $("#decision").style.color = "var(--l-ink)";
    $("#decision-sub").innerHTML = `This doesn't look like a dermoscopy image, so RareLens won't give a diagnosis. Its forced guess would be ${esc(names[d])} (${pct(probs[d])}) — the model has no "not skin" option. <button type="button" class="linkish" id="show-guess">Show the model's guess anyway</button>`;
    $("#show-guess").addEventListener("click", () => { S.showGuess = true; renderReport(); });
    $("#bigpct").textContent = "—"; $("#bigpct").style.color = "var(--l-mut)";
    $("#bigpct-sub").textContent = "out of scope";
    drawRing(null, false);
    $("#lens-label").textContent = "out of scope"; $("#lens-label").style.color = "var(--d-mut)";
    $("#truth").textContent = "";
  }
  if (!reject) {
  $("#decision").textContent = names[d];
  $("#decision").style.color = isRare ? "var(--l-amber)" : "var(--l-ink)";
  const share = (counts[d] / total) * 100;
  $("#decision-sub").textContent = `${isRare ? "Rare class" : "Common class"} — ${counts[d].toLocaleString()} of ${total.toLocaleString()} images in the dataset (${share.toFixed(share < 10 ? 2 : 1)} %)`;
  $("#bigpct").textContent = pct(probs[d]);
  $("#bigpct").style.color = isRare ? "var(--l-amber)" : "var(--l-ink)";
  $("#bigpct-sub").textContent = S.thr == null ? "model probability" : `probability · threshold ${S.thr.toFixed(2)}`;
  drawRing(probs[d], isRare);
  $("#lens-label").textContent = `${d} ${pct(probs[d])}`;
  $("#lens-label").style.color = isRare ? "var(--d-amber)" : "var(--d-txt)";
  if (pr.truth) {
    const ok = pr.truth === d;
    $("#truth").innerHTML = `Ground truth (test set): <b>${esc(names[pr.truth])}</b> — the model ${ok ? "agrees" : "disagrees"}.`;
  } else $("#truth").textContent = "Your image — no ground-truth label.";
  }

  // flags
  $("#flags").innerHTML = (pr.flags || []).map((f) => `<div class="flag" role="note">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 1.5 15 14H1L8 1.5Z" stroke="#A65A14" stroke-width="1.4"/><path d="M8 6v4M8 11.5v1" stroke="#A65A14" stroke-width="1.4"/></svg>
    <span>${esc(f.text)}</span></div>`).join("");

  // probability bars
  const sorted = [...S.meta.classes].sort((a, b) => probs[b] - probs[a]);
  $("#probs").innerHTML = sorted.map((c) => `<div class="prow ${c === S.focus ? "rare" : ""} ${c === d ? "decided" : ""}">
      <span class="nm">${esc(names[c])}</span><span class="cd">${c}</span>
      <span class="track"><span class="fill" style="width:${Math.max(0.5, probs[c] * 100)}%"></span></span>
      <span class="pv">${pct(probs[c])}</span></div>`).join("");

  // threshold card
  $$(".focus-code").forEach((e) => (e.textContent = S.focus));
  $("#thr-title").textContent = `Decision threshold · ${S.focus}`;
  const t = S.thr ?? 0.5;
  $("#thr-range").value = Math.round(t * 100);
  $("#thr-val").textContent = S.thr == null ? "off" : t.toFixed(2);
  const cv = S.ev.diseases[S.focus]?.threshold_curve;
  if (cv) {
    if (S.thr == null) {
      $("#thr-stat").innerHTML = `Default rule: highest probability wins. On the test set this model catches <b>${cv.default.caught} of ${cv.n_rare}</b> ${S.focus} cases with <b>${cv.default.false_alarms}</b> false alarms. Drag to try a threshold.`;
    } else {
      const q = curvePoint(S.focus, t);
      $("#thr-stat").innerHTML = `At ${t.toFixed(2)} on the test set: <b>${q.caught} of ${cv.n_rare}</b> ${S.focus} caught · <b>${q.false_alarms}</b> false alarms · F1 ${f2(q.f1)}. Lower catches more but raises more false alarms.`;
    }
  } else $("#thr-stat").textContent = "";
  $("#thr-reset").hidden = S.thr == null;

  // five methods
  $$("[data-focus]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.focus === S.focus)));
  $("#methods-sub").textContent = `probability of ${names[S.focus].toLowerCase()} (${S.focus})`;
  const rows = (pr.per_method[S.focus] || []).slice().sort((a, b) => ORDER.indexOf(a.config) - ORDER.indexOf(b.config));
  $("#methods").innerHTML = rows.map((m) => `<div class="mcard ${m.config === "gan_aug" ? "gan" : ""}">
      <span class="nm">${SHORT[m.config] || esc(m.label)}</span><span class="v">${pct(m.p_rare)}</span>
      <span class="t"><span style="width:${m.p_rare * 100}%"></span></span></div>`).join("") ||
    `<p class="foot">No ${S.focus} models found.</p>`;
}

function initExamine() {
  drawRing(null, false);
  buildThumbs();
  $("#file").addEventListener("change", (e) => runUpload(e.target.files[0]));
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => runUpload(e.dataTransfer.files[0]));
  $("#mode-heat").addEventListener("click", () => { S.heatMode = true; applyHeat(); });
  $("#mode-image").addEventListener("click", () => { S.heatMode = false; applyHeat(); });
  $("#heat").addEventListener("input", applyHeat);
  $("#thr-range").addEventListener("input", (e) => { S.thr = e.target.value / 100; renderReport(); });
  $("#thr-reset").addEventListener("click", () => { S.thr = null; renderReport(); });
  $$("[data-focus]").forEach((b) => b.addEventListener("click", () => { S.focus = b.dataset.focus; S.thr = null; renderReport(); }));
  $("#print").addEventListener("click", () => window.print());
  if (S.meta.examples.length) runExample(S.meta.examples[0].id);
}

/* ================================================================ SYNTHESIZE */
function buildDiseaseSeg(el, list, current, onPick) {
  el.innerHTML = list.map((r) => `<button type="button" data-r="${r}" aria-pressed="${r === current}">${r}</button>`).join("");
  $$("button", el).forEach((b) => b.addEventListener("click", () => {
    $$("button", el).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    onPick(b.dataset.r);
  }));
}

async function generate() {
  const g = S.gen;
  if (!S.meta.gan.length) { $("#grid-gen").innerHTML = `<p class="caption-dark">No generator found — run prepare_app.py.</p>`; return; }
  g.seed = Math.max(0, parseInt($("#seed").value, 10) || 0);
  $("#seed").value = g.seed;
  g.loaded = true;
  const btn = $("#generate"); btn.disabled = true; btn.textContent = "Generating…";
  try {
    const r = await api("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rare: g.rare, seed: g.seed, n: 12 }) });
    g.images = r.images; g.checks = r.checks; g.sel = [0, 5]; g.next = 0;
    renderTiles(); renderMorph(); renderGanFacts();
    $("#download").href = `/api/grid.png?rare=${g.rare}&seed=${g.seed}&n=12`;
  } catch (e) { $("#grid-gen").innerHTML = `<p class="caption-dark">${esc(e.message)}</p>`; }
  finally { btn.disabled = false; btn.textContent = "Generate 12"; }
}

function renderTiles() {
  const g = S.gen;
  $("#grid-gen").innerHTML = g.images.map((src, i) => {
    const c = g.checks?.[i];
    const chk = c ? `<span class="chk ${c.copy ? "copy" : ""}">${c.copy ? "near-copy" : "new"} · ${c.distance.toFixed(2)}</span>` : "";
    const ab = g.sel.indexOf(i) >= 0 ? `<span class="ab">${g.sel.indexOf(i) === 0 ? "A" : "B"}</span>` : "";
    return `<button type="button" class="tile ${g.sel.includes(i) ? "sel" : ""}" data-i="${i}" aria-label="Synthetic ${g.rare} image ${i + 1}; set as morph end">
      <img src="${src}" alt="">${ab}${chk}<span class="num">#${String(i + 1).padStart(3, "0")}</span></button>`;
  }).join("");
  $$("#grid-gen .tile").forEach((t) => t.addEventListener("click", () => {
    const i = +t.dataset.i;
    if (g.sel.includes(i)) return;
    g.sel[g.next] = i; g.next = 1 - g.next;
    renderTiles(); renderMorph();
  }));
}

let morphTimer = null;
async function morphAt(t) {
  const g = S.gen;
  return (await api("/api/morph", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rare: g.rare, seed: g.seed, n: 12, a: g.sel[0], b: g.sel[1], t }) })).image;
}
async function renderMorph() {
  const g = S.gen;
  $("#morph-label").textContent = `latent interpolation · #${String(g.sel[0] + 1).padStart(3, "0")} → #${String(g.sel[1] + 1).padStart(3, "0")}`;
  const ts = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1];
  const frames = await Promise.all(ts.map(morphAt));
  $("#strip").innerHTML = frames.map((src, k) => `<img src="${src}" alt="Morph frame ${k + 1} of 7" data-k="${k}">`).join("");
  updateMorphLive();
}
function updateMorphLive() {
  const t = $("#morph-range").value / 100;
  $$("#strip img").forEach((im) => im.classList.toggle("on", Math.round(t * 6) === +im.dataset.k));
  clearTimeout(morphTimer);
  morphTimer = setTimeout(async () => { try { $("#morph-img").src = await morphAt(t); } catch (_) {} }, 60);
}

function renderGanFacts() {
  const g = S.gen, info = S.ev.gan?.[g.rare];
  if (!info) { $("#gan-stats").innerHTML = ""; $("#pairs").innerHTML = ""; return; }
  $("#gen-n").textContent = info.train_images ?? "—";
  const pairs = info.pairs || [];
  g.pair = Math.min(g.pair, Math.max(0, pairs.length - 1));
  $("#pair-nav").innerHTML = pairs.map((_, i) => `<button type="button" aria-pressed="${i === g.pair}" aria-label="Example ${i + 1}">${i + 1}</button>`).join("");
  $$("#pair-nav button").forEach((b, i) => b.addEventListener("click", () => { g.pair = i; renderGanFacts(); }));
  const pp = pairs[g.pair];
  $("#pairs").innerHTML = pp ? `<figure class="gen"><img src="/${pp.gen}" alt="Generated ${g.rare} image"><figcaption>Generated</figcaption></figure>
      <figure><img src="/${pp.real}" alt="Its closest real training image"><figcaption>Closest real training image · ${esc(pp.real_id)}</figcaption></figure>` : "";
  $("#pair-text").innerHTML = pp
    ? `Distance <b>${f3(pp.distance)}</b> in Inception feature space. Every generated image is compared with all ${info.train_images} real training images; anything closer than <b>${f3(info.copy_threshold)}</b> (the closest 5 % of real-vs-real pairs) is thrown away as a copy.`
    : "";
  const ft = info.fid_table || [];
  const ref = ft.find((r) => /reference/i.test(r.set)), gan = ft.find((r) => /GAN/.test(r.set) && /filtered|final/i.test(r.set) && !/unfiltered/i.test(r.set));
  const cards = [];
  if (gan && ref) cards.push(["Realism · size-matched FID (lower = closer to real)", `${gan.fid}`, `vs ${ref.fid} for two sets of real ${g.rare} images`]);
  if (info.gen_real_median != null) cards.push(["Novelty · median distance to the closest real image", f3(info.gen_real_median), `vs ${f3(info.real_real_median)} between two different real lesions`]);
  const fc = info.filter_counts;
  if (fc) cards.push(["Filtering", `${(info.candidates || 2000).toLocaleString()} → ${fc.selected ?? 500}`, `${fc.copy ?? 0} near-copies and ${fc.outlier ?? 0} outliers removed`]);
  let live = "";
  if (g.checks) {
    const n = g.checks.length, fresh = g.checks.filter((c) => !c.copy).length;
    live = `<div class="statcard live"><span class="k">Live copy check · this batch (seed ${g.seed})</span><span class="v">${fresh} of ${n} new</span><span class="s">median distance ${f3(g.checks.map((c) => c.distance).sort((a, b) => a - b)[Math.floor(n / 2)])} · copy threshold ${f3(info.copy_threshold)}</span></div>`;
  }
  $("#gan-stats").innerHTML = live + cards.map(([k, v, s]) => `<div class="statcard"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span><span class="s">${esc(s)}</span></div>`).join("");
}

function initSynthesize() {
  const rares = S.meta.gan.length ? S.meta.gan : ["DF"];
  S.gen.rare = rares[0];
  buildDiseaseSeg($("#gen-disease"), rares, S.gen.rare, (r) => { S.gen.rare = r; S.gen.pair = 0; generate(); });
  $("#generate").addEventListener("click", generate);
  $("#seed").addEventListener("keydown", (e) => { if (e.key === "Enter") generate(); });
  $("#dice").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 100000); generate(); });
  $("#morph-range").addEventListener("input", updateMorphLive);
}

/* ================================================================== EVIDENCE */
function methodStats(m) {
  const s = m.seeds, pick = (k) => s.map((x) => x[k]).filter((v) => v != null);
  return { cfg: m.config, label: SHORT[m.config] || m.label, n: s.length,
    f1: pick("f1"), recall: pick("recall"), precision: pick("precision"), macro: pick("macro_f1"),
    fa: pick("false_alarms"), ap: pick("ap"), tuned: pick("tuned_f1"), tunedFa: pick("tuned_false_alarms"),
    caught: s.reduce((a, x) => a + x.caught, 0), support: s.reduce((a, x) => a + x.support, 0), paired: m.paired_f1 };
}

function barChart(stats) {
  const W = 440, H = 230, L = 34, R = 8, T = 18, B = 34, iw = W - L - R, ih = H - T - B, band = iw / stats.length;
  const y = (v) => T + ih * (1 - v);
  const out = [];
  [0, 0.25, 0.5, 0.75, 1].forEach((g) => out.push(`<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" stroke="#ECE7DD"/><text x="${L - 6}" y="${y(g) + 3.5}" text-anchor="end">${g.toFixed(2)}</text>`));
  const base = stats.find((s) => s.cfg === "baseline_no_aug");
  stats.forEach((s, i) => {
    const m = mean(s.f1), cx = L + band * i + band / 2, bw = band * 0.56, x0 = cx - bw / 2, top = y(m), r = 4;
    const col = s.cfg === "gan_aug" ? "#1F6F66" : "#B9B2A6";
    out.push(`<path d="M${x0} ${y(0)}V${top + r}Q${x0} ${top} ${x0 + r} ${top}H${x0 + bw - r}Q${x0 + bw} ${top} ${x0 + bw} ${top + r}V${y(0)}Z" fill="${col}"/>`);
    s.f1.forEach((v, k) => out.push(`<circle cx="${cx + (k - (s.f1.length - 1) / 2) * 9}" cy="${y(v)}" r="3.6" fill="#1A1916" stroke="#fff" stroke-width="2"/>`));
    const labY = Math.min(top, ...s.f1.map(y)) - 9;
    out.push(`<text class="val" x="${cx}" y="${labY}" text-anchor="middle">${f2(m)}</text>`);
    out.push(`<text x="${cx}" y="${H - 12}" text-anchor="middle" style="fill:${s.cfg === "gan_aug" ? "#1F6F66" : "#5A554C"};font-weight:${s.cfg === "gan_aug" ? 600 : 400}">${esc(s.label)}</text>`);
    const tipTxt = `<b>${esc(s.label)}</b><br>F1 ${f3(m)} ± ${f3(sd(s.f1))} (${s.n} seeds)<br>seeds: ${s.f1.map(f2).join(" · ")}<br>caught ${s.caught} of ${s.support}` + (s.paired && s.cfg !== "baseline_no_aug" ? `<br>vs baseline, same seed: ${s.paired.mean >= 0 ? "+" : ""}${f3(s.paired.mean)} (better in ${s.paired.better}/${s.n})` : "");
    out.push(`<rect x="${L + band * i}" y="${T}" width="${band}" height="${ih}" fill="transparent" data-tip="${esc(tipTxt)}"/>`);
  });
  if (base) out.push(`<line x1="${L}" x2="${W - R}" y1="${y(mean(base.f1))}" y2="${y(mean(base.f1))}" stroke="#5A554C" stroke-dasharray="4 4" stroke-width="1"/>`);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="F1 by method">${out.join("")}</svg>`;
}

function pairedChart(stats) {
  const have = stats.filter((s) => s.tuned.length);
  if (!have.length) return `<p class="foot">Run the Day 5 notebook to add the threshold test.</p>`;
  const W = 440, H = 230, L = 34, R = 8, T = 18, B = 34, iw = W - L - R, ih = H - T - B, band = iw / have.length;
  const y = (v) => T + ih * (1 - v), out = [];
  [0, 0.25, 0.5, 0.75, 1].forEach((g) => out.push(`<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" stroke="#ECE7DD"/><text x="${L - 6}" y="${y(g) + 3.5}" text-anchor="end">${g.toFixed(2)}</text>`));
  have.forEach((s, i) => {
    const cx = L + band * i + band / 2, bw = band * 0.34;
    [[mean(s.f1), "#3F72A8", -1], [mean(s.tuned), "#C47A2A", 1]].forEach(([v, col, side]) => {
      const x0 = side < 0 ? cx - bw - 1 : cx + 1, top = y(v), r = 4;
      out.push(`<path d="M${x0} ${y(0)}V${top + r}Q${x0} ${top} ${x0 + r} ${top}H${x0 + bw - r}Q${x0 + bw} ${top} ${x0 + bw} ${top + r}V${y(0)}Z" fill="${col}"/>`);
      out.push(`<text class="val" x="${x0 + bw / 2}" y="${top - 6}" text-anchor="middle" style="font-size:9.5px">${f2(v).replace(/^0/, "")}</text>`);
    });
    out.push(`<text x="${cx}" y="${H - 12}" text-anchor="middle">${esc(s.label)}</text>`);
    const tipTxt = `<b>${esc(s.label)}</b><br>default rule: F1 ${f3(mean(s.f1))}, ${f2(mean(s.fa))} false alarms/run<br>tuned threshold: F1 ${f3(mean(s.tuned))}, ${f2(mean(s.tunedFa))} false alarms/run`;
    out.push(`<rect x="${L + band * i}" y="${T}" width="${band}" height="${ih}" fill="transparent" data-tip="${esc(tipTxt)}"/>`);
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Default rule versus tuned threshold">${out.join("")}</svg>`;
}

function renderEvidence() {
  const rare = S.evRare, D = S.ev.diseases[rare];
  if (!D) return;
  const stats = ORDER.map((c) => D.methods.find((m) => m.config === c)).filter(Boolean).map(methodStats);
  const base = stats.find((s) => s.cfg === "baseline_no_aug"), gan = stats.find((s) => s.cfg === "gan_aug");
  const name = D.name.toLowerCase();

  const tiles = [];
  if (base && gan) {
    tiles.push([`${rare} F1 · baseline → GAN`, f2(mean(base.f1)), f2(mean(gan.f1)), `mean of ${gan.n} seeds · GAN better in ${gan.paired?.better ?? "?"} of ${gan.n}`]);
    if (base.fa.length && gan.fa.length) tiles.push(["False alarms per run", mean(base.fa).toFixed(1), mean(gan.fa).toFixed(1), `other diseases wrongly called ${rare}`]);
    if (base.ap.length && gan.ap.length) tiles.push(["Average precision · threshold-free", f2(mean(base.ap)), f2(mean(gan.ap)), "scores the model at every threshold at once"]);
    tiles.push([`${rare} cases caught · all seeds`, `${base.caught}`, `${gan.caught}`, `out of ${gan.support} real test images`]);
  }
  $("#ev-tiles").innerHTML = tiles.map(([k, a, b, s]) => `<div class="tile-stat"><span class="k">${esc(k)}</span>
      <span class="v"><span class="from">${a}</span><span class="arrow" aria-hidden="true">→</span><span class="to">${b}</span></span><span class="s">${esc(s)}</span></div>`).join("");

  const best = (k) => stats.reduce((a, s) => (mean(s[k]) > mean(a[k]) ? s : a), stats[0]);
  const tunedBase = base?.tuned.length ? mean(base.tuned) : null;
  $("#ev-verdict").innerHTML = rare === "DF"
    ? `On the harder rare disease, adding <b>new</b> images helped and plain copies did not. The GAN gave the highest F1 and average precision and the fewest false alarms` +
      (tunedBase != null ? `; simply tuning the baseline's threshold reached only ${f2(tunedBase)}.` : ".") +
      ` GAN, SMOTE and classic augmentation are within noise of each other — the honest claim is "competitive and most precise", not "clearly best".`
    : `${D.name}s are easier: the baseline already reaches F1 ${f2(mean(base.f1))}. Every kind of extra rare data helped a little, and the methods are too close to separate. The pattern that repeats from dermatofibroma: the GAN-trained model is the most precise, with the fewest false alarms.`;

  $("#ch1-title").textContent = `${rare} F1 on the test set`;
  $("#ch1").innerHTML = barChart(stats);
  $("#ch2").innerHTML = pairedChart(stats);
  bindTips($("#ev-report"));

  const cols = [["F1", (s) => s.f1], ["Recall", (s) => s.recall], ["Precision", (s) => s.precision], ["AP", (s) => s.ap]];
  const bestOf = Object.fromEntries(cols.map(([n, f]) => [n, stats.reduce((a, s) => ((mean(f(s)) ?? -1) > (mean(f(a)) ?? -1) ? s : a), stats[0]).cfg]));
  $("#results").innerHTML = `<thead><tr><th>Method</th>${cols.map(([n]) => `<th>${n}</th>`).join("")}<th>False alarms / run</th><th>Caught</th><th>vs baseline</th></tr></thead><tbody>` +
    stats.map((s) => `<tr class="${s.cfg === "gan_aug" ? "gan" : ""}"><td>${esc(s.label)}</td>` +
      cols.map(([n, f]) => { const v = f(s); return `<td class="${bestOf[n] === s.cfg ? "best" : ""}">${v.length ? `${f3(mean(v))}<span class="muted"> ± ${f2(sd(v))}</span>` : "—"}</td>`; }).join("") +
      `<td>${s.fa.length ? mean(s.fa).toFixed(1) : "—"}</td><td>${s.caught}/${s.support}</td>` +
      `<td>${s.cfg === "baseline_no_aug" ? "—" : `${s.paired.mean >= 0 ? "+" : ""}${f3(s.paired.mean)} (${s.paired.better}/${s.n})`}</td></tr>`).join("") + "</tbody>";

  const cases = (D.cases || []).map((c) => ({ ...c, total: Object.values(c.hits).reduce((a, v) => a + (v || 0), 0) }))
    .sort((a, b) => b.total - a.total);
  const labels = ORDER.map((c) => D.methods.find((m) => m.config === c)?.label).filter(Boolean);
  const never = cases.filter((c) => c.total === 0).length;
  $("#cases-sub").textContent = `${cases.length} real ${rare} test images · 5 cells = Baseline, Oversample, Classic, SMOTE, GAN (darker = caught in more seeds) · ${never} never caught`;
  $("#cases").innerHTML = cases.map((c) => {
    const cells = labels.map((l, i) => { const v = c.hits[l] ?? 0; const gan = i === labels.length - 1;
      return `<i style="background:${gan ? `rgba(31,111,102,${0.12 + 0.88 * v})` : `rgba(26,25,22,${0.08 + 0.8 * v})`}"></i>`; }).join("");
    const tipTxt = `<b>${esc(c.id)}</b><br>` + labels.map((l) => `${esc(SHORT[ORDER[labels.indexOf(l)]])}: ${Math.round((c.hits[l] ?? 0) * 100)}% of seeds`).join("<br>");
    return `<div class="case ${c.total === 0 ? "never" : ""}" data-tip="${esc(tipTxt)}"><img src="/${c.src}" alt="${esc(D.name)} test image ${esc(c.id)}" loading="lazy">
      <span class="hitbar">${cells}</span><span class="lbl">${c.total === 0 ? "never caught" : esc(c.id.replace("ISIC_", ""))}</span></div>`;
  }).join("");
  bindTips($("#cases"));
}

function initEvidence() {
  const rares = Object.keys(S.ev.diseases);
  S.evRare = rares.includes("DF") ? "DF" : rares[0];
  buildDiseaseSeg($("#ev-disease"), rares, S.evRare, (r) => { S.evRare = r; renderEvidence(); });
  renderEvidence();
}

/* ==================================================================== METHOD */
function initMethod() {
  const ds = S.ev.dataset, D = S.ev.diseases.DF, V = S.ev.diseases.VASC, g = S.ev.gan || {};
  const st = (d, c) => d && methodStats(d.methods.find((m) => m.config === c) || { seeds: [], config: c });
  const bDF = st(D, "baseline_no_aug"), gDF = st(D, "gan_aug");
  const steps = [
    ["01", "Data", `HAM10000: ${ds.images.toLocaleString()} dermoscopy images, 7 diagnoses. Dermatofibroma has only ${ds.counts.DF} images (${(ds.counts.DF / ds.images * 100).toFixed(2)} %). Split 70/15/15 by lesion, so no lesion appears in both training and test.`],
    ["02", "Baseline", `ResNet-18 with ImageNet weights, fine-tuned for 15 epochs. Checkpoint chosen on validation, test used once.` + (bDF?.f1.length ? ` Dermatofibroma F1 ${f2(mean(bDF.f1))}.` : "")],
    ["03", "Simple augmentation", "500 extra rare-class images three ways: plain copies, flips/rotations/crops/colour changes, and SMOTE blends of two similar real lesions."],
    ["04", "GAN", `A DCGAN with DiffAugment, spectral norm and EMA, trained on just ${g.DF?.train_images ?? 71} real images. Near-copies and outliers removed, 500 kept.`, "gan"],
    ["05", "Experiment", `5 methods × 3 seeds, one fixed real test set, the same recipe for every run.` + (gDF?.f1.length ? ` GAN augmentation: F1 ${f2(mean(bDF.f1))} → ${f2(mean(gDF.f1))}.` : "")],
    ["06", "Stress tests", `A threshold tuned on validation for every model, and the whole study repeated on a second rare disease (vascular lesions${V ? `, ${ds.counts.VASC} images` : ""}).`],
  ];
  const am = S.ev.app_model;
  if ($("#app-model-sub")) $("#app-model-sub").textContent = `measured once on the ${Object.values(ds.test_counts).reduce((a, b) => a + b, 0).toLocaleString()}-image test set`;
  if (am) {
    const rows = (am.candidates || []).map((c) => [c.name, c.test, c.chosen, c.val_rare_f1]);
    $("#app-model-table").innerHTML = `<thead><tr><th>Candidate</th><th>Val rare-F1</th><th>Accuracy</th><th>Macro-F1</th><th>DF F1</th><th>DF caught</th><th>VASC F1</th><th>VASC caught</th></tr></thead><tbody>` +
      rows.map(([n, v, me, sc]) => `<tr class="${me ? "gan" : ""}"><td>${esc(n)}${me ? " · chosen" : ""}</td><td>${f3(sc)}</td><td>${f3(v.accuracy)}</td><td>${f3(v.macro_f1)}</td><td>${f3(v.DF?.f1)}</td><td>${v.DF ? `${v.DF.caught}/${ds.test_counts.DF}` : "—"}</td><td>${f3(v.VASC?.f1)}</td><td>${v.VASC ? `${v.VASC.caught}/${ds.test_counts.VASC}` : "—"}</td></tr>`).join("") +
      `</tbody>` + (am.knn ? `<caption style="caption-side:bottom;text-align:left;padding-top:10px;font-family:var(--sans);font-size:12px;color:var(--l-mut)">Test columns are shown for transparency only; the choice used the validation column. Unusual-image check: flags ${pct(am.knn.test_flagged_share, 1)} of real test images (by design about 1%).</caption>` : "");
  } else $("#app-model-card").hidden = true;
  $("#steps").innerHTML = steps.map(([n, h, p, cls]) => `<li class="${cls || ""}"><span class="n">${n}</span><div><h3>${h}</h3><p>${esc(p)}</p></div></li>`).join("");
}

/* ====================================================================== boot */
async function boot() {
  try {
    [S.meta, S.ev] = await Promise.all([api("/api/meta"), api("/api/evidence")]);
  } catch (e) {
    document.querySelector("main").innerHTML = `<div class="error-box">RareLens could not load its data: ${esc(e.message)}.<br>Run <code>python prepare_app.py</code>, then restart <code>python app.py</code>.</div>`;
    return;
  }
  $("#device").textContent = S.meta.device === "CPU" ? "CPU" : `GPU · ${S.meta.device}`;
  if (S.meta.app_model_name) $("#report-eyebrow").textContent = `Report · RareLens model · ${S.meta.app_model_name}`;
  initExamine(); initSynthesize(); initEvidence(); initMethod();
  addEventListener("hashchange", route);
  route();
}
boot();
