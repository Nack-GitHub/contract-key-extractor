// Demo page for the contract key-information extraction API (contract_ner.api).
// No framework and no build step: the page is served by `contract-ner serve` and works offline.

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const LABELS = {
  PARTY: "Party",
  AGREEMENT_DATE: "Agreement date",
  START_DATE: "Start date",
  END_DATE: "End date",
  JURISDICTION: "Jurisdiction",
  CONTRACT_VALUE: "Contract value",
  PAYMENT_DUE: "Payment due",
  PENALTY_RATE: "Penalty rate",
};

// table cells, in the order a reader expects; `slot` ties a cell to the gold check of the slot evaluation
const CELLS = [
  { key: "Party_A", title: "Party A", label: "PARTY", slot: "Parties" },
  { key: "Party_B", title: "Party B", label: "PARTY", slot: "Parties" },
  { key: "Agreement_Date", title: "Agreement date", label: "AGREEMENT_DATE", slot: "Agreement_Date" },
  { key: "Start_Date", title: "Start date", label: "START_DATE", slot: "Start_Date" },
  { key: "End_Date", title: "End date", label: "END_DATE", slot: "End_Date" },
  { key: "Jurisdiction", title: "Jurisdiction / governing law", label: "JURISDICTION", slot: "Jurisdiction" },
  { key: "Contract_Value", title: "Contract value", label: "CONTRACT_VALUE", lang: "th" },
  { key: "Payment_Due_Date", title: "Payment due", label: "PAYMENT_DUE", lang: "th" },
  { key: "Penalty_Rate", title: "Penalty rate", label: "PENALTY_RATE", lang: "th" },
];
const REVIEW_BELOW = 0.8;
const VERDICT_TEXT = {
  correct: "matches gold",
  wrong: "differs from gold",
  missed: "missed",
  extra: "not in gold",
};
const VERDICT_HELP = {
  extra: "Predicted where the gold table is empty: a model error, or a value the annotators missed.",
  missed: "The gold table has a value; the model gave none.",
  wrong: "The model's value differs from the gold value.",
  correct: "Same value as the gold table (after normalisation).",
};

const state = {
  models: [],
  samples: [],
  sample: null,          // full sample (text + gold entities) when a curated contract is selected
  custom: null,          // { text, name } for pasted / uploaded text
  result: null,          // last /api/extract response
  hidden: new Set(),     // labels switched off in the legend
};

// ---------------------------------------------------------------- utilities

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `${res.status} ${res.statusText}`);
  return body;
}
const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function toast(msg, ms = 4000) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), ms);
}
function busy(on, msg = "Running the model…") {
  $("#loading div").textContent = msg;
  $("#loading").classList.toggle("show", on);
}
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};
const fmt = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const modelById = (id) => state.models.find((m) => m.id === id);

// ---------------------------------------------------------------- tabs, theme, notes

function showTab(id) {
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    const on = tab.id === id;
    tab.setAttribute("aria-selected", on);
    $("#" + tab.getAttribute("aria-controls")).classList.toggle("active", on);
  }
  store.set("tab", id);
  if (id === "tab-eval") renderEvaluation();
}
document.querySelectorAll('[role="tab"]').forEach((t) => t.addEventListener("click", () => showTab(t.id)));

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
$("#theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  applyTheme(next);
  store.set("theme", next);
  if ($("#panel-eval").classList.contains("active")) renderEvaluation();
});

function setNotes(on) {
  $("#notes-toggle").setAttribute("aria-pressed", on);
  document.querySelectorAll(".notes").forEach((n) => n.classList.toggle("show", on && n.textContent.trim() !== ""));
  store.set("notes", on ? "1" : "0");
}
$("#notes-toggle").addEventListener("click", () => setNotes($("#notes-toggle").getAttribute("aria-pressed") !== "true"));
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "n") $("#notes-toggle").click();
  if (e.key === "1") showTab("tab-extract");
  if (e.key === "2") showTab("tab-compare");
  if (e.key === "3") showTab("tab-eval");
});

// ---------------------------------------------------------------- start-up

async function init() {
  applyTheme(store.get("theme"));
  try {
    const [health, models, samples] = await Promise.all([api("/api/health"), api("/api/models"), api("/api/samples")]);
    state.models = models;
    state.samples = samples;
    $("#server-status").textContent = health.cached_only ? "server · stored records only" : "server · live models";
  } catch (err) {
    $("#server-status").textContent = "server unreachable";
    $("#server-status").classList.add("error");
    toast("Cannot reach the API: " + err.message, 8000);
    return;
  }
  fillSampleSelect($("#sample-select"), true);
  fillSampleSelect($("#cmp-sample"), false);
  fillModelSelect();
  const first = store.get("sample") && state.samples.some((s) => s.id === store.get("sample")) ? store.get("sample") : state.samples[0]?.id;
  if (first) {
    $("#sample-select").value = first;
    await selectSample(first);
  }
  setNotes(store.get("notes") === "1");
  showTab(store.get("tab") || "tab-extract");
}

function fillSampleSelect(sel, withCustom) {
  const groups = { en: [], th: [] };
  for (const s of state.samples) (groups[s.lang] || groups.en).push(s);
  sel.innerHTML = "";
  for (const [lang, title] of [["en", "English — CUAD"], ["th", "Thai"]]) {
    if (!groups[lang].length) continue;
    const og = document.createElement("optgroup");
    og.label = title;
    for (const s of groups[lang]) {
      const o = new Option(`${s.title}  ·  ${s.chars.toLocaleString()} chars${s.has_gold ? "  ·  gold" : ""}`, s.id);
      og.append(o);
    }
    sel.append(og);
  }
  if (withCustom) {
    const o = new Option("Your text (pasted or uploaded)", "__custom");
    o.disabled = true;
    sel.append(o);
  }
}

function fillModelSelect() {
  const sel = $("#model-select");
  sel.innerHTML = "";
  for (const m of state.models) {
    const tag = m.available ? "" : "  (stored records only)";
    sel.append(new Option(`${m.title} [${m.lang}]${tag}`, m.id));
  }
}

function currentLang() {
  return state.sample ? state.sample.lang : (modelById($("#model-select").value)?.lang || "en");
}

async function selectSample(id) {
  if (id === "__custom") return;
  busy(true, "Loading contract…");
  try {
    state.sample = await api(`/api/samples/${encodeURIComponent(id)}`);
  } finally {
    busy(false);
  }
  state.custom = null;
  state.result = null;
  store.set("sample", id);
  const meta = state.samples.find((s) => s.id === id);
  $("#model-select").value = meta?.default_model || (state.sample.lang === "th" ? "thai" : "roberta");
  $("#gold-toggle").disabled = !state.sample.entities?.length;
  $("#gold-toggle").checked = false;
  $("#cmp-sample").value = id;
  setNoteText();
  renderAll();
}

function setNoteText() {
  const meta = state.sample && state.samples.find((s) => s.id === state.sample.id);
  const text = meta?.talking_point ? `<b>${esc(meta.title)}.</b> ${esc(meta.talking_point)}` : "";
  $("#notes-extract").innerHTML = text;
  $("#notes-compare").innerHTML = text;
  setNotes($("#notes-toggle").getAttribute("aria-pressed") === "true");
}

// ---------------------------------------------------------------- extract

$("#sample-select").addEventListener("change", (e) => selectSample(e.target.value));
$("#conf").addEventListener("input", (e) => { $("#conf-val").textContent = Number(e.target.value).toFixed(2); });
$("#run-btn").addEventListener("click", runExtract);
$("#gold-toggle").addEventListener("change", renderAll);
$("#invalid-toggle").addEventListener("change", () => $("#doc").classList.toggle("hide-invalid", !$("#invalid-toggle").checked));
$("#doc").classList.add("hide-invalid");

$("#paste-btn").addEventListener("click", () => {
  $("#paste-text").value = state.custom?.text || "";
  $("#paste-dialog").showModal();
});
$("#paste-dialog").addEventListener("close", () => {
  if ($("#paste-dialog").returnValue !== "ok") return;
  const text = $("#paste-text").value;
  if (!text.trim()) return;
  useCustomText(text, "pasted text");
  runExtract();
});
$("#file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("model", $("#model-select").value);
  form.append("min_confidence", $("#conf").value);
  busy(true);
  try {
    const res = await api("/api/extract/file", { method: "POST", body: form });
    useCustomText(res.text, file.name);
    state.result = res;
    renderAll();
  } catch (err) {
    toast(err.message, 7000);
  } finally {
    busy(false);
  }
});

function useCustomText(text, name) {
  state.sample = null;
  state.custom = { text, name };
  state.result = null;
  $("#sample-select").value = "__custom";
  $("#gold-toggle").checked = false;
  $("#gold-toggle").disabled = true;
  setNoteText();
  renderAll();
}

async function runExtract() {
  const body = { model: $("#model-select").value, min_confidence: Number($("#conf").value) };
  if (state.sample) body.sample_id = state.sample.id;
  else if (state.custom) body.text = state.custom.text;
  else return toast("Pick a contract first.");
  const m = modelById(body.model);
  if (m && m.lang !== currentLang()) toast(`Note: ${m.title} is a ${m.lang === "th" ? "Thai" : "English"} model.`);
  busy(true);
  try {
    state.result = await postJSON("/api/extract", body);
    if (state.result.source === "cache") toast("Shown from the stored record" + (state.result.note ? ` (${state.result.note})` : ""), 5000);
    renderAll();
  } catch (err) {
    toast(err.message, 7000);
  } finally {
    busy(false);
  }
}

function docText() {
  return state.sample?.text ?? state.custom?.text ?? "";
}

function renderAll() {
  renderDocument();
  renderCells();
  renderLegend();
  renderRunMeta();
}

// Split the text at every predicted and gold boundary so overlapping highlights nest cleanly.
function renderDocument() {
  const text = docText();
  const doc = $("#doc");
  if (!text) {
    doc.innerHTML = '<p class="empty">Pick a contract, paste text or upload a .txt file, then press <b>Extract</b>.</p>';
    return;
  }
  const ents = state.result?.record?.entities || [];
  const showGold = $("#gold-toggle").checked && state.sample?.entities?.length;
  const gold = showGold ? state.sample.entities : [];
  const cuts = new Set([0, text.length]);
  for (const e of ents) { cuts.add(e.start); cuts.add(e.end); }
  for (const g of gold) { cuts.add(g.start); cuts.add(g.end); }
  const points = [...cuts].sort((a, b) => a - b);
  // predicted spans from one tagger never overlap; neither do gold spans
  const entAt = new Map();
  ents.forEach((e, i) => entAt.set(e.start, i));
  const goldAt = new Map();
  gold.forEach((g, i) => goldAt.set(g.start, i));

  const parts = [];
  let curEnt = -1;
  let curGold = -1;
  for (let k = 0; k < points.length - 1; k++) {
    const a = points[k];
    const b = points[k + 1];
    if (entAt.has(a)) curEnt = entAt.get(a);
    if (curEnt >= 0 && a >= ents[curEnt].end) curEnt = -1;
    if (goldAt.has(a)) curGold = goldAt.get(a);
    if (curGold >= 0 && a >= gold[curGold].end) curGold = -1;
    let html = esc(text.slice(a, b));
    if (curGold >= 0) html = `<span class="gold" data-gold="${curGold}">${html}</span>`;
    if (curEnt >= 0) {
      const e = ents[curEnt];
      const cls = ["ent"];
      if (!e.valid) cls.push("invalid");
      else if (e.confidence < REVIEW_BELOW) cls.push("low");
      if (state.hidden.has(e.label)) cls.push("off");
      const id = a === e.start ? ` id="ent-${curEnt}"` : "";
      html = `<span class="${cls.join(" ")}" data-label="${e.label}" data-ent="${curEnt}"${id}>${html}</span>`;
    }
    parts.push(html);
  }
  doc.innerHTML = parts.join("");
}

function entityIndex(start, end) {
  const ents = state.result?.record?.entities || [];
  return ents.findIndex((e) => e.start === start && e.end === end);
}

function jumpTo(start, end) {
  const i = entityIndex(start, end);
  const el = i >= 0 ? $(`#ent-${i}`) : null;
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  const all = document.querySelectorAll(`[data-ent="${i}"]`);
  all.forEach((x) => { x.classList.remove("flash"); void x.offsetWidth; x.classList.add("flash"); });
}

function cellRecord(rec, cell) {
  // the governing-law clause is what CUAD's category means; the slot evaluation reads it first
  if (cell.key === "Jurisdiction" && rec.lang !== "th") return rec.Governing_Law || rec.Jurisdiction;
  return rec[cell.key];
}

function cellValue(rec, cell) {
  const s = rec.summary || {};
  if (cell.key === "Jurisdiction") {
    const r = cellRecord(rec, cell);
    const n = r?.normalized;
    return (n && typeof n === "object" ? n.english : null) || s.Jurisdiction || r?.text || null;
  }
  return s[cell.key] ?? null;
}

function renderCells() {
  const ul = $("#cells");
  const rec = state.result?.record;
  const lang = state.result?.lang || currentLang();
  const cells = CELLS.filter((c) => !c.lang || c.lang === lang);
  const checks = Object.fromEntries((state.result?.gold_check || []).map((c) => [c.slot, c]));
  const showGold = $("#gold-toggle").checked && state.result?.gold_check;
  if (!rec) {
    ul.innerHTML = cells.map((c) => `<li class="cell empty" data-label="${c.label}"><div class="cell-head"><span class="swatch"></span>${esc(c.title)}</div><div class="cell-value">—</div></li>`).join("");
    $("#table-meta").textContent = "";
    return;
  }
  let right = 0;
  let graded = 0;
  const html = [];
  const partyCheckShown = { done: false };
  for (const c of cells) {
    const r = cellRecord(rec, c);
    const value = cellValue(rec, c);
    const conf = r?.confidence;
    const flags = [];
    if (r?.inferred_from) flags.push(`<span class="badge" title="No separate start date was found; the agreement date is used">inferred from ${esc(r.inferred_from.replace("_", " "))}</span>`);
    if (r && conf < REVIEW_BELOW) flags.push('<span class="badge flag-review" title="Confidence below 0.8: check this cell against the source">review</span>');
    if (r?.mentions > 1) flags.push(`<span class="badge">${r.mentions} mentions</span>`);
    let goldHtml = "";
    if (showGold && c.slot && checks[c.slot]) {
      const ch = checks[c.slot];
      const isParty = c.slot === "Parties";
      if (!isParty || !partyCheckShown.done) {
        if (ch.verdict !== "none") {
          graded += 1;
          if (ch.verdict === "correct") right += 1;
        }
        if (isParty) partyCheckShown.done = true;
      }
      // parties are graded as one set, so the verdict sits on Party A only
      if (ch.verdict !== "none" && (!isParty || c.key === "Party_A")) {
        flags.push(`<span class="verdict ${ch.verdict}" title="${esc(VERDICT_HELP[ch.verdict])}">${isParty ? "party set " : ""}${VERDICT_TEXT[ch.verdict]}</span>`);
        if (ch.verdict !== "correct") {
          const g = Array.isArray(ch.gold) && ch.gold.length ? ch.gold.join(" · ") : "(no gold value)";
          goldHtml = `<div class="gold-line">gold: <b>${esc(g)}</b></div>`;
          if (isParty && ch.extra?.length) goldHtml += `<div class="gold-line">also tagged as a party: <b>${esc(ch.extra.join(" · "))}</b></div>`;
          if (isParty && ch.missing?.length) goldHtml += `<div class="gold-line">not found: <b>${esc(ch.missing.join(" · "))}</b></div>`;
        }
      }
    }
    const src = r && r.text && r.text !== value ? `<div class="cell-src">“${esc(r.text)}”</div>` : "";
    const confHtml = conf != null ? `<span class="conf" title="Mean label probability of the span">${conf.toFixed(2)}<span class="conf-bar"><i style="width:${Math.round(conf * 100)}%"></i></span></span>` : "";
    html.push(`<li class="cell${r ? "" : " empty"}" data-label="${c.label}" ${r ? `data-start="${r.start}" data-end="${r.end}" tabindex="0" role="button" title="Show in the document"` : ""}>
      <div class="cell-head"><span class="swatch"></span>${esc(c.title)}${confHtml}</div>
      <div class="cell-value">${value ? esc(value) : "not found"}</div>${src}
      ${flags.length ? `<div class="cell-flags">${flags.join("")}</div>` : ""}${goldHtml}
    </li>`);
  }
  const others = (rec.Other_Parties || []).filter((x) => x && x.trim());
  if (others.length) {
    html.splice(2, 0, `<li class="cell empty" data-label="PARTY"><div class="cell-head"><span class="swatch"></span>Other names tagged as parties</div>
      <div class="cell-src" style="-webkit-line-clamp:3">${others.map((o) => esc(o.replace(/\s+/g, " "))).join(" · ")}</div>
      <div class="cell-flags"><span class="badge" title="Aliases such as &quot;Customer&quot; and the &quot;referred to as the Parties&quot; boilerplate are dropped from the party set; anything else here counts as a party">not in the table</span></div></li>`);
  }
  ul.innerHTML = html.join("");
  $("#table-meta").textContent = showGold && graded ? `${right} / ${graded} slots match gold` : "";
}

$("#cells").addEventListener("click", (e) => {
  const li = e.target.closest(".cell[data-start]");
  if (li) jumpTo(Number(li.dataset.start), Number(li.dataset.end));
});
$("#cells").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  const li = e.target.closest(".cell[data-start]");
  if (li) { e.preventDefault(); jumpTo(Number(li.dataset.start), Number(li.dataset.end)); }
});

function renderLegend() {
  const ents = state.result?.record?.entities || [];
  const counts = {};
  for (const e of ents) if (e.valid) counts[e.label] = (counts[e.label] || 0) + 1;
  const rejected = ents.filter((e) => !e.valid).length;
  const lang = state.result?.lang || currentLang();
  const labels = Object.keys(LABELS).filter((l) => counts[l] || lang === "th" || !["CONTRACT_VALUE", "PAYMENT_DUE", "PENALTY_RATE"].includes(l));
  $("#legend").innerHTML = labels.map((l) => `<button data-label="${l}" aria-pressed="${!state.hidden.has(l)}" title="Show / hide ${esc(LABELS[l])} highlights"><span class="swatch"></span>${esc(LABELS[l])} <span class="count">${counts[l] || 0}</span></button>`).join("");
  $("#span-count").textContent = ents.length ? `${ents.length - rejected} kept · ${rejected} rejected` : "";
}
$("#legend").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-label]");
  if (!b) return;
  const l = b.dataset.label;
  if (state.hidden.has(l)) state.hidden.delete(l); else state.hidden.add(l);
  renderDocument();
  renderLegend();
});

function renderRunMeta() {
  const r = state.result;
  if (!r) {
    $("#run-meta").innerHTML = "<span>Nothing extracted yet.</span>";
    $("#export-btn").disabled = true;
    return;
  }
  const m = modelById(r.model);
  const chars = docText().length;
  $("#run-meta").innerHTML = [
    `<span>model <b>${esc(m?.title || r.model)}</b></span>`,
    `<span class="badge ${r.source}" title="${r.source === "cache" ? "Record stored when the demo contract was curated" : "Model ran on this request"}">${r.source === "cache" ? "stored record" : "live"}</span>`,
    r.seconds != null ? `<span><b>${r.seconds.toFixed(2)} s</b> for ${chars.toLocaleString()} characters</span>` : "",
    m?.metrics?.slot != null ? `<span title="${esc(m.metrics.report)}">test-split slot F1 <b>${fmt(m.metrics.slot)}</b></span>` : "",
  ].join("");
  $("#export-btn").disabled = false;
}

$("#export-btn").addEventListener("click", () => {
  const r = state.result;
  if (!r) return;
  const blob = new Blob([JSON.stringify(r.record, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.sample?.id || "contract"}__${r.model}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// tooltip over highlighted spans
const tip = $("#tooltip");
function placeTip(ev) {
  const pad = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
$("#doc").addEventListener("mousemove", (ev) => {
  const span = ev.target.closest(".ent:not(.off)");
  const goldEl = ev.target.closest(".gold");
  if (!span && !goldEl) { tip.style.display = "none"; return; }
  const rows = [];
  let title = "";
  if (span) {
    const e = state.result.record.entities[Number(span.dataset.ent)];
    title = `<span class="swatch" data-label="${e.label}" style="background:var(--c)"></span>${esc(LABELS[e.label] || e.label)}`;
    rows.push(["confidence", e.confidence.toFixed(3)]);
    rows.push(["validation", e.valid ? "kept" : "rejected (surface form cannot be this type)"]);
    rows.push(["characters", `${e.start}–${e.end}`]);
  }
  if (goldEl) {
    const g = state.sample.entities[Number(goldEl.dataset.gold)];
    if (!title) title = "Gold span";
    rows.push(["gold", `${LABELS[g.label] || g.label}`]);
  }
  tip.innerHTML = `<div class="tt-title">${title}</div><dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
  tip.style.display = "block";
  placeTip(ev);
});
$("#doc").addEventListener("mouseleave", () => { tip.style.display = "none"; });
$("#doc").addEventListener("click", (ev) => {
  const span = ev.target.closest(".ent:not(.off)");
  if (!span) return;
  const e = state.result.record.entities[Number(span.dataset.ent)];
  const li = [...document.querySelectorAll("#cells .cell[data-start]")].find((x) => Number(x.dataset.start) === e.start && Number(x.dataset.end) === e.end);
  if (li) { li.scrollIntoView({ block: "nearest", behavior: "smooth" }); li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash"); }
});

// ---------------------------------------------------------------- compare

$("#cmp-sample").addEventListener("change", (e) => {
  $("#sample-select").value = e.target.value;
  selectSample(e.target.value).then(renderCompareModels);
});
$("#cmp-run").addEventListener("click", runCompare);

function renderCompareModels() {
  const lang = state.sample?.lang || "en";
  const box = $("#cmp-models");
  const usable = state.models.filter((m) => m.lang === lang);
  const order = ["regex", "crf", "roberta", "thai"];
  usable.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  box.innerHTML = usable.map((m) => `<label><input type="checkbox" value="${m.id}" checked> ${esc(m.title)}</label>`).join(" ");
  box.style.display = "inline-flex";
  box.style.gap = "14px";
}

async function runCompare() {
  const models = [...document.querySelectorAll("#cmp-models input:checked")].map((i) => i.value);
  if (!models.length) return toast("Tick at least one model.");
  const body = { models, min_confidence: Number($("#conf").value) };
  if (state.sample) body.sample_id = state.sample.id;
  else if (state.custom) body.text = state.custom.text;
  else return toast("Pick a contract first.");
  busy(true, `Running ${models.length} model${models.length > 1 ? "s" : ""}…`);
  try {
    renderCompare(await postJSON("/api/compare", body), models);
  } catch (err) {
    toast(err.message, 7000);
  } finally {
    busy(false);
  }
}

function renderCompare(res, models) {
  const lang = state.sample?.lang || "en";
  const cells = CELLS.filter((c) => !c.lang || c.lang === lang);
  const disagree = new Set(res.disagreements);
  const head = models.map((id) => {
    const r = res.results[id];
    const m = modelById(id);
    const sub = r.error ? `<span class="sub error">${esc(r.error)}</span>`
      : `<span class="sub">${r.source === "cache" ? "stored record" : "live"}${r.seconds != null ? ` · ${r.seconds.toFixed(2)} s` : ""}${m?.metrics?.slot != null ? ` · test slot F1 ${fmt(m.metrics.slot)}` : ""}</span>`;
    return `<th>${esc(m?.title || id)}${sub}</th>`;
  }).join("");
  const rows = cells.map((c) => {
    const tds = models.map((id) => {
      const r = res.results[id];
      if (r.error) return "<td></td>";
      const v = r.summary[c.key];
      const ch = (r.gold_check || []).find((x) => x.slot === c.slot);
      const verdict = ch && ch.verdict !== "none" && (c.slot !== "Parties" || c.key === "Party_A")
        ? `<div><span class="verdict ${ch.verdict}" title="${esc(VERDICT_HELP[ch.verdict])}">${c.slot === "Parties" ? "party set " : ""}${VERDICT_TEXT[ch.verdict]}</span></div>` : "";
      return `<td><div class="val${v ? "" : " none"}">${v ? esc(v) : "not found"}</div>${verdict}</td>`;
    }).join("");
    const dis = c.slot && disagree.has(c.slot);
    return `<tr class="${dis ? "disagree" : ""}"><th data-label="${c.label}"><span class="swatch" style="display:inline-block;margin-right:6px"></span>${esc(c.title)}${dis ? '<div class="hint" style="margin:2px 0 0">models disagree</div>' : ""}</th>${tds}</tr>`;
  }).join("");
  const anyGold = models.some((id) => res.results[id].gold_check);
  const foot = anyGold ? `<tfoot><tr><th>Slots matching gold</th>${models.map((id) => {
    const ch = (res.results[id].gold_check || []).filter((x) => x.verdict !== "none");
    return `<td>${ch.length ? `<b>${ch.filter((x) => x.verdict === "correct").length} / ${ch.length}</b>` : ""}</td>`;
  }).join("")}</tr></tfoot>` : "";
  $("#cmp-out").innerHTML = `<table class="compare-table"><thead><tr><th>Cell</th>${head}</tr></thead><tbody>${rows}</tbody>${foot}</table>
    <p class="hint">Values are what each model's table shows after the same validation and normalisation. Parties are graded as one set (both names together).</p>`;
}

// ---------------------------------------------------------------- evaluation

let metricsCache = null;
async function renderEvaluation() {
  const root = $("#eval");
  if (!metricsCache) {
    try { metricsCache = await api("/api/metrics"); } catch (err) { root.innerHTML = `<p class="error">${esc(err.message)}</p>`; return; }
  }
  const M = metricsCache;
  const by = Object.fromEntries(M.models.map((m) => [m.id, m]));
  const en = ["regex", "crf", "roberta"].filter((id) => by[id]);
  const shipped = by.roberta;
  const thai = by.thai;
  const tiles = [
    shipped && { k: "Slot F1 — shipped English model", v: fmt(shipped.slot), s: `${shipped.n_docs} held-out CUAD contracts · target ≥ ${M.target.with_review.toFixed(2)} with review` },
    shipped && { k: "Slot precision / recall", v: `${fmt(shipped.slot_precision, 2)} / ${fmt(shipped.slot_recall, 2)}`, s: "one value per table cell, normalised" },
    by.regex && shipped && { k: "Gain over the pattern baseline", v: `+${(shipped.slot - by.regex.slot).toFixed(3)}`, s: `regex ${fmt(by.regex.slot)} · CRF ${fmt(by.crf?.slot)}` },
    thai && { k: "Thai — blind gold slot F1", v: fmt(thai.slot), s: `${thai.n_docs} contracts written after the last iteration` },
  ].filter(Boolean);
  root.innerHTML = tiles.map((t) => `<div class="tile"><div class="k">${esc(t.k)}</div><div class="v">${esc(t.v)}</div><div class="s">${esc(t.s)}</div></div>`).join("")
    + `<div class="card chart-card"><h2>Slot F1 by model <span class="meta">CUAD test split</span></h2><div class="chart" id="chart-models"></div>
         <p class="caption">Same contracts, same validation and normalisation; only the tagger changes. The dashed line is the "usable with review" target.</p>
         ${tableView(["Model", "slot F1", "precision", "recall", "strict F1", "relaxed F1"], en.map((id) => [by[id].title, fmt(by[id].slot), fmt(by[id].slot_precision), fmt(by[id].slot_recall), fmt(by[id].strict), fmt(by[id].relaxed)]))}</div>`
    + `<div class="card chart-card"><h2>English model across iterations <span class="meta">test split</span></h2><div class="chart" id="chart-iter"></div>
         <p class="caption">v6 → v7 kept the same weights: normalisation, selection rules and the confidence floor moved the table more than any retrain. Strict span F1 later falls because the model tags every mention of a party, while CUAD's gold has only the defining one.</p>
         ${tableView(["Iteration", "change", "strict F1", "slot F1"], M.iterations.map((it) => [it.name, it.change, fmt(it.strict), fmt(it.slot)]))}</div>`
    + `<div class="card chart-card wide"><h2>Slot F1 per table cell <span class="meta">CUAD test split</span></h2><div class="chart-legend" id="cells-legend"></div><div class="chart" id="chart-cells"></div>
         <p class="caption">The conditional random field holds up where the decision is local — a governing-law clause, a dated preamble — and falls away on Parties, where telling the defining mention of a name from every later one needs the whole preamble. The pattern baseline is close on the preamble cells and falls behind on End date, where the term clause is phrased a hundred ways.</p>
         ${tableView(["Cell", ...en.map((id) => by[id].title), "docs with gold"], Object.keys(shipped?.cells || {}).map((cell) => [cell.replace("_", " "), ...en.map((id) => fmt(by[id].cells[cell]?.f1, 2)), shipped.cells[cell].docs_with_gold]))}</div>`;
  barChart($("#chart-models"), en.map((id) => ({ label: by[id].title, value: by[id].slot, id })), M.target.with_review);
  lineChart($("#chart-iter"), M.iterations);
  groupedBars($("#chart-cells"), $("#cells-legend"), Object.keys(shipped?.cells || {}), en.map((id) => ({ id, title: by[id].title, cells: by[id].cells })));
}

function tableView(head, rows) {
  return `<details class="table-view"><summary>Table view</summary><table class="num-table"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></details>`;
}

const SVG = "http://www.w3.org/2000/svg";
// draw at the card's own width so 11px text stays 11px
const chartWidth = (host) => Math.max(320, Math.round(host.clientWidth - 28));
function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.append(el);
  return el;
}
// a bar with 4px rounded data-end and a square base on the axis
function barPath(x, y, w, h, horizontal) {
  const r = Math.min(4, horizontal ? h / 2 : w / 2, horizontal ? w : h);
  if (horizontal) return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}
function hoverTip(el, html) {
  el.addEventListener("mousemove", (ev) => { tip.innerHTML = html; tip.style.display = "block"; placeTip(ev); });
  el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}

function barChart(host, data, target) {
  const W = chartWidth(host), rowH = 34, left = 170, right = 70, top = 8;
  const H = top + data.length * rowH + 24;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Slot F1 by model" }, host);
  const x = (v) => left + v * (W - left - right);
  const grid = svgEl("g", { class: "grid tick" }, svg);
  for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    svgEl("line", { x1: x(t), x2: x(t), y1: top, y2: H - 22 }, grid);
    svgEl("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, grid).textContent = t.toFixed(1);
  }
  // the target goes under the bars and their labels, so a label near 0.80 stays readable
  svgEl("line", { x1: x(target), x2: x(target), y1: top - 4, y2: H - 22, class: "target" }, svg);
  svgEl("text", { x: x(target) + 4, y: top + 4, class: "tick" }, svg).textContent = `target ${target.toFixed(2)}`;
  data.forEach((d, i) => {
    const y = top + i * rowH + 7;
    const h = rowH - 14;
    svgEl("text", { x: left - 10, y: y + h / 2 + 4, "text-anchor": "end" }, svg).textContent = d.label;
    svgEl("path", { d: barPath(left, y, x(d.value) - left, h, true), fill: d.id === "roberta" ? "var(--series-1)" : "var(--axis)" }, svg);
    svgEl("text", { x: x(d.value) + 6, y: y + h / 2 + 4, class: "value-label" }, svg).textContent = d.value.toFixed(3);
    const hit = svgEl("rect", { x: 0, y: y - 6, width: W, height: rowH, class: "hit" }, svg);
    hoverTip(hit, `<div class="tt-title">${esc(d.label)}</div><dl><dt>slot F1</dt><dd>${d.value.toFixed(3)}</dd><dt>vs target</dt><dd>${(d.value - target >= 0 ? "+" : "") + (d.value - target).toFixed(3)}</dd></dl>`);
  });
  svgEl("line", { x1: left, x2: left, y1: top, y2: H - 22, class: "baseline" }, svg);
}

function lineChart(host, its) {
  const W = chartWidth(host), H = 250, left = 36, right = 56, top = 12, bottom = 28;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Strict and slot F1 across iterations" }, host);
  const x = (i) => left + (i * (W - left - right)) / (its.length - 1);
  const y = (v) => top + (1 - v) * (H - top - bottom);
  const grid = svgEl("g", { class: "grid tick" }, svg);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { x1: left, x2: W - right, y1: y(t), y2: y(t) }, grid);
    svgEl("text", { x: left - 6, y: y(t) + 4, "text-anchor": "end" }, grid).textContent = t.toFixed(2);
  }
  its.forEach((it, i) => { svgEl("text", { x: x(i), y: H - 8, "text-anchor": "middle", class: "tick" }, svg).textContent = it.name; });
  const series = [
    { key: "slot", label: "slot F1 (the table)", color: "var(--series-1)" },
    { key: "strict", label: "strict span F1", color: "var(--series-2)" },
  ];
  for (const s of series) {
    const pts = its.map((it, i) => (it[s.key] == null ? null : [x(i), y(it[s.key])])).filter(Boolean);
    svgEl("path", { d: "M" + pts.map((p) => p.join(",")).join(" L"), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    pts.forEach(([px, py]) => svgEl("circle", { cx: px, cy: py, r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg));
    const last = pts[pts.length - 1];
    svgEl("text", { x: last[0] + 8, y: last[1] + 4, class: "value-label" }, svg).textContent = its[its.length - 1][s.key].toFixed(3);
  }
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = series.map((s) => `<span><i class="line" style="background:${s.color}"></i>${esc(s.label)}</span>`).join("") + '<span style="color:var(--muted)">slot F1 was introduced at v6</span>';
  host.before(legend);
  const hover = svgEl("line", { class: "hover-line", y1: top, y2: H - bottom, x1: 0, x2: 0, visibility: "hidden" }, svg);
  its.forEach((it, i) => {
    const w = (W - left - right) / (its.length - 1);
    const hit = svgEl("rect", { x: x(i) - w / 2, y: top, width: w, height: H - top - bottom, class: "hit" }, svg);
    hit.addEventListener("mouseenter", () => { hover.setAttribute("x1", x(i)); hover.setAttribute("x2", x(i)); hover.setAttribute("visibility", "visible"); });
    hoverTip(hit, `<div class="tt-title">${esc(it.name)}</div><div style="margin-top:4px;color:var(--ink-2)">${esc(it.change)}</div><dl><dt>slot F1</dt><dd>${fmt(it.slot)}</dd><dt>strict F1</dt><dd>${fmt(it.strict)}</dd></dl>`);
  });
  svg.addEventListener("mouseleave", () => hover.setAttribute("visibility", "hidden"));
}

function groupedBars(host, legendHost, cells, models) {
  const W = chartWidth(host), H = 260, left = 36, right = 10, top = 14, bottom = 30;
  const colors = ["var(--series-2)", "var(--series-3)", "var(--series-1)"];   // regex, CRF, shipped model
  legendHost.innerHTML = models.map((m, i) => `<span><i style="background:${colors[i]}"></i>${esc(m.title)}</span>`).join("");
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Slot F1 per table cell and model" }, host);
  const y = (v) => top + (1 - v) * (H - top - bottom);
  const grid = svgEl("g", { class: "grid tick" }, svg);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { x1: left, x2: W - right, y1: y(t), y2: y(t) }, grid);
    svgEl("text", { x: left - 6, y: y(t) + 4, "text-anchor": "end" }, grid).textContent = t.toFixed(2);
  }
  const groupW = (W - left - right) / cells.length;
  const barW = Math.min(44, (groupW - 40) / models.length - 2);
  cells.forEach((cell, ci) => {
    const gx = left + ci * groupW + (groupW - models.length * (barW + 2)) / 2;
    svgEl("text", { x: left + ci * groupW + groupW / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = cell.replace("_", " ");
    models.forEach((m, mi) => {
      const v = m.cells[cell]?.f1 ?? 0;
      const bx = gx + mi * (barW + 2);
      svgEl("path", { d: barPath(bx, y(v), barW, y(0) - y(v), false), fill: colors[mi] }, svg);
      if (mi === models.length - 1) svgEl("text", { x: bx + barW / 2, y: y(v) - 5, "text-anchor": "middle", class: "value-label" }, svg).textContent = v.toFixed(2);
      const hit = svgEl("rect", { x: bx - 1, y: top, width: barW + 2, height: H - top - bottom, class: "hit" }, svg);
      const c = m.cells[cell] || {};
      hoverTip(hit, `<div class="tt-title"><span class="swatch" style="background:${colors[mi]}"></span>${esc(m.title)} — ${esc(cell.replace("_", " "))}</div><dl><dt>F1</dt><dd>${fmt(c.f1)}</dd><dt>precision</dt><dd>${fmt(c.precision)}</dd><dt>recall</dt><dd>${fmt(c.recall)}</dd><dt>docs with gold</dt><dd>${c.docs_with_gold ?? "–"}</dd></dl>`);
    });
  });
  svgEl("line", { x1: left, x2: W - right, y1: y(0), y2: y(0), class: "baseline" }, svg);
}

let resizeTimer = null;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if ($("#panel-eval").classList.contains("active")) renderEvaluation(); }, 150);
});

init().then(renderCompareModels);
