// Arabic-first UI for the CCTV search PoC. Plain ES module, no build step, no external requests.
// Every string the page shows comes from web/locales/<language>.json through t() (see i18n.js).
// The page has two views: the business view at / (search, recordings, audit log; always the
// hybrid search) and the technical view at /tech, which adds the search mode switch, the
// "Under the hood" tab and the links into it.

import { currentLanguage, initLanguage, maybe, switchLanguage, t, translatePage } from "/i18n.js";

const $ = (sel, root = document) => root.querySelector(sel);

const TECH = location.pathname.replace(/\/+$/, "") === "/tech";

// Example queries that have matches in the public starter footage. They are sample search
// inputs, not page text, so they stay the same in every language.
const EXAMPLES = [
  "امرأة ترتدي سترة حمراء",
  "شخص يرتدي سترة حمراء بقلنسوة",
  "زلمة لابس بنطلون بني",
  "رجّال يسحب شنطة سفر",
  "طفلة ترتدي قميصاً أبيض وبنطال جينز",
  "شنطة سفر بنية",
  "a person in a red hoodie",
];

// "Why it matched": one mark per result of an attribute comparison.
const MARKS = { match: "✓", near: "≈", unknown: "?", missing: "–", mismatch: "✗" };

const STAGE_ORDER = ["input", "parse", "embed", "filter", "candidates", "attributes", "fusion", "ranking", "check", "final"];

// Per-viewer conveniences only; the audit log on the server is the record.
function remember(key, value, storage = "localStorage") {
  try { window[storage].setItem(key, value); } catch { /* storage may be blocked */ }
}
function recall(key, storage = "localStorage") {
  try { return window[storage].getItem(key) || ""; } catch { return ""; }
}

function actor() {
  return $("#actor").value.trim() || "anonymous";
}

// Build an element: el("div", {class: "x", text: "y", onclick: fn}, child, ...).
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child != null && child !== false) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

// A server value such as "English rewrite" as a table key suffix: "english-rewrite".
function slug(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = response.headers.get("content-type")?.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function postJSON(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

function fmtTime(iso) {
  return iso ? iso.replace("T", " ").replace(/\.\d+$/, "") : "";
}

function fmtDuration(seconds) {
  if (seconds == null) return "";
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function num(value, digits = 3) {
  return value == null ? "–" : Number(value).toFixed(digits);
}

function ms(value) {
  return value == null ? null : t("unit.ms", { ms: value });
}

function list(items) {
  return items.join(t("list.separator"));
}

// Wrap a value from the data (often English) so it keeps its own direction inside a sentence.
function isolate(value) {
  return `\u2068${value}\u2069`;
}

function errorText(err) {
  return t("error.generic", { message: isolate(err.message ?? err) });
}

// Places of the starter footage are stored as "Arabic · English"; show the half in the page language.
// Any other place (such as one typed at upload) is shown as it was entered.
const ARABIC_LETTER = /[\u0600-\u06ff]/;

function placeText(place) {
  const parts = String(place ?? "").split(" · ");
  if (parts.length !== 2) return place;
  const arabic = parts.map((part) => ARABIC_LETTER.test(part));
  if (arabic[0] === arabic[1]) return place;
  return parts[arabic.indexOf(currentLanguage() === "ar")];
}

// Words for the fixed description vocabulary (avs/fields.py) in the current language.

function word(group, value) {
  return value == null ? "" : maybe(`${group}.${value}`, String(value));
}

// "red or white jacket" in the page language; in Arabic the colours agree with a feminine noun.
function phrase(colours, noun) {
  const gender = noun ? maybe(`noun.${noun}@gender`, "m") : "m";
  const shades = (colours || []).map((c) => maybe(`colour.${c}@${gender}`, c)).join(t("phrase.or"));
  if (!noun) return shades;
  return shades ? t("phrase.coloured", { thing: word("noun", noun), colours: shades }) : word("noun", noun);
}

function hairPhrase(colour, length) {
  return t("phrase.hair", { colour: word("hairColour", colour), length: word("hairLength", length) }).replace(/\s+/g, " ").trim();
}

// The query's value for one compared field. Trace rows carry it; result cards only name the
// field, so it comes from the parsed query (nth picks the carried object).
function queryValue(match, fields, nth) {
  if (match.query !== undefined) return match.query;
  if (!fields) return undefined;
  switch (match.field) {
    case "gender": return fields.gender;
    case "age_band": return fields.age_band;
    case "upper_colour": return fields.upper?.colours;
    case "lower_colour": return fields.lower?.colours;
    case "upper_garment": return fields.upper?.garment;
    case "lower_garment": return fields.lower?.garment;
    case "hair_colour": return fields.hair?.colour;
    case "hair_length": return fields.hair?.length;
    case "headwear": return fields.headwear;
    case "carried": return fields.carried?.[nth];
    default: return undefined;
  }
}

// The words for one compared field in the page language, or the server's English label when
// the query value is not at hand.
function whyLabel(match, fields, nth = 0) {
  const q = queryValue(match, fields, nth);
  if (q == null) return match.label;
  switch (match.field) {
    case "gender": return word("gender", q);
    case "age_band": return word("age", q);
    case "upper_colour": return phrase(q, fields?.upper?.garment || "top");
    case "lower_colour": return phrase(q, fields?.lower?.garment || "trousers");
    case "upper_garment":
    case "lower_garment": return word("noun", q);
    case "hair_colour": return hairPhrase(q, null);
    case "hair_length": return hairPhrase(null, q);
    case "headwear": return phrase(q.colours, q.type);
    case "carried": return phrase(q.colours, q.object);
    default: return match.label;
  }
}

// Tabs

function showTab(name) {
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === name));
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.hidden = panel.id !== `tab-${name}`;
  }
  if (name === "recordings") loadRecordings();
  if (name === "audit") loadAudit();
  if (name === "search") loadCameras();
  if (name === "hood") {
    loadRecordingChoices();
    checkHealth();
  }
}

function showHood(name) {
  for (const button of document.querySelectorAll("[data-hood]")) {
    button.setAttribute("aria-selected", String(button.dataset.hood === name));
  }
  $("#hood-search").hidden = name !== "search";
  $("#hood-recording").hidden = name !== "recording";
}

// Search

async function loadCameras() {
  const box = $("#camera-filter");
  const checked = new Set([...box.querySelectorAll("input:checked")].map((i) => i.value));
  let cameras;
  try {
    cameras = await api("/api/cameras");
  } catch {
    return;
  }
  if (!cameras.length) return;
  box.replaceChildren(...cameras.map((cam) => {
    const label = document.createElement("label");
    label.className = "chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = cam.camera_id;
    input.checked = checked.has(cam.camera_id);
    const text = document.createElement("span");
    text.textContent = t("filters.camera", { camera: cam.camera_id, places: list(cam.places.map(placeText)) });
    label.append(input, text);
    return label;
  }));
}

function searchFilters() {
  const cameras = [...document.querySelectorAll("#camera-filter input:checked")].map((i) => i.value);
  return {
    cameras: cameras.length ? cameras : null,
    time_from: $("#time-from").value || null,
    time_to: $("#time-to").value || null,
  };
}

// The business view has no mode switch, so it always searches in the hybrid mode.
function searchMode() {
  return document.querySelector("input[name=mode]:checked")?.value || "hybrid";
}

let searchToken = 0;
let searchCheck = null; // the search whose double-check this page is following
const CHECK_FINISHED = ["done", "failed", "skipped", "cancelled", "interrupted"];

// Stop the double-check of a search the page no longer shows, so a new search is not queued behind it.
function cancelCheck(searchId) {
  if (searchId) fetch(`/api/search/${encodeURIComponent(searchId)}/check/cancel`, { method: "POST" }).catch(() => {});
}

async function runSearch(event) {
  event?.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  const token = ++searchToken;
  cancelCheck(searchCheck);
  searchCheck = null;
  const status = $("#search-status");
  status.textContent = t("search.searching");
  $("#results").replaceChildren();
  $("#parsed").hidden = true;
  $("#check-status").textContent = "";
  try {
    const data = await postJSON("/api/search", {
      query,
      mode: searchMode(),
      ...searchFilters(),
      actor: actor(),
      case_id: $("#case-id").value.trim() || null,
    });
    if (token !== searchToken) return;
    const fallback = searchMode() === "hybrid" && data.mode !== "hybrid"
      ? t(TECH ? "search.fallbackTech" : "search.fallback") : "";
    const found = data.results.length
      ? t("search.results", { n: data.results.length, ms: data.took_ms })
      : t("search.noResults");
    status.textContent = [found, fallback].filter(Boolean).join(t("list.join"));
    if (data.parsed) renderParsed(data);
    const checking = data.check && data.check.status !== "skipped" ? data.check.total : 0;
    const cards = new Map(data.results.map((r) => [r.track_key, resultCard(r, data.search_id, r.rank <= checking, data.parsed)]));
    $("#results").replaceChildren(...cards.values());
    if (data.check && data.check.status !== "skipped") {
      searchCheck = data.search_id;
      followCheck(data.search_id, cards, token, checkLabels(data.check.details, data.parsed));
    }
  } catch (err) {
    status.textContent = errorText(err);
  }
}

// The fields the query model read from the description, one chip each.
function fieldChips(fields) {
  const chips = [];
  if (fields.subject === "bag") chips.push(t("parsed.bag"));
  if (fields.gender) chips.push(word("gender", fields.gender));
  if (fields.age_band && fields.age_band !== "adult") chips.push(word("age", fields.age_band));
  if (fields.upper.colours.length || fields.upper.garment) chips.push(phrase(fields.upper.colours, fields.upper.garment || "top"));
  if (fields.lower.colours.length || fields.lower.garment) chips.push(phrase(fields.lower.colours, fields.lower.garment || "trousers"));
  if (fields.hair.colour || fields.hair.length) chips.push(hairPhrase(fields.hair.colour, fields.hair.length));
  if (fields.headwear.type) chips.push(phrase(fields.headwear.colours, fields.headwear.type));
  for (const item of fields.carried) {
    const thing = phrase(item.colours, item.object);
    chips.push(fields.subject === "bag" ? thing : t("parsed.carries", { thing }));
  }
  // Other details are the model's own English words, so only the technical view shows them.
  if (TECH) for (const other of fields.other) chips.push(t("parsed.other", { text: isolate(other) }));
  return chips.map((text) => el("span", { class: "chip field-chip", text }));
}

function renderParsed(data) {
  const box = $("#parsed");
  box.replaceChildren(
    el("div", { class: "parsed-head", text: t("parsed.title") }),
    el("div", { class: "parsed-chips" }, ...fieldChips(data.parsed)),
  );
  // The English rewrite the embedder reads, and the way into the trace, are for the technical view.
  if (TECH) {
    box.append(el("div", { class: "parsed-english" }, t("parsed.english"), " ", el("span", { dir: "ltr", lang: "en", text: `“${data.parsed.english}”` })),
      el("button", { type: "button", class: "link", onclick: () => openTrace(data.search_id), text: t("parsed.how") }));
  }
  box.hidden = false;
}

function whyChips(why, fields) {
  let carried = 0;
  return (why || []).map((w) => el("span", {
    class: `why-chip why-${w.result}`,
    title: t("why.title", { field: word("field", w.field), result: t(`why.${w.result}`) }),
    text: `${MARKS[w.result]} ${whyLabel(w, fields, w.field === "carried" ? carried++ : 0)}`,
  }));
}

function answerChip(detail, answer) {
  return el("span", { class: `why-chip answer-${answer}`, text: t("check.answer", { detail: isolate(detail), answer: word("answer", answer) }) });
}

// The details the vision model is asked about, in the page language, keyed by the English text the
// model answers to. Free-text details (field "other") have no words of their own and are left out.
function checkLabels(details, fields) {
  const labels = new Map();
  let carried = 0;
  for (const d of details || []) {
    let label = null;
    if (d.field === "gender" && fields.age_band === "child") label = word("child", fields.gender);
    else if (d.field === "carried") label = whyLabel({ field: "carried" }, fields, carried++);
    else if (d.field !== "other") label = whyLabel({ field: d.field }, fields);
    if (label) labels.set(d.text, label);
  }
  return labels;
}

// The technical view shows exactly what the model was asked and its own reasoning; the business
// view shows the verdict and the answers for the details the page has words for.
function setCheck(card, verdict, labels) {
  const box = $(".check", card);
  if (!verdict) return;
  const label = word("verdict", verdict.verdict);
  const answers = Object.entries(verdict.answers || {})
    .filter(([detail]) => TECH || labels.has(detail))
    .map(([detail, answer]) => answerChip(TECH ? detail : labels.get(detail), answer));
  const note = TECH && verdict.note ? [el("div", { class: "muted small", dir: "auto", text: verdict.note })] : [];
  box.replaceChildren(el("div", { class: `verdict-badge verdict-${verdict.verdict}`, text: label }),
    el("div", { class: "answers" }, ...answers), ...note);
  card.dataset.check = verdict.verdict;
}

async function followCheck(searchId, cards, token, labels) {
  const status = $("#check-status");
  for (;;) {
    if (token !== searchToken) return;
    let job;
    try {
      job = await api(`/api/search/${searchId}/check`);
    } catch {
      return;
    }
    if (token !== searchToken) return; // a newer search started while this reply was on its way
    if (CHECK_FINISHED.includes(job.status) && searchCheck === searchId) searchCheck = null;
    for (const [key, verdict] of Object.entries(job.verdicts || {})) {
      const card = cards.get(key);
      if (card && card.dataset.check !== verdict.verdict) setCheck(card, verdict, labels);
    }
    status.textContent = t("check.progress", { done: job.done, total: job.total });
    if (job.status === "done") {
      const order = job.results.map((r) => cards.get(r.track_key)).filter(Boolean);
      job.results.forEach((r) => { const card = cards.get(r.track_key); if (card) $(".rank", card).textContent = `#${r.rank}`; });
      $("#results").replaceChildren(...order);
      const count = (v) => Object.values(job.verdicts).filter((x) => x.verdict === v).length;
      status.textContent = t("check.done", { confirmed: count("confirmed"), rejected: count("rejected"), unsure: count("unsure"), ms: job.took_ms });
      return;
    }
    if (job.status === "failed") {
      status.textContent = t("check.failed", { error: isolate(job.error || "") });
      return;
    }
    if (job.status === "cancelled" || job.status === "interrupted") {
      status.textContent = t("check.stopped", { done: job.done, total: job.total, reason: job.reason || job.status });
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function resultCard(result, searchId, checking, fields) {
  const card = $("#result-card").content.firstElementChild.cloneNode(true);
  translatePage(card);
  const video = $("video", card);
  $(".crop", card).src = result.crop_url;
  if (result.video_url) {
    video.src = `${result.video_url}#t=${result.clip_start},${result.clip_end}`;
    video.controls = true;
    // Keep playback inside the clip: start at the clip, stop at its end.
    video.addEventListener("play", () => {
      if (video.currentTime < result.clip_start || video.currentTime >= result.clip_end - 0.1) {
        video.currentTime = result.clip_start;
      }
    });
    video.addEventListener("timeupdate", () => {
      if (video.currentTime >= result.clip_end) video.pause();
    });
  }
  $(".rank", card).textContent = `#${result.rank}`;
  $(".camera", card).textContent = result.camera_id;
  $(".label-badge", card).textContent = word("label", result.label);
  $(".score", card).textContent = result.similarity == null
    ? t("card.score", { score: result.score.toFixed(3) })
    : t("card.scoreSimilarity", { score: result.score.toFixed(2), similarity: result.similarity.toFixed(3) });
  $(".place", card).textContent = placeText(result.place);
  $(".time", card).textContent = `${fmtTime(result.time)}  (${fmtTime(result.seen_from).slice(11)} – ${fmtTime(result.seen_to).slice(11)})`;
  if (result.why) {
    const why = $(".why", card);
    why.replaceChildren(...whyChips(result.why, fields));
    if (!result.described) why.append(el("span", { class: "muted small", text: t("card.noAttributes") }));
  }
  if (checking) $(".check", card).replaceChildren(el("div", { class: "verdict-badge verdict-pending", text: t("verdict.pending") }));
  for (const verdict of ["confirm", "reject"]) {
    $(`.${verdict}`, card).addEventListener("click", async () => {
      try {
        await postJSON("/api/feedback", { search_id: searchId, track_key: result.track_key, verdict, actor: actor() });
        card.dataset.verdict = verdict;
      } catch (err) {
        alert(errorText(err));
      }
    });
  }
  return card;
}

function renderExamples() {
  $("#examples").replaceChildren(...EXAMPLES.map((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.dir = "auto";
    button.textContent = text;
    button.addEventListener("click", () => {
      $("#query").value = text;
      runSearch();
    });
    return button;
  }));
}

// Under the hood: search trace

let traceToken = 0;
let traceCheck = null; // the traced hybrid search whose double-check the page is following

function modeTitle(mode) {
  return t(mode === "hybrid" ? "mode.hybrid" : "mode.baseline");
}

async function runTrace(event) {
  event?.preventDefault();
  const query = $("#trace-query").value.trim();
  if (!query) return;
  const token = ++traceToken;
  cancelCheck(traceCheck);
  traceCheck = null;
  const status = $("#trace-status");
  status.textContent = t("trace.tracing");
  $("#trace-columns").replaceChildren();
  const body = { query, ...searchFilters(), actor: actor(), case_id: $("#case-id").value.trim() || null };
  try {
    const baseline = await postJSON("/api/search?trace=1", { ...body, mode: "baseline" });
    const hybrid = await postJSON("/api/search?trace=1", { ...body, mode: "hybrid" });
    if (token !== traceToken) return;
    status.textContent = t("trace.times", { baseline: baseline.took_ms, hybrid: hybrid.took_ms });
    $("#trace-columns").replaceChildren(traceColumn(baseline.trace), traceColumn(hybrid.trace));
    if (hybrid.check && hybrid.check.status !== "skipped") {
      traceCheck = hybrid.search_id;
      followTrace(hybrid.search_id, 1, token);
    }
  } catch (err) {
    status.textContent = errorText(err);
  }
}

async function followTrace(searchId, index, token) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    if (token !== traceToken) return;
    let trace;
    try {
      trace = await api(`/api/search/${searchId}/trace`);
    } catch {
      return;
    }
    const columns = $("#trace-columns");
    const old = columns.children[index];
    if (old) old.replaceWith(traceColumn(trace));
    const check = trace.stages.find((s) => s.key === "check");
    if (!check || CHECK_FINISHED.includes(check.output.status)) {
      if (traceCheck === searchId) traceCheck = null;
      return;
    }
  }
}

async function openTrace(searchId) {
  showTab("hood");
  showHood("search");
  $("#trace-id").value = searchId;
  const token = ++traceToken;
  const status = $("#trace-status");
  status.textContent = t("trace.opening");
  try {
    const trace = await api(`/api/search/${encodeURIComponent(searchId)}/trace`);
    if (token !== traceToken) return;
    status.textContent = t("trace.stored", { id: searchId });
    $("#trace-columns").replaceChildren(traceColumn(trace));
    const check = trace.stages.find((s) => s.key === "check");
    if (check && ["queued", "running"].includes(check.output.status)) followTrace(searchId, 0, token);
  } catch (err) {
    status.textContent = errorText(err);
  }
}

function thumb(url, size = "small") {
  return url ? el("img", { class: `thumb thumb-${size}`, src: url, alt: t("trace.cropAlt"), loading: "lazy" }) : el("span");
}

function shortKey(key) {
  return key ? key.split("/").pop() : "";
}

// A definition list of [table key, value] pairs; empty values are left out.
function kv(pairs) {
  return el("dl", { class: "kv" }, ...pairs.filter(([, v]) => v != null && v !== "")
    .flatMap(([k, v]) => [el("dt", { text: t(`kv.${k}`) }), el("dd", { dir: "auto" }, v)]));
}

function pre(value) {
  return el("pre", { class: "json", dir: "ltr", text: typeof value === "string" ? value : JSON.stringify(value, null, 2) });
}

function details(summary, ...content) {
  return el("details", {}, el("summary", { text: summary }), ...content);
}

// A table with headers named by table keys under "col.".
function table(headers, rows) {
  return el("div", { class: "table-wrap" }, el("table", { class: "trace-table" },
    el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", { text: h ? t(`col.${h}`) : "" })))),
    el("tbody", {}, ...rows.map((cells) => el("tr", {}, ...cells.map((c) => el("td", {}, c)))))));
}

// What a track's description holds for one compared field, in the page language.
const TRACK_WORDS = { gender: "gender", age_band: "age", upper_garment: "noun", lower_garment: "noun", hair_colour: "hairColour", hair_length: "hairLength" };

function trackValue(value, field) {
  if (value == null) return t("value.nothing");
  if (Array.isArray(value)) {
    if (!value.length) return t("value.nothing");
    return list(value.map((v) => (typeof v === "object" ? phrase(v.colours, v.object) : word("colour", v))));
  }
  if (typeof value === "object") return value.type || value.object ? phrase(value.colours, value.type || value.object) : t("value.none");
  return word(TRACK_WORDS[field] || "value", value);
}

function traceColumn(trace) {
  const crops = new Map();
  for (const stage of trace.stages) {
    const rows = Array.isArray(stage.output) ? stage.output : stage.output?.rows;
    for (const row of rows || []) if (row.crop_url) crops.set(row.track_key, row.crop_url);
  }
  const fields = trace.stages.find((s) => s.key === "parse")?.output?.fields;
  const models = trace.models || {};
  const query = models.query_model;
  const request = trace.request;
  const head = el("header", { class: "trace-head" },
    el("h3", { text: modeTitle(trace.mode) }),
    kv([
      ["searchId", trace.search_id],
      ["mode", trace.mode],
      ["firstResults", ms(trace.took_ms)],
      ["cameras", request.cameras ? list(request.cameras) : t("value.allCameras")],
      ["time", request.time_from || request.time_to ? `${request.time_from || "…"} → ${request.time_to || "…"}` : t("value.anyTime")],
      ["embedder", models.embedder],
      ["queryModel", query ? `${query.model}${query.quantization ? ` (${query.quantization}, ${query.parameters})` : ""} · Ollama ${query.ollama_version || "?"} · ${query.status}` : null],
      ["visionModel", models.vision_model],
    ]),
    ...(trace.notes || []).map((n) => el("p", { class: "note", text: n })));
  const stages = STAGE_ORDER.map((key, i) => {
    const stage = trace.stages.find((s) => s.key === key);
    const title = `${i + 1}. ${t(`stage.${key}`)}`;
    if (!stage) {
      return el("article", { class: "stage stage-absent" },
        el("h4", { text: title }),
        el("p", { class: "muted small", text: t(trace.mode === "baseline" ? "trace.notInBaseline" : "trace.notRunYet") }));
    }
    return el("article", { class: `stage stage-${key}` },
      el("h4", {}, title, el("span", { class: "ms", text: ms(stage.ms) ?? "" })),
      stageBody(key, stage.output || {}, crops, fields));
  });
  return el("section", { class: "trace-col" }, head, ...stages);
}

function stageBody(key, out, crops, fields) {
  switch (key) {
    case "input":
      return kv([["text", out.text], ["characters", out.characters], ["script", maybe(`script.${slug(out.script)}`, out.script)]]);
    case "parse":
      if (out.error) return el("div", {}, el("p", { class: "error", text: out.error }), kv([["fallback", out.fallback]]), out.raw ? pre(out.raw) : null);
      return el("div", {},
        el("div", { class: "parsed-chips" }, ...fieldChips({ ...out.fields, english: out.english })),
        el("p", { class: "parsed-english" }, t("parsed.english"), " ", el("span", { dir: "ltr", lang: "en", text: `“${out.english}”` })),
        kv([["model", out.model], ["modelTime", out.cached ? t("value.cached") : ms(out.model_ms)],
          ["tokens", out.prompt_tokens != null ? `${out.prompt_tokens} / ${out.output_tokens}` : null],
          ["load", out.load_ms ? ms(out.load_ms) : null]]),
        details(t("trace.rawJson"), pre(out.raw)));
    case "embed":
      return el("div", {}, kv([["model", out.model], ["textInput", out.padding]]),
        ...(out.texts || []).map((v) => el("div", { class: "embed" },
          el("strong", { text: `${maybe(`embedRole.${slug(v.role)}`, v.role)}: ` }), el("span", { text: t("trace.quoted", { text: isolate(v.text) }) }),
          el("div", { class: "muted small", text: t("trace.vector", { dim: v.dim, norm: v.norm, first: v.first.join(", ") }) }))));
    case "filter":
      return el("div", {}, el("ul", { dir: "ltr" }, ...(out.explain || []).map((line) => el("li", { text: line }))),
        kv([["cameras", Array.isArray(out.cameras) ? list(out.cameras) : out.cameras], ["from", out.time_from], ["to", out.time_to],
          ["labels", Array.isArray(out.labels) ? list(out.labels.map((l) => word("label", l))) : out.labels], ["dropGender", out.exclude_gender]]),
        details(t("trace.qdrantFilter"), pre(out.qdrant_filter)));
    case "candidates": {
      const rows = out.rows || [];
      const sims = rows.length && rows[0].sim_english != null;
      return el("div", {},
        kv([["perText", out.pool_per_text], ["foundByEnglish", out.found_by_english],
          ["foundByOriginal", out.found_by_original], ["candidateTracks", out.tracks], ["cropsScored", out.crops_scored]]),
        table(sims ? ["", "track", "camera", "cosEnglish", "cosOriginal", "similarity"] : ["", "track", "camera", "label", "cosine"],
          rows.slice(0, 20).map((r) => sims
            ? [thumb(r.crop_url), shortKey(r.track_key), r.camera_id, num(r.sim_english), num(r.sim_original), num(r.similarity)]
            : [thumb(r.crop_url), shortKey(r.track_key), r.camera_id, word("label", r.label), num(r.similarity)])),
        rows.length > 20 ? el("p", { class: "muted small", text: t("trace.showing", { shown: 20, n: rows.length }) }) : null);
    }
    case "attributes":
      return el("div", {},
        // Every row compares the same query fields in the same order, so the first row names them.
        kv([["queryFields", list(out.rows?.length ? out.rows[0].matches.map((m) => whyLabel(m, fields)) : out.query_fields || []) || t("value.none")],
          ["describedTracks", out.described_tracks],
          ["values", list(Object.entries(out.values || {}).map(([k, v]) => `${word("why", k)} ${v}`))]]),
        table(["", "track", "attributeScore", "perField"], (out.rows || []).slice(0, 15).map((r) => [
          thumb(crops.get(r.track_key)), shortKey(r.track_key), num(r.attribute_score, 2),
          el("div", { dir: "auto" }, ...r.matches.map((m) => el("span", {
            class: `why-chip why-${m.result}`, text: `${MARKS[m.result]} ${t("trace.matchPair", { query: whyLabel(m, fields), track: trackValue(m.track, m.field) })}`,
          }))),
        ])));
    case "fusion":
      return el("div", {}, el("p", { class: "formula", dir: "ltr", text: out.formula }),
        kv([["similarityMean", num(out.similarity_mean)], ["similarityStd", num(out.similarity_std)]]),
        table(["", "track", "similarity", "z", "attribute", "score"], (out.rows || []).slice(0, 15).map((r) => [
          thumb(crops.get(r.track_key)), shortKey(r.track_key), num(r.similarity), num(r.z, 2), num(r.attribute_score, 2), num(r.score, 2)])));
    case "ranking":
      return el("div", { class: "rank-grid" }, ...(Array.isArray(out) ? out : []).map((r) => el("figure", {},
        thumb(r.crop_url, "medium"), el("figcaption", {}, el("span", { dir: "ltr", text: `#${r.rank} ${shortKey(r.track_key)} · ${num(r.score, 2)}` }),
          r.why ? el("div", {}, ...whyChips(r.why, fields)) : null))));
    case "check": {
      const verdicts = Object.entries(out.verdicts || {});
      return el("div", {},
        kv([["status", word("jobStatus", out.status)], ["checked", t("trace.checked", { done: out.done, total: out.total })], ["model", out.model],
          ["took", out.took_ms ? ms(out.took_ms) : null],
          ["detailsAsked", (out.details || []).map((d) => (d.key ? d.text : t("trace.advisory", { text: isolate(d.text) }))).join("; ")],
          ["note", out.reason || out.error]]),
        table(["", "track", "verdict", "answers", "note", "ms"], verdicts.map(([trackKey, v]) => [
          el("span", {}, ...(v.crops || []).map((u) => thumb(u))), shortKey(trackKey),
          el("span", { class: `verdict-badge verdict-${v.verdict}`, text: word("verdictShort", v.verdict) }),
          el("div", {}, ...Object.entries(v.answers || {}).map(([d, a]) => answerChip(d, a))),
          el("span", { class: "small", dir: "auto", text: v.note || "" }), v.ms ?? ""])));
    }
    case "final":
      return el("div", { class: "rank-grid" }, ...(Array.isArray(out) ? out : []).map((r) => {
        const moved = r.rank_before_check - r.rank;
        const arrow = moved > 0 ? `↑${moved}` : moved < 0 ? `↓${-moved}` : "=";
        const check = r.check.replace(" ", "-");
        return el("figure", {}, thumb(r.crop_url, "medium"),
          el("figcaption", {}, el("span", { dir: "ltr", text: `#${r.rank} (${arrow}) ${shortKey(r.track_key)}` }),
            el("div", { class: `verdict-badge verdict-${check}`, text: word("verdictShort", check) })));
      }));
    default:
      return pre(out);
  }
}

// Under the hood: recording trace

async function loadRecordingChoices() {
  let recordings;
  try {
    recordings = await api("/api/recordings");
  } catch {
    return;
  }
  const select = $("#rectrace-recording");
  const current = select.value;
  select.replaceChildren(...recordings.filter((r) => r.status === "indexed").map((r) =>
    el("option", { value: r.id, text: t("rectrace.option", { camera: r.camera_id, start: fmtTime(r.start_time), n: r.track_count }) })));
  if (current) select.value = current;
}

async function runRecordingTrace(event, at) {
  event?.preventDefault();
  const id = $("#rectrace-recording").value;
  if (!id) return;
  const moments = at ?? $("#rectrace-at").value.trim();
  const status = $("#rectrace-status");
  status.textContent = t("rectrace.running");
  try {
    const trace = await api(`/api/recordings/${id}/trace${moments ? `?at=${encodeURIComponent(moments)}` : ""}`);
    status.textContent = "";
    $("#rectrace").replaceChildren(...recordingTrace(trace));
  } catch (err) {
    status.textContent = errorText(err);
  }
}

function frameFigure(frame, fps) {
  const boxes = (box, cls, label) => el("div", {
    class: `box ${cls}`,
    style: `left:${box[0] * 100}%;top:${box[1] * 100}%;width:${(box[2] - box[0]) * 100}%;height:${(box[3] - box[1]) * 100}%`,
  }, el("span", { text: label }));
  return el("figure", { class: "frame" },
    el("div", { class: "frame-image", dir: "ltr" }, el("img", { src: frame.url, alt: t("rectrace.frameAlt", { t: frame.t }) }),
      ...frame.detections.map((d) => boxes(d.box, "det", `${word("label", d.label)} ${d.confidence.toFixed(2)}`)),
      ...frame.tracks.map((tr) => boxes(tr.box, "trk", tr.track_id))),
    el("figcaption", {
      text: t("rectrace.frameCaption", {
        t: frame.t.toFixed(2), detections: frame.detections.length, ms: frame.detect_ms,
        tracks: frame.tracks.length, every: (1 / fps).toFixed(2),
      }),
    }));
}

function attributeSummary(attrs) {
  if (!attrs) return "–";
  if (!attrs.attributes) return attrs.error ? t("rectrace.unreadable") : "–";
  const a = attrs.attributes;
  if (a.subject === "bag") return trackValue(a.carried);
  const part = (p) => p && (p.colours.length || p.garment) ? phrase(p.colours, p.garment) : null;
  return [word("gender", a.gender), word("age", a.age_band), part(a.upper), part(a.lower),
    a.hair?.colour && hairPhrase(a.hair.colour, null), a.headwear?.type && word("noun", a.headwear.type),
    a.carried?.length && t("parsed.carries", { thing: trackValue(a.carried) })].filter(Boolean).join(" · ");
}

function trackDetails(track, onFrame) {
  const chosen = track.crops.map((c) => el("figure", {}, thumb(c.url, "medium"),
    el("figcaption", { text: t("rectrace.cropCaption", { t: c.t.toFixed(2), quality: num(c.quality) }) })));
  const pool = track.selection?.candidates || [];
  const attrs = track.attributes;
  const selection = track.selection;
  return el("div", { class: "track-details" },
    el("h5", { text: t("rectrace.chosenCrops") }), el("div", { class: "rank-grid" }, ...chosen),
    track.crops.length ? el("button", { type: "button", onclick: () => onFrame(track.crops[0].t), text: t("rectrace.showFrame") }) : null,
    el("h5", { text: t("rectrace.whyCrops") }),
    selection ? el("p", { class: "muted small", text: t("rectrace.selection", {
      detections: selection.detections, kept: selection.pool_capacity, gap: selection.min_gap,
      labels: isolate(JSON.stringify(selection.labels)),
    }) }) : el("p", { class: "muted", text: t("rectrace.noSelection") }),
    pool.length ? table(["t", "quality", "confidence", "size", "edge", "picked", "reason"], pool.map((c) => [
      c.t.toFixed(2), num(c.quality), num(c.confidence), num(c.size), c.edge, c.chosen ? "✓" : "", c.reason])) : null,
    el("h5", { text: t("rectrace.attributes") }),
    attrs ? el("div", {}, kv([["source", attrs.source], ["time", ms(attrs.ms)],
      ["summary", attributeSummary(attrs)], ["sentence", attrs.attributes?.description], ["error", attrs.error]]),
    details(t("rectrace.attributesJson"), pre(attrs.attributes)), attrs.raw ? details(t("rectrace.rawReply"), pre(attrs.raw)) : null,
    attrs.probabilities ? details(t("rectrace.probabilities"), pre(attrs.probabilities)) : null)
      : el("p", { class: "muted", text: t("rectrace.noAttributes") }));
}

function recordingTrace(trace) {
  const rec = trace.recording;
  const stats = trace.stats || {};
  const rules = trace.rules;
  const models = stats.models || {};
  const out = [];
  out.push(el("div", { class: "cards-row" },
    el("article", { class: "stage" }, el("h4", { text: t("rectrace.recordingCard") }), kv([
      ["camera", rec.camera_id], ["place", placeText(rec.place)], ["startLocal", fmtTime(rec.start_time)], ["length", fmtDuration(rec.duration_s)],
      ["size", `${rec.width}x${rec.height}`], ["video", stats.video ? t("unit.fps", { fps: stats.video.fps?.toFixed(1) }) : null],
      ["file", rec.original_name]])),
    el("article", { class: "stage" }, el("h4", { text: t("rectrace.lastIndexing") }), stats.frames != null ? kv([
      ["framesSampled", stats.frames], ["detections", stats.detections], ["tracksKept", stats.tracks],
      ["shortTracksDropped", stats.dropped_short_tracks], ["indexingTime", t("unit.s", { s: stats.seconds })],
      ["attributes", stats.attributes_source ? t("rectrace.attributesTime", { source: stats.attributes_source, s: stats.attributes_seconds }) : t("value.none")],
      ["detector", models.detector ? t("rectrace.detector", { detector: models.detector, threshold: models.detection_threshold }) : null],
      ["embedder", models.embedder ? `${models.embedder}@${(models.embedder_revision || "").slice(0, 10)}` : null],
      ["visionModel", models.vlm], ["indexedAt", rec.indexed_at]]) : el("p", { class: "muted", text: t("rectrace.noStats") })),
    el("article", { class: "stage" }, el("h4", { text: t("rectrace.rules") }), kv([
      ["sampling", t("rectrace.sampling", { fps: rules.sample_fps })], ["detectorThreshold", rules.detector_threshold],
      ["trackNeeds", t("rectrace.trackNeeds", { n: rules.min_track_detections })], ["cropsPerTrack", rules.crops_per_track],
      ["candidatesKept", rules.pool_capacity],
      ["smallestCrop", t("rectrace.smallestCrop", { person: rules.min_crop_height_px.person, bag: rules.min_crop_height_px.bag })],
      ["quality", rules.quality], ["spacing", rules.spacing]]))));
  out.push(el("h3", { text: t("rectrace.framesTitle") }));
  out.push(el("p", { class: "muted small", text: t("rectrace.framesNote") }));
  if (trace.frames) out.push(el("div", { class: "frames" }, ...trace.frames.frames.map((f) => frameFigure(f, rules.sample_fps))));
  else out.push(el("p", { class: "error", text: t("rectrace.framesUnavailable", { error: trace.frames_error }) }));
  out.push(el("h3", { text: t("rectrace.tracksTitle", { n: trace.tracks.length }) }));
  const onFrame = (at) => runRecordingTrace(null, String(at));
  const rows = trace.tracks.map((tr) => {
    const row = el("tr", { class: "track-row" },
      el("td", {}, thumb(tr.crops[0]?.url)), el("td", { dir: "ltr", text: tr.track_id }), el("td", { text: word("label", tr.label) }),
      el("td", { text: t("unit.secondsRange", { from: tr.t_start.toFixed(1), to: tr.t_end.toFixed(1) }) }), el("td", { text: tr.detections }),
      el("td", { text: tr.crops.length }), el("td", { text: attributeSummary(tr.attributes) }));
    const detailRow = el("tr", { class: "track-detail", hidden: true }, el("td", { colspan: 7 }));
    row.addEventListener("click", () => {
      if (detailRow.hidden && !detailRow.firstChild.firstChild) detailRow.firstChild.append(trackDetails(tr, onFrame));
      detailRow.hidden = !detailRow.hidden;
    });
    return [row, detailRow];
  });
  out.push(el("div", { class: "table-wrap" }, el("table", { class: "trace-table" },
    el("thead", {}, el("tr", {}, ...["", "track", "label", "seen", "detections", "crops", "attributes"].map((h) => el("th", { text: h ? t(`col.${h}`) : "" })))),
    el("tbody", {}, ...rows.flat()))));
  out.push(el("p", { class: "muted small", text: t("rectrace.clickTrack") }));
  return out;
}

// Recordings

let refreshTimer = null;

async function loadRecordings() {
  let recordings;
  try {
    recordings = await api("/api/recordings");
  } catch {
    return;
  }
  $("#recordings-body").replaceChildren(...recordings.map(recordingRow));
  clearTimeout(refreshTimer);
  if (recordings.some((r) => r.status === "queued" || r.status === "indexing")) {
    refreshTimer = setTimeout(() => !$("#tab-recordings").hidden && loadRecordings(), 2000);
  }
}

function recordingRow(r) {
  const row = document.createElement("tr");
  const cells = [r.camera_id, placeText(r.place), fmtTime(r.start_time), fmtDuration(r.duration_s)];
  for (const value of cells) {
    const td = document.createElement("td");
    td.textContent = value;
    td.dir = "auto";  // keeps "2018-03-16 10:00:00" in reading order inside the right-to-left table
    row.append(td);
  }
  const status = document.createElement("td");
  status.className = `state state-${r.status}`;
  status.textContent = word("recordingStatus", r.status);
  if (r.status === "indexing") {
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = r.progress;
    status.append(bar);
  }
  if (r.error) status.title = r.error;
  const tracks = document.createElement("td");
  tracks.textContent = r.track_count ?? "";
  const actions = document.createElement("td");
  actions.className = "row-actions";
  const reindex = el("button", { type: "button", text: t("recordings.reindex"), disabled: r.status === "queued" || r.status === "indexing" });
  reindex.addEventListener("click", async () => {
    await api(`/api/recordings/${r.id}/reindex?actor=${encodeURIComponent(actor())}`, { method: "POST" });
    loadRecordings();
  });
  const del = el("button", { type: "button", text: t("recordings.delete"), disabled: r.status === "indexing" });
  del.addEventListener("click", async () => {
    if (!confirm(t("recordings.confirmDelete", { camera: r.camera_id, start: fmtTime(r.start_time) }))) return;
    try {
      await api(`/api/recordings/${r.id}?actor=${encodeURIComponent(actor())}`, { method: "DELETE" });
    } catch (err) {
      alert(errorText(err));
    }
    loadRecordings();
  });
  actions.append(reindex, del);
  if (TECH) {
    actions.append(el("button", { type: "button", disabled: r.status !== "indexed", onclick: () => openRecordingTrace(r.id), text: t("recordings.trace") }));
  }
  row.append(status, tracks, actions);
  return row;
}

async function openRecordingTrace(id) {
  showTab("hood");
  showHood("recording");
  await loadRecordingChoices();
  $("#rectrace-recording").value = id;
  runRecordingTrace();
}

function upload(event) {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  data.append("actor", actor());
  const bar = $("#upload-progress");
  const status = $("#upload-status");
  bar.hidden = false;
  bar.value = 0;
  status.textContent = t("upload.uploading");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/recordings");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = e.loaded / e.total; };
  xhr.onload = () => {
    bar.hidden = true;
    if (xhr.status === 201) {
      status.textContent = t("upload.done");
      form.reset();
    } else {
      let detail = xhr.responseText;
      try { detail = JSON.parse(xhr.responseText).detail; } catch { /* plain text */ }
      status.textContent = errorText({ message: typeof detail === "string" ? detail : JSON.stringify(detail) });
    }
    loadRecordings();
  };
  xhr.onerror = () => {
    bar.hidden = true;
    status.textContent = t("upload.networkError");
  };
  xhr.send(data);
}

// Audit log

function auditSummary(e) {
  const d = e.details;
  const parts = [];
  if (e.action === "search") {
    parts.push(t("audit.searched", { query: isolate(d.query), outcome: d.error ? t("audit.error", { error: isolate(d.error) }) : t("audit.results", { n: d.result_count }) }));
    if (d.mode) parts.push(word("modeName", d.mode));
    if (d.cameras?.length) parts.push(t("audit.cameras", { cameras: list(d.cameras) }));
  } else if (e.action === "confirm" || e.action === "reject") {
    parts.push(d.track_key, t("audit.query", { query: isolate(d.query) }));
    if (d.note) parts.push(d.note);
  } else if (e.action === "check") {
    parts.push(word("jobStatus", d.status), t("audit.checked", {
      confirmed: d.confirmed ?? 0, unsure: d.unsure ?? 0, rejected: d.rejected ?? 0,
    }), t("unit.ms", { ms: d.took_ms ?? "?" }));
  } else {
    parts.push(...[d.camera_id, d.file, d.recording_id].filter(Boolean));
  }
  if (d.case_id) parts.push(t("audit.case", { id: d.case_id }));
  return parts.join(t("list.join"));
}

async function loadAudit() {
  let entries;
  try {
    entries = await api("/api/audit?limit=300");
  } catch {
    return;
  }
  $("#audit-body").replaceChildren(...entries.map((e) => {
    const row = document.createElement("tr");
    for (const value of [e.id, fmtTime(e.at.replace("+00:00", "")), e.actor, word("action", e.action)]) {
      const td = document.createElement("td");
      td.textContent = value;
      td.dir = "auto";
      row.append(td);
    }
    const summary = el("td", { text: auditSummary(e) });
    if (TECH && (e.action === "search" || e.action === "check") && e.search_id && !e.details.error) {
      summary.append(" ", el("button", { type: "button", class: "link", onclick: () => openTrace(e.search_id), text: t("audit.trace") }));
    }
    row.append(summary);
    return row;
  }));
}

// Health, on the Under the hood tab

async function checkHealth() {
  const badge = $("#health");
  if (!badge) return;
  try {
    const h = await api("/api/health");
    const ok = h.inference?.status === "ok" && h.qdrant === "ok";
    const llm = h.query_model?.status === "ok";
    badge.textContent = t(ok ? (llm ? "health.ready" : "health.noQueryModel") : "health.notReady");
    badge.dataset.ok = String(ok);
    badge.title = JSON.stringify(h);
  } catch {
    badge.textContent = t("health.offline");
    badge.dataset.ok = "false";
  }
}

// Links such as /tech#trace=<search id> open a stored trace, and /tech#recording=<id> a recording trace.
function route() {
  const [key, value] = location.hash.slice(1).split("=");
  if (key === "trace" && value) openTrace(decodeURIComponent(value));
  if (key === "recording" && value) openRecordingTrace(decodeURIComponent(value));
}

// Boot

// Trace links made before the views were split point at /; they belong to the technical view.
if (!TECH && /^#(trace|recording)=/.test(location.hash)) location.replace(`/tech${location.hash}`);

await initLanguage();
if (!TECH) for (const node of document.querySelectorAll("[data-tech]")) node.remove();
translatePage();
const otherLanguage = currentLanguage() === "ar" ? "en" : "ar";
$("#lang-switch").lang = otherLanguage;
$("#lang-switch").addEventListener("click", () => {
  remember("avs.case", $("#case-id").value.trim(), "sessionStorage"); // keep the case number across the reload
  switchLanguage(otherLanguage);
});
document.documentElement.dataset.ready = "";

$("#actor").value = recall("avs.actor");
$("#actor").addEventListener("change", (e) => remember("avs.actor", e.target.value.trim()));
$("#case-id").value = recall("avs.case", "sessionStorage");
remember("avs.case", "", "sessionStorage");
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => showTab(button.dataset.tab));
}
$("#search-form").addEventListener("submit", runSearch);
$("#upload-form").addEventListener("submit", upload);
$("#audit-refresh").addEventListener("click", loadAudit);
renderExamples();
loadCameras();
if (TECH) {
  for (const button of document.querySelectorAll("[data-hood]")) {
    button.addEventListener("click", () => showHood(button.dataset.hood));
  }
  $("#trace-form").addEventListener("submit", runTrace);
  $("#trace-open").addEventListener("submit", (e) => { e.preventDefault(); const id = $("#trace-id").value.trim(); if (id) openTrace(id); });
  $("#rectrace-form").addEventListener("submit", runRecordingTrace);
  window.addEventListener("hashchange", route);
  setInterval(() => { if (!$("#tab-hood").hidden) checkHealth(); }, 15000);
  route();
}
