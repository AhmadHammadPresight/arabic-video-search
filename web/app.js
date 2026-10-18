// Arabic-first UI for the CCTV search PoC. Plain ES module, no build step, no external requests.

const $ = (sel, root = document) => root.querySelector(sel);

// Example queries that have matches in the public starter footage.
const EXAMPLES = [
  "امرأة ترتدي سترة حمراء",
  "شخص يرتدي سترة حمراء بقلنسوة",
  "زلمة لابس بنطلون بني",
  "رجّال يسحب شنطة سفر",
  "طفلة ترتدي قميصاً أبيض وبنطال جينز",
  "شنطة سفر بنية",
  "a person in a red hoodie",
];

const LABELS = {
  person: "شخص · person",
  backpack: "حقيبة ظهر · backpack",
  handbag: "حقيبة يد · handbag",
  suitcase: "حقيبة سفر · suitcase",
};

const STATUS = {
  queued: "في الانتظار · queued",
  indexing: "قيد الفهرسة · indexing",
  indexed: "جاهز · indexed",
  failed: "فشل · failed",
};

const ACTIONS = {
  search: "بحث · search",
  confirm: "تأكيد · confirm",
  reject: "رفض · reject",
  upload: "رفع · upload",
  reindex: "إعادة فهرسة · reindex",
  delete: "حذف · delete",
  check: "تحقق بصري · vision check",
};

// "Why it matched": one mark per result of an attribute comparison.
const MARKS = { match: "✓", near: "≈", unknown: "?", missing: "–", mismatch: "✗" };
const FIELD_NAMES = {
  gender: "الجنس", age_band: "العمر", upper_colour: "اللباس العلوي", upper_garment: "نوع اللباس العلوي",
  lower_colour: "اللباس السفلي", lower_garment: "نوع اللباس السفلي", hair_colour: "لون الشعر",
  hair_length: "طول الشعر", headwear: "غطاء الرأس", carried: "يحمل",
};
const VERDICTS = {
  confirmed: "✓ أكّد النموذج البصري · vision check: confirmed",
  rejected: "✗ رفض النموذج البصري · vision check: rejected",
  unsure: "? غير متأكد · vision check: unsure",
  pending: "… قيد التحقق البصري · vision check pending",
};

const STAGE_ORDER = ["input", "parse", "embed", "filter", "candidates", "attributes", "fusion", "ranking", "check", "final"];
const STAGE_TITLES = {
  input: ["النص المدخل", "Raw text"],
  parse: ["فهم الاستعلام بالنموذج اللغوي", "Query model: fields and English rewrite"],
  embed: ["تمثيل النص بـ SigLIP 2", "SigLIP 2 text embedding"],
  filter: ["التصفية قبل الترتيب", "Filters applied before ranking"],
  candidates: ["المرشحون من Qdrant", "Candidate tracks from Qdrant"],
  attributes: ["مطابقة السمات", "Attribute match per field"],
  fusion: ["دمج الدرجات", "Score fusion"],
  ranking: ["الترتيب الأول", "First ranking (shown at once)"],
  check: ["التحقق البصري", "Vision-model double-check"],
  final: ["الترتيب النهائي", "Final ranking"],
};

// Per-viewer conveniences only; the audit log on the server is the record.
function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage may be blocked */ }
}
function recall(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
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

function bi(ar, en) {
  return [ar, " ", el("span", { class: "en", text: en })];
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
  if (name === "hood") loadRecordingChoices();
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
    text.textContent = `${cam.camera_id} · ${cam.places.join("، ")}`;
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
  status.textContent = "جارٍ البحث… · searching…";
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
      ? " · نموذج الاستعلام غير متاح، عُرضت نتائج الخيار أ · query model unavailable, showing option A" : "";
    status.textContent = data.results.length
      ? `${data.results.length} نتيجة · ${data.results.length} results · ${data.took_ms} ms${fallback}`
      : `لا توجد نتائج · no results${fallback}`;
    if (data.parsed) renderParsed(data);
    const checking = data.check && data.check.status !== "skipped" ? data.check.total : 0;
    const cards = new Map(data.results.map((r) => [r.track_key, resultCard(r, data.search_id, r.rank <= checking)]));
    $("#results").replaceChildren(...cards.values());
    if (data.check && data.check.status !== "skipped") {
      searchCheck = data.search_id;
      followCheck(data.search_id, cards, token);
    }
  } catch (err) {
    status.textContent = `خطأ · error: ${err.message}`;
  }
}

function fieldChips(fields) {
  const chips = [];
  const colours = (part, noun) => `${part.colours.join(" / ")} ${part.garment || noun}`.trim();
  if (fields.subject === "bag") chips.push(["حقيبة", "bag"]);
  if (fields.gender) chips.push(fields.gender === "female" ? ["امرأة", "woman"] : ["رجل", "man"]);
  if (fields.age_band && fields.age_band !== "adult") chips.push(fields.age_band === "child" ? ["طفل", "child"] : ["كبير السن", "elderly"]);
  if (fields.upper.colours.length || fields.upper.garment) chips.push([FIELD_NAMES.upper_colour, colours(fields.upper, "top")]);
  if (fields.lower.colours.length || fields.lower.garment) chips.push([FIELD_NAMES.lower_colour, colours(fields.lower, "trousers")]);
  if (fields.hair.colour || fields.hair.length) chips.push(["الشعر", [fields.hair.length, fields.hair.colour, "hair"].filter(Boolean).join(" ")]);
  if (fields.headwear.type) chips.push([FIELD_NAMES.headwear, `${fields.headwear.colours.join(" / ")} ${fields.headwear.type}`.trim()]);
  for (const item of fields.carried) chips.push([fields.subject === "bag" ? "الحقيبة" : FIELD_NAMES.carried, `${item.colours.join(" / ")} ${item.object}`.trim()]);
  for (const other of fields.other) chips.push(["تفاصيل", other]);
  return chips.map(([ar, en]) => el("span", { class: "chip field-chip" }, ...bi(ar, en)));
}

function renderParsed(data) {
  const box = $("#parsed");
  box.replaceChildren(
    el("div", { class: "parsed-head" }, ...bi("فهمنا من الوصف", "What the query model understood")),
    el("div", { class: "parsed-chips" }, ...fieldChips(data.parsed)),
    el("div", { class: "parsed-english", dir: "ltr" }, "English rewrite: ", el("q", { text: data.parsed.english })),
    el("button", { type: "button", class: "link", onclick: () => openTrace(data.search_id) },
      ...bi("كيف وُجدت هذه النتائج؟", "How were these found? (under the hood)")),
  );
  box.hidden = false;
}

function whyChips(why) {
  return (why || []).map((w) => el("span", {
    class: `why-chip why-${w.result}`,
    title: `${FIELD_NAMES[w.field] || w.field} · ${w.result}`,
    text: `${MARKS[w.result]} ${w.label}`,
  }));
}

function setCheck(card, verdict) {
  const box = $(".check", card);
  if (!verdict) return;
  const label = VERDICTS[verdict.verdict] || verdict.verdict;
  const answers = Object.entries(verdict.answers || {}).map(([detail, answer]) =>
    el("span", { class: `why-chip answer-${answer}`, text: `${detail}: ${answer}` }));
  box.replaceChildren(el("div", { class: `verdict-badge verdict-${verdict.verdict}`, text: label }),
    el("div", { class: "answers" }, ...answers), verdict.note ? el("div", { class: "muted small", text: verdict.note }) : null);
  card.dataset.check = verdict.verdict;
}

async function followCheck(searchId, cards, token) {
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
      if (card && card.dataset.check !== verdict.verdict) setCheck(card, verdict);
    }
    status.textContent = `التحقق البصري ${job.done}/${job.total} · vision double-check ${job.done} of ${job.total}`;
    if (job.status === "done") {
      const order = job.results.map((r) => cards.get(r.track_key)).filter(Boolean);
      job.results.forEach((r) => { const card = cards.get(r.track_key); if (card) $(".rank", card).textContent = `#${r.rank}`; });
      $("#results").replaceChildren(...order);
      const count = (v) => Object.values(job.verdicts).filter((x) => x.verdict === v).length;
      const [yes, no, unsure] = [count("confirmed"), count("rejected"), count("unsure")];
      status.textContent = `اكتمل التحقق البصري: ${yes} مؤكدة، ${no} مرفوضة نزلت إلى الأسفل، ${unsure} غير مؤكدة · `
        + `double-check done: ${yes} confirmed and moved up, ${no} rejected and moved down, ${unsure} unsure · ${job.took_ms} ms`;
      return;
    }
    if (job.status === "failed") {
      status.textContent = `تعذّر التحقق البصري · double-check failed: ${job.error || ""}`;
      return;
    }
    if (job.status === "cancelled" || job.status === "interrupted") {
      status.textContent = `توقّف التحقق البصري بعد ${job.done} من ${job.total}، والترتيب لم يتغير · `
        + `double-check stopped after ${job.done} of ${job.total} (${job.reason || job.status}); the order is unchanged`;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function resultCard(result, searchId, checking) {
  const card = $("#result-card").content.firstElementChild.cloneNode(true);
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
  $(".label-badge", card).textContent = LABELS[result.label] || result.label;
  $(".score", card).textContent = result.similarity == null
    ? `score ${result.score.toFixed(3)}`
    : `score ${result.score.toFixed(2)} · similarity ${result.similarity.toFixed(3)}`;
  $(".place", card).textContent = result.place;
  $(".time", card).textContent = `${fmtTime(result.time)}  (${fmtTime(result.seen_from).slice(11)} – ${fmtTime(result.seen_to).slice(11)})`;
  if (result.why) {
    const why = $(".why", card);
    why.replaceChildren(...whyChips(result.why));
    if (!result.described) why.append(el("span", { class: "muted small", text: "no attributes for this track" }));
  }
  if (checking) $(".check", card).replaceChildren(el("div", { class: "verdict-badge verdict-pending", text: VERDICTS.pending }));
  for (const verdict of ["confirm", "reject"]) {
    $(`.${verdict}`, card).addEventListener("click", async () => {
      try {
        await postJSON("/api/feedback", { search_id: searchId, track_key: result.track_key, verdict, actor: actor() });
        card.dataset.verdict = verdict;
      } catch (err) {
        alert(`خطأ · error: ${err.message}`);
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

async function runTrace(event) {
  event?.preventDefault();
  const query = $("#trace-query").value.trim();
  if (!query) return;
  const token = ++traceToken;
  cancelCheck(traceCheck);
  traceCheck = null;
  const status = $("#trace-status");
  status.textContent = "جارٍ التتبع في الوضعين… · tracing both modes…";
  $("#trace-columns").replaceChildren();
  const body = { query, ...searchFilters(), actor: actor(), case_id: $("#case-id").value.trim() || null };
  try {
    const baseline = await postJSON("/api/search?trace=1", { ...body, mode: "baseline" });
    const hybrid = await postJSON("/api/search?trace=1", { ...body, mode: "hybrid" });
    if (token !== traceToken) return;
    status.textContent = `الخيار أ ${baseline.took_ms} ms · الهجين ${hybrid.took_ms} ms · option A ${baseline.took_ms} ms, hybrid ${hybrid.took_ms} ms (first results)`;
    const columns = [traceColumn(baseline.trace, ["الخيار أ", "Option A baseline"]), traceColumn(hybrid.trace, ["الهجين", "Hybrid (phase 2)"])];
    $("#trace-columns").replaceChildren(...columns);
    if (hybrid.check && hybrid.check.status !== "skipped") {
      traceCheck = hybrid.search_id;
      followTrace(hybrid.search_id, 1, ["الهجين", "Hybrid (phase 2)"], token);
    }
  } catch (err) {
    status.textContent = `خطأ · error: ${err.message}`;
  }
}

async function followTrace(searchId, index, title, token) {
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
    if (old) old.replaceWith(traceColumn(trace, title));
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
  status.textContent = "جارٍ فتح التتبع… · opening the trace…";
  try {
    const trace = await api(`/api/search/${encodeURIComponent(searchId)}/trace`);
    if (token !== traceToken) return;
    const title = trace.mode === "hybrid" ? ["الهجين", "Hybrid (phase 2)"] : ["الخيار أ", "Option A baseline"];
    status.textContent = `بحث محفوظ · stored search ${searchId}`;
    $("#trace-columns").replaceChildren(traceColumn(trace, title));
    const check = trace.stages.find((s) => s.key === "check");
    if (check && ["queued", "running"].includes(check.output.status)) followTrace(searchId, 0, title, token);
  } catch (err) {
    status.textContent = `خطأ · error: ${err.message}`;
  }
}

function thumb(url, size = "small") {
  return url ? el("img", { class: `thumb thumb-${size}`, src: url, alt: "crop", loading: "lazy" }) : el("span");
}

function shortKey(key) {
  return key ? key.split("/").pop() : "";
}

function kv(pairs) {
  return el("dl", { class: "kv" }, ...pairs.filter(([, v]) => v != null && v !== "").flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { dir: "auto" }, v)]));
}

function pre(value) {
  return el("pre", { class: "json", dir: "ltr", text: typeof value === "string" ? value : JSON.stringify(value, null, 2) });
}

function details(summary, ...content) {
  return el("details", {}, el("summary", { text: summary }), ...content);
}

function table(headers, rows) {
  return el("div", { class: "table-wrap" }, el("table", { class: "trace-table", dir: "ltr" },
    el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", { text: h })))),
    el("tbody", {}, ...rows.map((cells) => el("tr", {}, ...cells.map((c) => el("td", {}, c)))))));
}

function trackValue(value) {
  if (value == null) return "nothing";
  if (Array.isArray(value)) {
    if (!value.length) return "nothing";
    return value.map((v) => (typeof v === "object" ? `${(v.colours || []).join("/")} ${v.object}`.trim() : v)).join(", ");
  }
  if (typeof value === "object") return `${(value.colours || []).join("/")} ${value.type || value.object || "none"}`.trim();
  return String(value);
}

function traceColumn(trace, title) {
  const crops = new Map();
  for (const stage of trace.stages) {
    const rows = Array.isArray(stage.output) ? stage.output : stage.output?.rows;
    for (const row of rows || []) if (row.crop_url) crops.set(row.track_key, row.crop_url);
  }
  const models = trace.models || {};
  const query = models.query_model;
  const head = el("header", { class: "trace-head" },
    el("h3", {}, ...bi(title[0], title[1])),
    kv([
      ["search id", trace.search_id],
      ["mode", trace.mode],
      ["first results", trace.took_ms != null ? `${trace.took_ms} ms` : null],
      ["cameras", (trace.request.cameras || ["all"]).join(", ")],
      ["time", trace.request.time_from || trace.request.time_to ? `${trace.request.time_from || "…"} → ${trace.request.time_to || "…"}` : "any"],
      ["embedder", models.embedder],
      ["query model", query ? `${query.model}${query.quantization ? ` (${query.quantization}, ${query.parameters})` : ""} · Ollama ${query.ollama_version || "?"} · ${query.status}` : null],
      ["vision model", models.vision_model],
    ]),
    ...(trace.notes || []).map((n) => el("p", { class: "note", text: n })));
  const stages = STAGE_ORDER.map((key) => {
    const stage = trace.stages.find((s) => s.key === key);
    const [ar, en] = STAGE_TITLES[key];
    const index = STAGE_ORDER.indexOf(key) + 1;
    if (!stage) {
      return el("article", { class: "stage stage-absent" },
        el("h4", {}, `${index}. `, ...bi(ar, en)),
        el("p", { class: "muted small" }, trace.mode === "baseline" ? "ليست جزءاً من الخيار أ · not part of option A" : "لم تُنفّذ بعد · not run yet"));
    }
    return el("article", { class: `stage stage-${key}` },
      el("h4", {}, `${index}. `, ...bi(ar, en), el("span", { class: "ms", text: stage.ms != null ? `${stage.ms} ms` : "" })),
      stageBody(key, stage.output || {}, crops));
  });
  return el("section", { class: "trace-col" }, head, ...stages);
}

function stageBody(key, out, crops) {
  switch (key) {
    case "input":
      return kv([["text", out.text], ["characters", out.characters], ["script", out.script]]);
    case "parse":
      if (out.error) return el("div", {}, el("p", { class: "error", text: out.error }), kv([["fallback", out.fallback]]), out.raw ? pre(out.raw) : null);
      return el("div", {},
        el("div", { class: "parsed-chips" }, ...fieldChips({ ...out.fields, english: out.english })),
        el("p", { class: "parsed-english", dir: "ltr" }, "English rewrite: ", el("q", { text: out.english })),
        kv([["model", out.model], ["model time", out.cached ? "cached (asked before)" : `${out.model_ms} ms`],
          ["tokens in / out", out.prompt_tokens != null ? `${out.prompt_tokens} / ${out.output_tokens}` : null],
          ["load", out.load_ms ? `${out.load_ms} ms` : null]]),
        details("the model's raw JSON", pre(out.raw)));
    case "embed":
      return el("div", {}, kv([["model", out.model], ["text input", out.padding]]),
        ...(out.texts || []).map((t) => el("div", { class: "embed" },
          el("strong", { text: `${t.role}: ` }), el("q", { dir: "auto", text: t.text }),
          el("div", { class: "muted small", dir: "ltr", text: `${t.dim} numbers, length ${t.norm}; first: [${t.first.join(", ")}, …]` }))));
    case "filter":
      return el("div", {}, el("ul", { dir: "ltr" }, ...(out.explain || []).map((line) => el("li", { text: line }))),
        kv([["cameras", Array.isArray(out.cameras) ? out.cameras.join(", ") : out.cameras], ["from", out.time_from], ["to", out.time_to],
          ["labels", Array.isArray(out.labels) ? out.labels.join(", ") : out.labels], ["drop gender", out.exclude_gender]]),
        details("Qdrant filter", pre(out.qdrant_filter)));
    case "candidates": {
      const rows = out.rows || [];
      const sims = rows.length && rows[0].sim_english != null;
      return el("div", {},
        kv([["tracks asked per text", out.pool_per_text], ["found by the English rewrite", out.found_by_english],
          ["found by the original text", out.found_by_original], ["candidate tracks", out.tracks], ["crops scored", out.crops_scored]]),
        table(sims ? ["", "track", "camera", "cos English", "cos original", "similarity"] : ["", "track", "camera", "label", "cosine"],
          rows.slice(0, 20).map((r) => sims
            ? [thumb(r.crop_url), shortKey(r.track_key), r.camera_id, num(r.sim_english), num(r.sim_original), num(r.similarity)]
            : [thumb(r.crop_url), shortKey(r.track_key), r.camera_id, r.label, num(r.similarity)])),
        rows.length > 20 ? el("p", { class: "muted small", text: `showing 20 of ${rows.length} kept in the trace` }) : null);
    }
    case "attributes":
      return el("div", {},
        kv([["fields the query names", (out.query_fields || []).join(", ") || "none"],
          ["tracks with attributes", out.described_tracks], ["values", Object.entries(out.values || {}).map(([k, v]) => `${k} ${v}`).join(", ")]]),
        table(["", "track", "attribute score", "per field (query → track)"], (out.rows || []).slice(0, 15).map((r) => [
          thumb(crops.get(r.track_key)), shortKey(r.track_key), num(r.attribute_score, 2),
          el("div", {}, ...r.matches.map((m) => el("span", { class: `why-chip why-${m.result}`, text: `${MARKS[m.result]} ${m.label} → ${trackValue(m.track)}` }))),
        ])));
    case "fusion":
      return el("div", {}, el("p", { class: "formula", dir: "ltr", text: out.formula }),
        kv([["pool similarity mean", num(out.similarity_mean)], ["pool similarity std", num(out.similarity_std)]]),
        table(["", "track", "similarity", "z", "attribute", "score"], (out.rows || []).slice(0, 15).map((r) => [
          thumb(crops.get(r.track_key)), shortKey(r.track_key), num(r.similarity), num(r.z, 2), num(r.attribute_score, 2), num(r.score, 2)])));
    case "ranking":
      return el("div", { class: "rank-grid" }, ...(Array.isArray(out) ? out : []).map((r) => el("figure", {},
        thumb(r.crop_url, "medium"), el("figcaption", { dir: "ltr" }, `#${r.rank} ${shortKey(r.track_key)} · ${num(r.score, 2)}`,
          r.why ? el("div", {}, ...whyChips(r.why)) : null))));
    case "check": {
      const verdicts = Object.entries(out.verdicts || {});
      return el("div", {},
        kv([["status", out.status], ["checked", `${out.done} of ${out.total}`], ["model", out.model], ["took", out.took_ms ? `${out.took_ms} ms` : null],
          ["details asked", (out.details || []).map((d) => `${d.text}${d.key ? "" : " (advisory)"}`).join("; ")], ["note", out.reason || out.error]]),
        table(["", "track", "verdict", "answers", "note", "ms"], verdicts.map(([key, v]) => [
          el("span", {}, ...(v.crops || []).map((u) => thumb(u))), shortKey(key),
          el("span", { class: `verdict-badge verdict-${v.verdict}`, text: v.verdict }),
          el("div", {}, ...Object.entries(v.answers || {}).map(([d, a]) => el("span", { class: `why-chip answer-${a}`, text: `${d}: ${a}` }))),
          el("span", { class: "small", text: v.note || "" }), v.ms ?? ""])));
    }
    case "final":
      return el("div", { class: "rank-grid" }, ...(Array.isArray(out) ? out : []).map((r) => {
        const moved = r.rank_before_check - r.rank;
        const arrow = moved > 0 ? `↑${moved}` : moved < 0 ? `↓${-moved}` : "=";
        return el("figure", {}, thumb(r.crop_url, "medium"),
          el("figcaption", { dir: "ltr" }, `#${r.rank} (${arrow}) ${shortKey(r.track_key)}`, el("div", { class: `verdict-badge verdict-${r.check.replace(" ", "-")}`, text: r.check })));
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
    el("option", { value: r.id, text: `${r.camera_id} · ${fmtTime(r.start_time)} · ${r.track_count} tracks` })));
  if (current) select.value = current;
}

async function runRecordingTrace(event, at) {
  event?.preventDefault();
  const id = $("#rectrace-recording").value;
  if (!id) return;
  const moments = at ?? $("#rectrace-at").value.trim();
  const status = $("#rectrace-status");
  status.textContent = "جارٍ فك الإطارات وتشغيل الكاشف… · decoding frames and running the detector…";
  try {
    const trace = await api(`/api/recordings/${id}/trace${moments ? `?at=${encodeURIComponent(moments)}` : ""}`);
    status.textContent = "";
    $("#rectrace").replaceChildren(...recordingTrace(trace));
  } catch (err) {
    status.textContent = `خطأ · error: ${err.message}`;
  }
}

function frameFigure(frame, fps) {
  const boxes = (box, cls, label) => el("div", {
    class: `box ${cls}`,
    style: `left:${box[0] * 100}%;top:${box[1] * 100}%;width:${(box[2] - box[0]) * 100}%;height:${(box[3] - box[1]) * 100}%`,
  }, el("span", { text: label }));
  return el("figure", { class: "frame" },
    el("div", { class: "frame-image" }, el("img", { src: frame.url, alt: `frame at ${frame.t} s` }),
      ...frame.detections.map((d) => boxes(d.box, "det", `${d.label} ${d.confidence.toFixed(2)}`)),
      ...frame.tracks.map((t) => boxes(t.box, "trk", t.track_id))),
    el("figcaption", { dir: "ltr" }, `t = ${frame.t.toFixed(2)} s · ${frame.detections.length} detections (${frame.detect_ms} ms) · `
      + `${frame.tracks.length} indexed tracks at this sampled frame (1 frame every ${(1 / fps).toFixed(2)} s)`));
}

function attributeSummary(attrs) {
  if (!attrs) return "–";
  if (!attrs.attributes) return attrs.error ? "unreadable reply" : "–";
  const a = attrs.attributes;
  if (a.subject === "bag") return trackValue(a.carried);
  return [a.gender, a.age_band, a.upper && `${a.upper.colours.join("/")} ${a.upper.garment || ""}`.trim(),
    a.lower && `${a.lower.colours.join("/")} ${a.lower.garment || ""}`.trim(), a.hair?.colour && `${a.hair.colour} hair`,
    a.headwear?.type, a.carried?.length && `carries ${trackValue(a.carried)}`].filter(Boolean).join(" · ");
}

function trackDetails(track, onFrame) {
  const chosen = track.crops.map((c) => el("figure", {}, thumb(c.url, "medium"),
    el("figcaption", { dir: "ltr", text: `t ${c.t.toFixed(2)} s · quality ${num(c.quality)}` })));
  const pool = track.selection?.candidates || [];
  const attrs = track.attributes;
  return el("div", { class: "track-details" },
    el("h5", {}, ...bi("اللقطات المختارة", "The chosen crops")), el("div", { class: "rank-grid" }, ...chosen),
    track.crops.length ? el("button", { type: "button", onclick: () => onFrame(track.crops[0].t) },
      ...bi("اعرض الإطار عند أول لقطة", "Show the frame at the first crop")) : null,
    el("h5", {}, ...bi("لماذا اختيرت", "Why these crops")),
    track.selection ? el("p", { class: "muted small", dir: "ltr", text:
      `${track.selection.detections} detections in the track; the best ${track.selection.pool_capacity} by quality are kept as candidates; `
      + `min_gap ${track.selection.min_gap} s; detector labels ${JSON.stringify(track.selection.labels)}` }) : el("p", { class: "muted", text: "no selection record (indexed before phase 2)" }),
    pool.length ? table(["t (s)", "quality", "confidence", "size", "edge", "picked", "reason"], pool.map((c) => [
      c.t.toFixed(2), num(c.quality), num(c.confidence), num(c.size), c.edge, c.chosen ? "✓" : "", c.reason])) : null,
    el("h5", {}, ...bi("السمات المرفقة", "Attributes attached")),
    attrs ? el("div", {}, kv([["source", attrs.source], ["time", attrs.ms != null ? `${attrs.ms} ms` : null],
      ["summary", attributeSummary(attrs)], ["sentence", attrs.attributes?.description], ["error", attrs.error]]),
      details("attributes JSON", pre(attrs.attributes)), attrs.raw ? details("the vision model's raw reply", pre(attrs.raw)) : null,
      attrs.probabilities ? details("SigLIP 2 prompt probabilities", pre(attrs.probabilities)) : null)
      : el("p", { class: "muted", text: "no attributes (indexed without the attribute step)" }));
}

function recordingTrace(trace) {
  const rec = trace.recording;
  const stats = trace.stats || {};
  const rules = trace.rules;
  const models = stats.models || {};
  const out = [];
  out.push(el("div", { class: "cards-row" },
    el("article", { class: "stage" }, el("h4", {}, ...bi("التسجيل", "Recording")), kv([
      ["camera", rec.camera_id], ["place", rec.place], ["start (local)", fmtTime(rec.start_time)], ["length", fmtDuration(rec.duration_s)],
      ["size", `${rec.width}x${rec.height}`], ["video", stats.video ? `${stats.video.fps?.toFixed(1)} fps` : null], ["file", rec.original_name]])),
    el("article", { class: "stage" }, el("h4", {}, ...bi("آخر فهرسة", "Last indexing")), stats.frames != null ? kv([
      ["frames sampled", stats.frames], ["detections", stats.detections], ["tracks kept", stats.tracks],
      ["short tracks dropped", stats.dropped_short_tracks], ["indexing time", `${stats.seconds} s`],
      ["attributes", stats.attributes_source ? `${stats.attributes_source}, ${stats.attributes_seconds} s` : "none"],
      ["detector", models.detector ? `${models.detector}, threshold ${models.detection_threshold}` : null],
      ["embedder", models.embedder ? `${models.embedder}@${(models.embedder_revision || "").slice(0, 10)}` : null],
      ["vision model", models.vlm], ["indexed at", rec.indexed_at]]) : el("p", { class: "muted", text: "no stats (indexed before phase 2)" })),
    el("article", { class: "stage" }, el("h4", {}, ...bi("القواعد", "Rules")), kv([
      ["sampling", `${rules.sample_fps} frames per second`], ["detector threshold", rules.detector_threshold],
      ["a track needs", `${rules.min_track_detections} detections`], ["crops per track", rules.crops_per_track],
      ["candidates kept", rules.pool_capacity], ["smallest crop", `${rules.min_crop_height_px.person} px person, ${rules.min_crop_height_px.bag} px bag`],
      ["quality", rules.quality], ["spacing", rules.spacing]]))));
  out.push(el("h3", {}, ...bi("إطارات مأخوذة وصناديق الكشف", "Sampled frames with detection boxes")));
  out.push(el("p", { class: "muted small" }, ...bi("الأزرق: ما يراه الكاشف الآن على هذا الإطار. الأصفر المتقطع: المسارات المحفوظة عند هذه اللحظة.",
    "Solid blue: what the detector finds on this frame now. Dashed yellow: the indexed tracks at this moment.")));
  if (trace.frames) out.push(el("div", { class: "frames" }, ...trace.frames.frames.map((f) => frameFigure(f, rules.sample_fps))));
  else out.push(el("p", { class: "error", text: `frames unavailable: ${trace.frames_error}` }));
  out.push(el("h3", {}, ...bi(`المسارات (${trace.tracks.length})`, `Tracks found (${trace.tracks.length})`)));
  const onFrame = (t) => runRecordingTrace(null, String(t));
  const rows = trace.tracks.map((t) => {
    const row = el("tr", { class: "track-row" },
      el("td", {}, thumb(t.crops[0]?.url)), el("td", { text: t.track_id }), el("td", { text: t.label }),
      el("td", { dir: "ltr", text: `${t.t_start.toFixed(1)}–${t.t_end.toFixed(1)} s` }), el("td", { text: t.detections }),
      el("td", { text: t.crops.length }), el("td", { dir: "ltr", text: attributeSummary(t.attributes) }));
    const detailRow = el("tr", { class: "track-detail", hidden: true }, el("td", { colspan: 7 }));
    row.addEventListener("click", () => {
      if (detailRow.hidden && !detailRow.firstChild.firstChild) detailRow.firstChild.append(trackDetails(t, onFrame));
      detailRow.hidden = !detailRow.hidden;
    });
    return [row, detailRow];
  });
  out.push(el("div", { class: "table-wrap" }, el("table", { class: "trace-table" },
    el("thead", {}, el("tr", {}, ...["", "track", "label", "seen", "detections", "crops", "attributes"].map((h) => el("th", { text: h })))),
    el("tbody", {}, ...rows.flat()))));
  out.push(el("p", { class: "muted small" }, ...bi("انقر على مسار لرؤية لقطاته وسبب اختيارها وسماته.", "Click a track to see its crops, why they were picked, and its attributes.")));
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
  const cells = [r.camera_id, r.place, fmtTime(r.start_time), fmtDuration(r.duration_s)];
  for (const value of cells) {
    const td = document.createElement("td");
    td.textContent = value;
    td.dir = "auto";  // keeps "2018-03-16 10:00:00" in reading order inside the right-to-left table
    row.append(td);
  }
  const status = document.createElement("td");
  status.className = `state state-${r.status}`;
  status.textContent = STATUS[r.status] || r.status;
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
  const reindex = document.createElement("button");
  reindex.type = "button";
  reindex.textContent = "إعادة فهرسة · reindex";
  reindex.disabled = r.status === "queued" || r.status === "indexing";
  reindex.addEventListener("click", async () => {
    await api(`/api/recordings/${r.id}/reindex?actor=${encodeURIComponent(actor())}`, { method: "POST" });
    loadRecordings();
  });
  const del = document.createElement("button");
  del.type = "button";
  del.textContent = "حذف · delete";
  del.disabled = r.status === "indexing";
  del.addEventListener("click", async () => {
    if (!confirm(`حذف التسجيل وكل ما استُخرج منه؟ Delete ${r.camera_id} ${fmtTime(r.start_time)} and its index?`)) return;
    try {
      await api(`/api/recordings/${r.id}?actor=${encodeURIComponent(actor())}`, { method: "DELETE" });
    } catch (err) {
      alert(`خطأ · error: ${err.message}`);
    }
    loadRecordings();
  });
  const how = el("button", { type: "button", disabled: r.status !== "indexed", onclick: () => openRecordingTrace(r.id) }, "كيف فُهرس · trace");
  actions.append(reindex, del, how);
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
  status.textContent = "جارٍ الرفع… · uploading…";
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/recordings");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = e.loaded / e.total; };
  xhr.onload = () => {
    bar.hidden = true;
    if (xhr.status === 201) {
      status.textContent = "تم الرفع وأُضيف إلى قائمة الفهرسة · uploaded and queued for indexing";
      form.reset();
    } else {
      let detail = xhr.responseText;
      try { detail = JSON.parse(xhr.responseText).detail; } catch { /* plain text */ }
      status.textContent = `خطأ · error: ${detail}`;
    }
    loadRecordings();
  };
  xhr.onerror = () => {
    bar.hidden = true;
    status.textContent = "خطأ في الاتصال · network error";
  };
  xhr.send(data);
}

// Audit log

async function loadAudit() {
  let entries;
  try {
    entries = await api("/api/audit?limit=300");
  } catch {
    return;
  }
  $("#audit-body").replaceChildren(...entries.map((e) => {
    const row = document.createElement("tr");
    const d = e.details;
    let summary;
    if (e.action === "search") {
      summary = `«${d.query}» → ${d.error ? `error: ${d.error}` : `${d.result_count} results`}`;
      if (d.mode) summary += ` · ${d.mode}`;
      if (d.cameras?.length) summary += ` · cameras ${d.cameras.join(", ")}`;
    } else if (e.action === "confirm" || e.action === "reject") {
      summary = `${d.track_key} · «${d.query}»${d.note ? ` · ${d.note}` : ""}`;
    } else if (e.action === "check") {
      summary = `${d.status} · ${d.confirmed ?? 0} confirmed, ${d.unsure ?? 0} unsure, ${d.rejected ?? 0} rejected · ${d.took_ms ?? "?"} ms`;
    } else {
      summary = [d.camera_id, d.file, d.recording_id].filter(Boolean).join(" · ");
    }
    if (d.case_id) summary += ` · case ${d.case_id}`;
    for (const value of [e.id, fmtTime(e.at.replace("+00:00", "")), e.actor, ACTIONS[e.action] || e.action, summary]) {
      const td = document.createElement("td");
      td.textContent = value;
      td.dir = "auto";
      row.append(td);
    }
    if ((e.action === "search" || e.action === "check") && e.search_id && !d.error) {
      row.lastChild.append(" ", el("button", { type: "button", class: "link", onclick: () => openTrace(e.search_id) }, "تتبع · trace"));
    }
    return row;
  }));
}

// Health

async function checkHealth() {
  const badge = $("#health");
  try {
    const h = await api("/api/health");
    const ok = h.inference?.status === "ok" && h.qdrant === "ok";
    const llm = h.query_model?.status === "ok";
    badge.textContent = ok ? (llm ? "● متصل · ready" : "● متصل، بدون نموذج الاستعلام · ready, no query model") : "● غير جاهز · not ready";
    badge.dataset.ok = String(ok);
    badge.title = JSON.stringify(h);
  } catch {
    badge.textContent = "● غير متصل · offline";
    badge.dataset.ok = "false";
  }
}

// Links such as #trace=<search id> open a stored trace, and #recording=<id> a recording trace.
function route() {
  const [key, value] = location.hash.slice(1).split("=");
  if (key === "trace" && value) openTrace(decodeURIComponent(value));
  if (key === "recording" && value) openRecordingTrace(decodeURIComponent(value));
}

// Boot

$("#actor").value = recall("avs.actor");
$("#actor").addEventListener("change", (e) => remember("avs.actor", e.target.value.trim()));
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => showTab(button.dataset.tab));
}
for (const button of document.querySelectorAll("[data-hood]")) {
  button.addEventListener("click", () => showHood(button.dataset.hood));
}
$("#search-form").addEventListener("submit", runSearch);
$("#upload-form").addEventListener("submit", upload);
$("#audit-refresh").addEventListener("click", loadAudit);
$("#trace-form").addEventListener("submit", runTrace);
$("#trace-open").addEventListener("submit", (e) => { e.preventDefault(); const id = $("#trace-id").value.trim(); if (id) openTrace(id); });
$("#rectrace-form").addEventListener("submit", runRecordingTrace);
window.addEventListener("hashchange", route);
renderExamples();
loadCameras();
checkHealth();
setInterval(checkHealth, 15000);
route();
