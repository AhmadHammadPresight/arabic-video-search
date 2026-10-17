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

async function runSearch(event) {
  event?.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  const cameras = [...document.querySelectorAll("#camera-filter input:checked")].map((i) => i.value);
  const status = $("#search-status");
  status.textContent = "جارٍ البحث… · searching…";
  $("#results").replaceChildren();
  try {
    const data = await postJSON("/api/search", {
      query,
      cameras: cameras.length ? cameras : null,
      time_from: $("#time-from").value || null,
      time_to: $("#time-to").value || null,
      actor: actor(),
      case_id: $("#case-id").value.trim() || null,
    });
    status.textContent = data.results.length
      ? `${data.results.length} نتيجة مرتبة حسب التشابه · ${data.results.length} results ranked by similarity · ${data.took_ms} ms`
      : "لا توجد نتائج · no results";
    $("#results").replaceChildren(...data.results.map((r) => resultCard(r, data.search_id)));
  } catch (err) {
    status.textContent = `خطأ · error: ${err.message}`;
  }
}

function resultCard(result, searchId) {
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
  $(".score", card).textContent = `score ${result.score.toFixed(3)}`;
  $(".place", card).textContent = result.place;
  $(".time", card).textContent = `${fmtTime(result.time)}  (${fmtTime(result.seen_from).slice(11)} – ${fmtTime(result.seen_to).slice(11)})`;
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
  actions.append(reindex, del);
  row.append(status, tracks, actions);
  return row;
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
      if (d.cameras?.length) summary += ` · cameras ${d.cameras.join(", ")}`;
    } else if (e.action === "confirm" || e.action === "reject") {
      summary = `${d.track_key} · «${d.query}»${d.note ? ` · ${d.note}` : ""}`;
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
    return row;
  }));
}

// Health

async function checkHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    const ok = h.inference?.status === "ok" && h.qdrant === "ok";
    el.textContent = ok ? "● متصل · ready" : "● غير جاهز · not ready";
    el.dataset.ok = String(ok);
    el.title = JSON.stringify(h);
  } catch {
    el.textContent = "● غير متصل · offline";
    el.dataset.ok = "false";
  }
}

// Boot

$("#actor").value = recall("avs.actor");
$("#actor").addEventListener("change", (e) => remember("avs.actor", e.target.value.trim()));
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => showTab(button.dataset.tab));
}
$("#search-form").addEventListener("submit", runSearch);
$("#upload-form").addEventListener("submit", upload);
$("#audit-refresh").addEventListener("click", loadAudit);
renderExamples();
loadCameras();
checkHealth();
setInterval(checkHealth, 15000);
