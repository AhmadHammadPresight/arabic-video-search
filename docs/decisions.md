# Decisions

This file records the choices behind the PoC: the seven decisions the captain settled before the build, the phased plan, and the choices made while building phases 0 and 1.

## The seven settled decisions

Settled between 15 and 17 October 2026; the captain's own words are quoted where they were given.

### D1. How Arabic text is matched to video

Hybrid, built in steps, with option A first as the baseline.

- Option A (this build): the Arabic query goes straight to SigLIP 2's multilingual text encoder and is ranked against person and bag crops by cosine similarity.
- The full hybrid (phase 2): a local language model turns the query, dialect included, into fields plus an English description; attribute and place/time filters narrow the candidates; embeddings rank them; a vision model double-checks the top results; a person confirms.

### D2. Footage for the demo

Public starter set now (about 10 clips: MEVA, NVIDIA synthetic, CAVIAR); a small consented recording later only if the demo needs children or Arabic dress.

- On 17 October 2026 the captain added: "feel free to use free cctv clips few ones".
  Two Pexels clips were added for neutral searches only; their people are never a suspect or part of a demo story, because the Pexels License forbids showing identifiable people in a bad light.
- Left out: VIRAT (no part may go to third parties), Mixkit (its use policy bans AI use), and every CC BY-NC dataset.

### D3. Where the PoC runs

"hmm for this demo, we will dockerize things of course and this on this Mac m4 48 that I currently run locally, however in production and further plan that's a different thing, so third option".

Amended on 17 October 2026: "btw one thnig it's fine to not use docker, i just need it to work and to be reproducible, bash script is enough to run service/ whatever".

- The PoC runs natively on the Mac from bash scripts (`scripts/setup.sh`, `scripts/start.sh`), with every version pinned.
- Containers on macOS get no Apple GPU, so the inference service would have been native in any case.
- Production hardware is a separate plan: the same inference service on an NVIDIA GPU, with the test set re-run after the move.

### D4. The model that reads the Arabic query

Local only, no hosted models even for testing.

- Phase 1 has no query model at all: SigLIP 2 reads the text directly.
- Phase 2 picks a local model by a dialect test; the shortlist is Gemma 4 26B-A4B first, then Gemma 4 31B, Qwen3.6-35B-A3B and Jais 3 32B.

### D5. Face recognition

"out of scope of course, this could come later or never, we need to search people based on appearance".

No face detection, recognition or face embeddings anywhere; search is by appearance only.

### D6. The demo's stories

All three, as search stories: a suspect from a witness description, a left bag, and a lost child with a stand-in.

- The starter footage has no real children.
  The lost-child story uses a child-sized synthetic figure from the NVIDIA store scene, which raises no privacy question; a consented recording can replace it later.
- There is no automatic left-bag alert; the left bag is found by searching for it.

### D7. Live cameras

"we don't need any live camera, we either pre-upload these camera based on dataset of cctv footages".

Only uploaded recordings are indexed; there are no live streams.

## Phased plan

- Phase 0, set-up (done in this build): the Mac set up from pinned scripts, the starter clips fetched and trimmed with attribution, and the ground truth written for the seeded queries.
- Phase 1, option A baseline (done in this build): upload, indexing, Arabic search with camera and time filters, the results grid with confirm and reject, the audit log, and baseline scores (see the README).
- Phase 2, the Arabic hybrid:
  - a local Arabic query model chosen by a dialect test (formal, Jordanian, Gulf and mixed Arabic-English queries, scored on valid output, per-field accuracy, a native speaker's rating of the English rewrite, and speed and memory on the Mac);
  - attribute filters (clothing colours, bag type, and other fields the query model extracts);
  - the vision-model double-check of the top candidates (Qwen3-VL-8B, Apache-2.0, run locally);
  - the next-sighting timeline across cameras.
- Phase 3, demo-ready: the three stories rehearsed end to end, speed tuning, and the success measures checked on the demo clips.

## Choices made while building phases 0 and 1

### Stack

- Detection: RF-DETR Medium (Apache-2.0), COCO classes person, backpack, handbag and suitcase, confidence threshold 0.35.
  The RF-DETR XL and 2XL weights are under the PML licence and are refused by the code.
- Tracking: Roboflow `trackers` ByteTrack (Apache-2.0), one tracker for people and one shared by the three bag classes, because the detector often flips between bag classes for the same object.
- Embedding: `google/siglip2-so400m-patch16-384` (Apache-2.0) at a pinned Hugging Face revision, in float16 on the Apple GPU (PyTorch MPS).
- Vector index: the Qdrant v1.19.2 macOS binary, checksum-pinned, listening on 127.0.0.1 with telemetry disabled.
- Database, job queue and audit log: one SQLite file in WAL mode.
  For a single-Mac PoC this removes a server process; the schema is plain SQL and can move to PostgreSQL for a pilot.
- Web UI: plain HTML, CSS and JavaScript with no build step and no external requests, served by the same FastAPI process as the API.
- Python 3.12 and every dependency are pinned by `uv.lock`; `scripts/setup.sh` installs with `uv sync --frozen`.

### Indexing

- Frames are sampled at 4 per second (the plan allows 2-5).
- A track needs at least 3 detections; up to 5 crops are kept per track, picked by detector confidence, box height (more detail) and whether the box touches the frame edge, and spread over the track in time.
- Each crop is one point in Qdrant, carrying the camera, place, absolute times and a pointer back to its recording; search results are grouped by track, so a person appears once with their best-matching crop.
- Each upload gets a browser-playable copy: a lossless remux when it is already H.264, otherwise a VideoToolbox H.264 encode; audio is always dropped.
- Camera times are the site's local wall-clock time, stored without a UTC offset.

### Evaluation

- Ground truth is written as anchors (camera, local time, a point on the target) rather than track ids, so it survives re-indexing with other settings.
- Every target was picked by eye from contact sheets of all indexed tracks, not from search results.
- Each query carries the camera and time filter an officer would use, which keeps the ground truth complete within that scope.

### Licences of tools outside the Python stack

- FFmpeg from Homebrew is a GPL build; the PoC runs it as a separate program and only encodes with Apple's VideoToolbox encoder, never with libx264.
  A product build should ship an LGPL FFmpeg.
- certifi and tqdm (MPL-2.0) are pulled in by other libraries and used unmodified.

### Privacy controls in the PoC

- Every service binds to 127.0.0.1; model weights are downloaded once by `scripts/setup.sh`, and the services run with the Hugging Face hub offline.
- Every upload, search, confirm, reject, re-index and delete is written to an append-only audit log; SQLite triggers reject any UPDATE or DELETE on it.
- Deleting a recording removes its video, playback copy, crops, tracks and vectors.
- Only playback copies and crops are served over HTTP; uploads, models and the database are not.

## Findings to carry into phase 2

- Formal Arabic queries reach the person far more often than Jordanian or Gulf phrasings of the same description (see the README baseline); dialect handling is the main gap, as the plan expected.
- The captain's own Levantine query for the grey-jacket woman missed, while a formal rewrite of the same description ranked her first.
- Small held objects, such as the iPad in the waiting-room clip, are not found by crop embeddings in any language; this needs the vision-model double-check or attributes.
- At CAVIAR's 384x288 resolution no bag was detected, so the left-bag story relies on the MEVA waiting-room clips.
- The waiting-room clip the captain found laggy (G331, 7 March 17:00) has a constant 30 fps, monotonic timestamps and no repeated frames; the lag comes from playing a 1080p AVI with a keyframe every 2 seconds, and the PoC plays an H.264 MP4 copy instead.
