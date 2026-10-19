# Decisions

This file records the choices behind the PoC: the seven decisions the project owner settled before the build, the phased plan, and the choices made while building phases 0, 1 and 2.

## The seven settled decisions

Settled between 15 and 17 October 2026.

### D1. How Arabic text is matched to video

Hybrid, built in steps, with option A first as the baseline.

- Option A (this build): the Arabic query goes straight to SigLIP 2's multilingual text encoder and is ranked against person and bag crops by cosine similarity.
- The full hybrid (phase 2): a local language model turns the query, dialect included, into fields plus an English description; attribute and place/time filters narrow the candidates; embeddings rank them; a vision model double-checks the top results; a person confirms.

### D2. Footage for the demo

Public starter set now (about 10 clips: MEVA, NVIDIA synthetic, CAVIAR); a small consented recording later only if the demo needs children or Arabic dress.

- On 17 October 2026 the project owner approved adding a few free CCTV clips.
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
- Phase 2, the Arabic hybrid (done in this build, on its own branch; see "Choices made while building phase 2"):
  - a local Arabic query model chosen by a dialect test (formal, Jordanian, Gulf and mixed Arabic-English queries, scored on valid output, per-field accuracy, a native speaker's rating of the English rewrite, and speed and memory on the Mac);
  - attribute filters (clothing colours, bag type, and other fields the query model extracts);
  - the vision-model double-check of the top candidates (Qwen3-VL-8B, Apache-2.0, run locally);
  - an "under the hood" page that traces every search and every indexed recording, added at the project owner's request during phase 2;
  - the next-sighting timeline across cameras moved to phase 3 at the phase 2 go-ahead.
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
- The project owner's own Levantine query for the grey-jacket woman missed, while a formal rewrite of the same description ranked her first.
- Small held objects, such as the iPad in the waiting-room clip, are not found by crop embeddings in any language; this needs the vision-model double-check or attributes.
- At CAVIAR's 384x288 resolution no bag was detected, so the left-bag story relies on the MEVA waiting-room clips.
- The waiting-room clip the project owner found laggy (G331, 7 March 17:00) has a constant 30 fps, monotonic timestamps and no repeated frames; the lag comes from playing a 1080p AVI with a keyframe every 2 seconds, and the PoC plays an H.264 MP4 copy instead.

## Choices made while building phase 2

On 17 October 2026 the project owner approved starting phase 2 on a separate branch, with the changes kept local.
During the build the project owner asked for a page that shows each search step by step, which also serves for debugging.

### The query model server: Ollama

- Ollama v0.40.6 (MIT), the official macOS build, checksum-pinned and unpacked into `data/bin/ollama` by `scripts/setup.sh`, like Qdrant.
- Chosen over LM Studio and mlx-lm for robustness: it runs headless from a script, it constrains decoding to a JSON schema (so every reply is valid JSON with known values), it serves the GGUF builds of all four candidates, and its model store is shared and content-addressed.
  LM Studio is a closed-source desktop app that a bash script cannot install cleanly; `mlx_lm.server` has no grammar-constrained output.
- `scripts/start.sh` runs it on 127.0.0.1 with `OLLAMA_NO_CLOUD=1` (no remote inference, no web search, no model-recommendation fetch), `OLLAMA_NOPRUNE=1` (it never deletes anything from the shared model store), a 4096-token context and the model kept loaded.
  Only `scripts/setup.sh` and the comparison tool pull models; at run time Ollama makes no outbound calls.
- Models live in Ollama's standard store (`~/.ollama/models`, or `AVS_OLLAMA_MODELS`), next to the Hugging Face cache that already holds the vision models.

### The query schema and prompt

- One fixed schema (`src/avs/fields.py`) is shared by the query model, the per-track attributes and the double-check: subject (person or bag), gender, age band, upper and lower garment with colours, hair colour and length, headwear, carried objects, other details, and the English rewrite.
  Colours, garments, headwear and objects come from small fixed lists, so an attribute match is a plain comparison.
- The prompt states the rules both sides follow (a dress or suit is the upper garment, shades fold into twelve basic colours, a bag searched for on its own is the one carried entry) and gives two worked examples written for the prompt, not taken from the test set.
- It also lists a short glossary of dialect words that are easy to misread, such as سكني (Levantine grey), رجّال (Gulf: one man, not "men"), حرمة and مرة (a woman).
  This is general dialect knowledge rather than the test set's answers, but it overlaps with words the test set uses, so the comparison was also run without it (see the README).
- The parser validates every reply against the schema; a reply that does not fit is an error, never a guess, and the search then falls back to option A and says so.

### Choosing the query model

- The candidates were the four named at the go-ahead (Gemma 4 26B-A4B first, Qwen3.6-35B-A3B, Jais 3 32B, Fanar 2 27B), all as Q4_K_M GGUF builds in the same Ollama with the production prompt; Gemma 4 31B was left out at the go-ahead.
- The rule, set before measuring: the fastest model (median seconds per query) within 2 points of the best field accuracy, scored on all eleven fields of the 113 test queries with `python -m avs.tools.compare_query_models`.
- Each model's run stops at the first query that takes longer than 20 s, four times the 5 s target for a whole search, and the model is recorded as too slow with the time it measured; the report is saved after every query.

| Model | Field accuracy | Median per query | Memory while loaded | Outcome |
|---|---|---|---|---|
| Gemma 4 26B-A4B | 99.4% | 1.9 s | 18.1 GB | chosen |
| Qwen3.6-35B-A3B | 99.3% | 2.4 s | 23.1 GB | within 2 points, slower |
| Fanar 2 27B | 98.7% | 8.3 s | 18.9 GB | within 2 points, four times slower |
| Jais 3 32B | 99.9% on the 61 queries it finished | 64 s | 22.4 GB | deferred to a later phase |

- Gemma 4 26B-A4B (Apache-2.0, `gemma4:26b-a4b-it-q4_K_M`) is chosen: it is the most accurate and the fastest, so no other model could win under the rule.
  The two mixture-of-experts models (Gemma 4 with about 4B active parameters, Qwen3.6 with about 3B) answer in about 2 s; the dense ones need every one of their 27B to 35B parameters per token.
- Jais 3 32B loads and runs in Ollama 0.40.6 and got its fields right, but took 37 to 268 s per query (median 64 s on the first 61 queries), and under the 20 s limit its first query does not finish.
  On 17 October 2026 the project owner put Jais on hold for a possible later phase.
  It is deferred to a later phase with that measurement as the reason, not replaced.
- Fanar 2 27B as published does not load in Ollama 0.40.6: its chat template uses `{% raw %}` around the tool-calling text, which the template engine in Ollama's llama.cpp runner rejects.
  Overriding the template in a Modelfile does not help, because the runner still parses the one inside the GGUF.
  `python -m avs.tools.patch_gguf_template` writes a copy with the two tags replaced by same-length template comments (the rendered prompt is identical, the weights untouched), and that copy was measured in a separate Ollama model store.
  It scored 98.7% at a median of 8.3 s per query: within 2 points of Gemma 4 but four times slower, and over the 5 s target on its own, so the pick does not change.
- Its licence note is under "Licences of the phase 2 models and packages" below; it would only matter if Fanar 2 were chosen.
- Ollama's `/api/ps` reports only 1.2 GB for Gemma 4 because the weights are mapped in 33 parts and it counts one, so memory here is added up from Ollama's load log for every model alike.
- The glossary in the prompt overlaps words the test set uses, so the two fastest models were also run without it: Gemma 4 scored 99.5% (99.4% with it) and Qwen3.6 98.5% (99.3% with it).
  The glossary stays, since it helped Qwen3.6 and costs Gemma 4 nothing, but the chosen model does not depend on it.

### Attributes per track: the vision model

- Two ways were measured on 81 tracks labelled by eye (a fixed random sample across the MEVA and NVIDIA cameras, `eval/attributes.yaml`), using `python -m avs.tools.attribute_benchmark`:

| Field | Labelled tracks | SigLIP 2 zero-shot prompts | Qwen3-VL-8B description |
|---|---|---|---|
| Gender | 52 | 90.4% | 96.2% |
| Upper garment colour | 65 | 72.3% | 92.3% |
| Lower garment colour | 55 | 38.2% | 85.5% |
| Hair colour | 30 | 86.7% | 86.7% |
| Headwear present | 54 | 53.7% | 77.8% |
| Carries a bag | 49 | 51.0% | 95.9% |
| Bag colour | 8 | 75.0% | 100.0% |
| All fields | | 64.2% | 89.5% |
| Time per track | | 22 ms | 4.1 s |

- The vision model is chosen: it is right 25 points more often, and the gap is largest on exactly what the zero-shot prompts mis-bind (the colour of the trousers rather than the jacket, whether a bag is carried).
  It adds about 4 seconds per track, 36.5 minutes for the 486 starter tracks in the clean run; the plan expected 30-40 minutes with descriptions.
- It looks at the track's two best crops and fills the shared schema through grammar-constrained JSON (llguidance in mlx-vlm), so the output always parses.
  The free-text `other` list is capped at four short phrases; without the cap the model sometimes repeated phrases until it hit its token limit (5 of 486 tracks), and the cut-off reply could not be read.
  Real replies use three or fewer, so the cap changes nothing else.
  A reply that still cannot be read leaves the track without attributes, so search treats its fields as unknown rather than as "not seen".
  For a bag track the detector's class (backpack, handbag, suitcase) is kept and the model gives the colours.
- `AVS_ATTRIBUTES=siglip` switches to the zero-shot prompts and `none` turns attributes off; `scripts/reindex_all.sh` re-indexes every existing recording with whichever is set.
- Attributes ride on every crop point in Qdrant (`attributes.gender` is indexed for filtering) and are stored with the track in SQLite, with the model's raw reply for debugging.

### The hybrid search

- The query model's English rewrite and the original text are both embedded by SigLIP 2; each candidate track's similarity is a weighted mean of its best cosine to each, three parts English to one part original.
- Before ranking, Qdrant filters by camera and time (as in option A) and keeps only the detector labels of the subject (people, or the three bag classes).
  Gender is matched softly with the other fields rather than filtered: the vision model described 6 of the 75 target tracks (8%) with the wrong gender, mostly older women in coats and hats, and a filter would hide them completely.
- Every named field is then compared with the track's attributes: match, near (beige for brown, a coat for a jacket), unknown, missing (a named object the track does not show) or mismatch.
  The weighted mean of those is the attribute score, from -1 to 1.
- The fused score is the similarity standardised within the candidate pool plus the attribute score times a weight; each result shows its per-field outcome ("why it matched").

- The three settings were measured on the 47 scored queries of the phase 1 set only, before the 65 phase 2 queries were written, with Gemma 4 parsing and the first ranking (no double-check); option A scores 0.85 top-10 hit rate, 0.48 precision at 5 and 0.79 MRR on the same queries:

| English share | Attribute weight | Gender | Top-10 hit rate | Precision at 5 | MRR |
|---|---|---|---|---|---|
| 0.5 | 1.0 | soft | 0.94 | 0.57 | 0.84 |
| 0.75 | 0 (similarity only) | soft | 1.00 | 0.59 | 0.86 |
| 0.75 | 1.0 | soft | 1.00 | 0.60 | 0.88 |
| 0.75 | 1.0 | filtered | 1.00 | 0.59 | 0.88 |
| 0.75 | 2.0 | soft | 1.00 | 0.59 | 0.89 |
| 1.0 (English only) | 1.0 | soft | 1.00 | 0.61 | 0.89 |

- The English rewrite matters most: giving the original dialect text half the weight loses three targets from the top 10.
  English only was marginally better than three-to-one (within one query's difference); three-to-one keeps the original text in play, as planned, in case a rewrite drops a detail.
- The attribute weight gives a steady gain from 0 to 1 and then a plateau up to 2, so the value chosen before measuring (1.0) stays.
- Filtering by gender never helped and lowered precision slightly, so gender stays a soft field.
- The 30 settings measured (English share 0.5, 0.75 or 1; attribute weight 0 to 2 in steps of 0.5; gender soft or filtered) ranged from 0.94 / 0.56 / 0.83 (half English, no attributes) to 1.00 / 0.61 / 0.89; every one of them beat option A on all three measures.
  The README reports the held-out phase 2 queries separately.

- Option A stays selectable (`"mode": "baseline"` in the API, a switch on the search page, and a mode in the evaluation), and its results on the same index are unchanged from phase 1.

### The double-check

- Qwen3-VL-8B (MLX, 4-bit, Apache-2.0) looks at the two best-matching crops of each of the top 8 results and answers yes, no or unclear for each detail of the parsed query, such as "a woman", "grey or white jacket", "carrying a beige bag".
- A clear "no" on a key detail rejects the result, "yes" on every key detail confirms it, anything else is unsure; free-text details (glasses, a pattern) are shown but never decide.
- Confirmed results gain and rejected ones lose two units of the fused score (about two standard deviations of similarity), so confirmed results move up and rejected ones drop, but every result stays visible and the officer still decides.
- The search answers at once; the check runs in a background thread and the page fills in each verdict as it arrives.
  Its outcome is stored in the search's trace and written to the audit log as a `check` entry by `vision-check`, so the log shows when a model changed the order.
- Checks run one at a time, so the page cancels the check of a search it no longer shows; a refined query is then not queued behind the old ones, and the cancellation is in the audit log too.
  Stopping the API cancels what is queued or running instead of working through it, and a check cut off by a restart reads as interrupted.
- Re-indexing writes the new crops to a new folder and removes the old one only after the new points have replaced the old ones in Qdrant, so a search during a re-index never checks a deleted crop, or a crop of another person that reused its file name.

### Under the hood: traces

- Every search stores a trace (table `search_traces`): each stage in order with its real output and time, from the raw text through the parsed fields, the embeddings, the filters, the Qdrant candidates and their similarities, the attribute match per field, the fused scores, the double-check answers and the final order, with model names, versions and digests.
- `POST /api/search?trace=1` returns it with the results, `GET /api/search/{search_id}/trace` reopens a past search, and the audit log links to it.
- The recording trace (`GET /api/recordings/{id}/trace`) decodes a few sampled frames, runs the detector on them live, and shows the boxes beside the indexed tracks, every candidate crop with why it was or was not picked, and the attributes with the vision model's raw reply.
  Indexing now stores its statistics (frames, detections, tracks kept and dropped, timings, models) to make this possible.
- Nothing on the page is a fixed example: every stage is traced live.
- Traces keep the query text and the matched tracks' descriptions for as long as the database lives, like the audit log; deleting a recording removes its trace frames with its other derived files.

### Running beside phase 1

- `scripts/start.sh` now sets Qdrant's ports from `AVS_QDRANT_URL` (gRPC on the next port), so a second copy runs on other ports through the `AVS_*` settings alone.
  The phase 2 build was tested on 8480, 8481, 8482 and 6443 while the phase 1 copy kept running on its own ports.
  The clean reproducibility run (a fresh clone with empty Hugging Face, Ollama and uv caches) used 8490, 8491, 8492 and 6453 beside both.

### Licences of the phase 2 models and packages

- Gemma 4 26B-A4B, Qwen3.6-35B-A3B, Jais 3 32B and Qwen3-VL-8B are Apache-2.0.
- Fanar 2 27B is labelled Apache-2.0 but is built on Gemma 3, so Gemma's terms of use may carry over to it; that is our reading, not legal advice, and it matters only if Fanar 2 were chosen.
- The added Python packages (mlx, mlx-vlm, llguidance and their dependencies) are MIT, BSD, Apache-2.0 or PSF licensed.
