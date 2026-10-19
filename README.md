# Arabic video search (PoC)

البحث الذكي في تسجيلات كاميرات المراقبة باللغة العربية

Upload CCTV recordings, describe a person or a bag in Arabic, and get matching clips with camera, place and time.
This is phases 0 to 2 of the PoC plan.
Phase 1 is the option A baseline, where the Arabic text goes straight to SigLIP 2's multilingual text encoder.
Phase 2 is the hybrid: a local language model reads the query into fields and a plain English description, search filters and ranks on those fields together with camera and time, a vision model double-checks the top results, and every result says why it matched.
Option A stays selectable, so the two can be compared on the same index.
Everything runs on one Apple Silicon Mac, and nothing leaves it once setup has downloaded the models and footage.
The choices behind it are in [docs/decisions.md](docs/decisions.md).
To set up and present the demo on another Mac, follow [docs/DEMO-GUIDE.md](docs/DEMO-GUIDE.md).

## What runs where

All five processes run natively on macOS and listen on 127.0.0.1 only.

| Process | What it does | Port |
|---|---|---|
| Inference service | FFmpeg decode (VideoToolbox), RF-DETR person and bag detection, ByteTrack tracking, SigLIP 2 embeddings (PyTorch MPS), and Qwen3-VL-8B for track attributes and the double-check (MLX), all on the Apple GPU | 8471 |
| Query model server | Ollama v0.40.6 running the query model, which turns the query into fields and an English rewrite | 8472 |
| Web/API server | The Arabic-first web UI, the JSON API, playback copies, crops and trace frames | 8470 |
| Worker | Takes queued uploads and indexes them through the inference service | - |
| Qdrant | Vector index of crop embeddings with camera, place, time and attribute payloads | 6433 |

SQLite (`data/avs.db`) holds the recordings, the job queue, the tracks with their attributes, the search traces and the append-only audit log.
Docker is not used: containers on macOS get no Apple GPU, and the project owner asked for plain scripts instead (see D3 in the decisions file).

## Requirements

- An Apple Silicon Mac (tested on an M4 Pro with 48 GB, macOS 26.6).
  The query model, the vision models and the search together use about 30 GB of memory while running.
- [Homebrew](https://brew.sh); `scripts/setup.sh` installs FFmpeg and uv with it if they are missing (tested with FFmpeg 8.1.1 and uv 0.10.6).
- About 40 GB of free disk; the clean run below used 37.1 GB:
  - 10.4 GB in the Hugging Face cache: SigLIP 2 (4.6 GB) and Qwen3-VL-8B, MLX 4-bit (5.8 GB);
  - 18.0 GB for the query model in Ollama's model store (`~/.ollama/models`);
  - 0.4 GB of RF-DETR weights, 0.6 GB for the Qdrant and Ollama binaries, 1.7 GB of Python packages and 1.6 GB of uv's download cache;
  - 2.2 GB of footage downloads and trimmed clips, and 2.3 GB of uploads, playback copies, crops and the index.
  - Re-running the query model comparison also downloads the other three candidates, about 62 GB more, and the patched Fanar 2 copy takes another 16.5 GB.
- Internet access for setup and the footage download only.

## Run it from a clean Mac

Each step is a script in `scripts/`; run them from the repository root.

```sh
scripts/setup.sh          # Python 3.12 env from uv.lock, pinned Qdrant and Ollama binaries, vision model weights, the query model
scripts/fetch_footage.sh  # download and trim the public starter clips, write data/footage/ATTRIBUTION.md
scripts/start.sh          # start Qdrant, Ollama, the inference service, the web/API server and the worker
scripts/index_starter.sh  # upload the starter clips through the API and wait until they are indexed
open http://127.0.0.1:8470
scripts/evaluate.sh       # score the test queries in option A and in the hybrid, side by side
scripts/stop.sh           # stop everything
```

The whole sequence was run from empty caches on the test Mac, with downloads at about 12 MB/s:

- `scripts/setup.sh`: 41.5 minutes, nearly all of it downloading about 29 GB.
- `scripts/fetch_footage.sh`: 4.6 minutes.
- `scripts/start.sh`: 74 seconds the first time, 25 seconds after that.
- Indexing the 11 starter clips (486 tracks): 44 minutes, 36.5 of them for the vision model's attributes.
- `scripts/evaluate.sh`: 36 minutes for the 113 queries in both modes, most of it waiting for the double-checks.

`scripts/start.sh` is safe to run again; it skips services that are already running, and loads the query model into memory before the API starts.
Logs are in `data/logs/`, and a job interrupted by `scripts/stop.sh` is queued again on the next start.

Recordings indexed before phase 2 have no attributes; `scripts/reindex_all.sh` queues every recording again and waits until all are re-indexed with the current pipeline.

### Running a second copy beside the first

Every port comes from the `AVS_*` settings, so a second copy (another checkout with its own `data/`) runs beside the first:

```sh
export AVS_WEB_PORT=8480 AVS_INFERENCE_URL=http://127.0.0.1:8481 AVS_LLM_URL=http://127.0.0.1:8482 AVS_QDRANT_URL=http://127.0.0.1:6443
scripts/setup.sh && scripts/start.sh
```

Qdrant takes its HTTP port from `AVS_QDRANT_URL` and uses the next port for gRPC.
Both copies share the Hugging Face cache and Ollama's model store, so models are downloaded once.

## Using it

The page has two views.
The business view at `/` (http://127.0.0.1:8470) has Search, Recordings and the Audit log, and always searches in the hybrid mode.
The technical view at `/tech` adds the search mode switch, the Under the hood tab and the trace links; it is a plain address, with no login.
The page shows one language at a time: Arabic, right to left, by default, and English, left to right, from the language button at the top.
The choice is remembered in the browser, and every string is in `web/locales/ar.json` and `web/locales/en.json`.

- **Search** (البحث): type a description in Arabic (formal or dialect), English or a mix, optionally tick cameras and set a time window.
  The hybrid mode (هجين) is the default; in the technical view option A (الخيار أ) is one click away.
  - In the hybrid mode a panel shows what the query model understood: the fields, such as woman and grey jacket, and, in the technical view, the English rewrite.
  - Each result shows the best crop, plays the clip at its moment, lists camera, place, time span and score, and shows why it matched, field by field: ✓ match, ≈ close (beige for brown), ? not known for this track, – not seen, ✗ different.
  - A few seconds later the vision model's double-check fills in on the top results: confirmed results move up, rejected ones move down (the technical view also shows the model's reason), and unsure ones stay put.
    Starting a new search stops the previous search's check.
  - Confirm (تأكيد) or reject (رفض) each result; the officer always decides.
- **Recordings** (التسجيلات): upload a video with its camera ID, place and recording start time (the camera's local time).
  It is queued and indexed in the background, with a progress bar; re-index and delete are on each row, and "how it was indexed" too in the technical view.
- **Audit log** (سجل التدقيق): every upload, search, confirm, reject, re-index and delete, with the officer's name and an optional case number, plus a `check` entry whenever the vision model re-orders a search.
  In the technical view each search links to its trace.
  The log cannot be edited or deleted.
- **Under the hood** (كيف يعمل, technical view only): see what happens inside, from live data.
  - Search trace: type a description and it runs in both modes side by side, showing every stage in order with its real output and time: the raw text, the query model's fields and English rewrite, the SigLIP 2 embeddings, the filters applied, the candidate tracks from Qdrant with their similarities, the attribute match per field, the fused scores, the double-check answer for each result and the final order.
    Any past search can be reopened by its id or from the audit log.
  - Recording trace: pick an indexed recording to see the indexing settings and statistics, a few sampled frames with the detector's boxes drawn on them (run live) beside the indexed tracks, every track with its chosen crops and why each candidate crop was or was not picked, and the attributes the vision model attached, with its raw reply.

The same search from the command line, with its trace:

```sh
curl -s 'http://127.0.0.1:8470/api/search?trace=1' -H 'Content-Type: application/json' \
  -d '{"query": "مرة لابسة جاكيت أحمر", "cameras": ["G331"], "time_from": "2018-03-07T17:00:00", "time_to": "2018-03-07T17:05:00", "limit": 10, "actor": "officer-1"}'
curl -s http://127.0.0.1:8470/api/search/<search_id>/check   # the double-check's progress, then the new order
curl -s http://127.0.0.1:8470/api/search/<search_id>/trace   # the stored trace of any past search
curl -s http://127.0.0.1:8470/api/recordings/<recording_id>/trace?frames=6
```

Add `"mode": "baseline"` to run option A.

## How it works

Indexing, per uploaded recording:

1. The inference service makes a browser-playable H.264 copy (a remux when the upload is already H.264).
2. FFmpeg decodes 4 frames per second with VideoToolbox.
3. RF-DETR Medium detects people, backpacks, handbags and suitcases.
4. ByteTrack follows each person and each bag within the camera.
5. Each track keeps its 5 best crops: confident, large, not cut off by the frame edge, and spread over time.
6. SigLIP 2 So400m embeds each crop.
7. Qwen3-VL-8B looks at the track's two best crops and fills in its attributes: gender, age band, upper and lower garment and colours, hair, headwear and carried objects, in the same fixed vocabulary the query model uses.
8. Qdrant stores one point per crop with camera, place, absolute time, the track's attributes and a pointer to the clip.

Searching in option A:

1. SigLIP 2's text encoder embeds the query as written.
2. Qdrant ranks crops by cosine similarity within the chosen cameras and time window, one result per track.

Searching in the hybrid mode:

1. The query model (in Ollama, with output constrained to a fixed JSON schema) turns the text into fields and a plain English rewrite, so dialect words such as سكني (grey) or رجّال (a man) stop mattering.
2. SigLIP 2 embeds both the English rewrite and the original text.
3. Qdrant filters by camera and time and keeps only people or only bags as the query asks.
   Gender is not filtered: the vision model gets it wrong on some tracks, so a gender mismatch lowers the score instead of hiding the track.
4. The most similar tracks for either text form the candidate pool; each candidate's attributes are matched field by field, and the attribute score is fused with the similarity.
5. The first results return at once; Qwen3-VL-8B then checks the top 8 against each detail of the query and the results re-rank.

If the query model is not answering, the hybrid search falls back to option A and says so.

## Choosing the query model

`python -m avs.tools.compare_query_models` runs every candidate in Ollama on the parsing part of the test set: each of the 113 queries in `eval/queries.yaml` has the fields it names written out, and eleven fields are scored per query (subject, gender, age band, upper garment and colours, lower garment and colours, hair colour and length, headwear, carried objects), so an invented detail counts as an error too.
The candidates follow the phase 2 go-ahead, at the same 4-bit quantisation (Q4_K_M GGUF) on the same server, with the production prompt.

| Model | Result | Valid JSON | Field accuracy | Named fields | Whole query right | Median per query | Memory while loaded |
|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B (`gemma4:26b-a4b-it-q4_K_M`) | **chosen** | 100% | 99.4% | 99.0% | 93.8% | 1.9 s | 18.1 GB |
| Qwen3.6-35B-A3B (`qwen3.6:35b-a3b-q4_K_M`) | ran | 100% | 99.3% | 98.3% | 92.0% | 2.4 s | 23.1 GB |
| Fanar 2 27B (`hf.co/mradermacher/Fanar-2-27B-Instruct-GGUF:Q4_K_M`) | ran, after a template fix (below) | 100% | 98.7% | 98.3% | 89.4% | 8.3 s | 18.9 GB |
| Jais 3 32B (`hf.co/IFM/Jais-3-32B-GGUF:Q4_K_M`) | deferred to a later phase: too slow on this Mac | 100% of 61 | 99.9% of 61 | - | - | 64 s | 22.4 GB |

- Field accuracy counts all eleven fields of every query (1,243 in all); named fields counts only the fields a query names; whole query right is the share of queries with all eleven right.
- Median per query is the time of one parse with the model loaded, as a search sees it.
- Memory is what Ollama allocates for the model with its 4096-token context (weights, cache and working buffers), added up from its load log.
  Ollama's `/api/ps` reports only 1.2 GB for Gemma 4, because it counts one of the 33 parts its weights are mapped in.
- The rule was set before measuring: the fastest model within 2 points of the best field accuracy.
  Gemma 4 26B-A4B is both the most accurate and the fastest, so it is the query model.
- Accuracy hardly separates the candidates: on these queries they differ by a few fields out of 1,243.
  Speed does: Gemma 4 and Qwen3.6 are mixture-of-experts models that use about 4B and 3B parameters per token, while Fanar 2 and Jais 3 are dense and use all 27B to 35B.
- Each model's run stops at the first query that takes longer than 20 s (four times the 5 s target for a whole search), and the model is recorded as too slow with what it measured.
- Jais 3 32B loads and runs in Ollama and its fields were right (99.9% on the 61 queries it finished), but it took 37 to 268 s per query, a median of 64 s, so the run was stopped there; under the 20 s limit its first query does not finish.
  At the project owner's request it is deferred to a later phase rather than tuned further.
- Fanar 2 27B as published does not load in Ollama 0.40.6: its built-in chat template wraps the tool-calling instructions in `{% raw %}`, which the template engine rejects ("Unknown statement: raw").
  The text inside has no template syntax, so `python -m avs.tools.patch_gguf_template` writes a copy of the GGUF with the two tags replaced by template comments of the same length, which produce exactly the same prompt; that copy was measured.
  To repeat it: find the published file with `ollama show --modelfile <model>`, patch it, import the copy with `ollama create` and the published template, and pass `--models "Fanar 2 27B=<name of the copy>"`.
  It scored 98.7% at a median of 8.3 s per query: within 2 points of Gemma 4, but four times slower and over the 5 s target for a whole search on its own, so the pick stays Gemma 4.
  Its weakest field was hair length (95.6%).
- Without the dialect glossary in the prompt (`--no-glossary`), Gemma 4 scored 99.5%, so its accuracy does not lean on the glossary's overlap with the test words.
  Qwen3.6 dropped to 98.5% (96.0% on named fields), mostly on Gulf and Jordanian words such as بنية (a girl) and ختيار (an old man).
  The times of that run are not comparable, because it ran alongside the clean-run downloads.
- The report under `data/eval/` lists every English rewrite for reading; a native speaker of each dialect should still rate them (a follow-up).

## Attributes per track

Each indexed track gets attributes in the same vocabulary as the query fields, so search can filter and boost on them.
Two ways were measured with `python -m avs.tools.attribute_benchmark` on 81 tracks labelled by eye ([eval/attributes.yaml](eval/attributes.yaml), a fixed random sample across the MEVA and NVIDIA cameras), each method looking at the same crops:

| Field | Labelled tracks | SigLIP 2 zero-shot prompts | Qwen3-VL-8B description |
|---|---|---|---|
| Gender | 52 | 90% | 96% |
| Upper garment colour | 65 | 72% | 92% |
| Lower garment colour | 55 | 38% | 86% |
| Hair colour | 30 | 87% | 87% |
| Headwear present | 54 | 54% | 78% |
| Carries a bag | 49 | 51% | 96% |
| Bag colour | 8 | 75% | 100% |
| All fields | | 64% | 90% |
| Time per track | | 22 ms | 4.1 s |

The vision model is used (`AVS_ATTRIBUTES=vlm`): it is right far more often, above all on what the zero-shot prompts mix up, such as the colour of the trousers rather than the jacket, or whether a bag is carried.
It costs about 4 seconds per track: indexing the 11 starter clips (486 tracks) took 44 minutes in the clean run, 36.5 of them for attributes, instead of about 7 without.

## Evaluation: phase 1 and phase 2 side by side

`scripts/evaluate.sh` runs [eval/queries.yaml](eval/queries.yaml) through the API in both modes and reports, per query and per group, the top-10 hit rate (the target is in the first 10 results), precision at 5 (the share of the first 5 results that are correct) and MRR (mean reciprocal rank of the first correct result).
The hybrid is scored twice: its first ranking, and the ranking after the double-check.

The set has 113 queries.
The first 48 are the phase 1 set: the project owner's own queries for the bus waiting room (kept verbatim), formal, Jordanian and Gulf rewrites, English originals for comparison, queries for the three stories on the other cameras, and one query nobody matches.
Phase 2 added 65: more formal, Jordanian and Gulf phrasings, a new mixed Arabic-English register, seven new targets on the school and store cameras, and two more queries nobody matches (Gulf and Levantine dress).
Each query uses the camera and time filter an officer would set, and every matching person or bag inside that scope was marked by eye from contact sheets of the indexed tracks, not from search results.
Precision at 5 cannot exceed the number of correct tracks divided by 5, so it is capped for targets seen in fewer than 5 tracks.
The fusion weights were set on the phase 1 set only, before the 65 new queries were written, so the new queries are also reported on their own as a held-out check.

These numbers come from the clean run described above, on an index built from scratch.
Option A is the phase 1 search, unchanged; on the phase 1 set it scores exactly what phase 1 reported (0.85, 0.48, 0.79).
"Hybrid" is the first ranking a search returns, and "Hybrid + check" the ranking after the double-check.

| Group | Queries | Option A hit@10 | P@5 | MRR | Hybrid hit@10 | P@5 | MRR | Hybrid + check hit@10 | P@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 110 | 0.91 | 0.44 | 0.81 | 0.99 | 0.51 | 0.88 | 0.99 | 0.49 | 0.90 |
| jordanian | 25 | 0.80 | 0.34 | 0.64 | 1.00 | 0.49 | 0.86 | 1.00 | 0.47 | 0.91 |
| formal | 31 | 0.97 | 0.50 | 0.93 | 1.00 | 0.48 | 0.90 | 1.00 | 0.47 | 0.91 |
| english | 6 | 0.83 | 0.67 | 0.83 | 1.00 | 0.67 | 0.87 | 1.00 | 0.63 | 0.85 |
| gulf | 23 | 0.91 | 0.44 | 0.77 | 1.00 | 0.54 | 0.89 | 1.00 | 0.53 | 0.90 |
| mixed | 25 | 0.96 | 0.43 | 0.84 | 0.96 | 0.48 | 0.85 | 0.96 | 0.48 | 0.88 |
| Arabic, all registers | 104 | 0.91 | 0.43 | 0.80 | 0.99 | 0.50 | 0.88 | 0.99 | 0.49 | 0.90 |
| suspect story | 79 | 0.87 | 0.42 | 0.76 | 0.99 | 0.50 | 0.85 | 0.99 | 0.50 | 0.88 |
| left-bag story | 8 | 1.00 | 0.38 | 1.00 | 1.00 | 0.30 | 1.00 | 1.00 | 0.23 | 1.00 |
| lost-child story | 23 | 1.00 | 0.54 | 0.90 | 1.00 | 0.58 | 0.91 | 1.00 | 0.58 | 0.91 |
| phase 1 set | 47 | 0.85 | 0.48 | 0.79 | 1.00 | 0.60 | 0.87 | 1.00 | 0.58 | 0.89 |
| added in phase 2 | 63 | 0.95 | 0.42 | 0.82 | 0.98 | 0.43 | 0.88 | 0.98 | 0.43 | 0.90 |

What phase 2 changes:

- The hybrid finds a correct track in the first 10 for 109 of the 110 scored queries, against 100 for option A, and ranks it first for 88 (91 after the double-check) against 81.
  Of option A's 10 misses it finds 9; the one left is "رجل ماسك iPad" (a man holding an iPad), which neither mode finds.
- The gain is largest where phase 1 was weakest, the dialects: Jordanian goes from 0.80 to 1.00 top-10 hit rate and from 0.64 to 0.91 MRR, Gulf from 0.91 to 1.00 and from 0.77 to 0.90.
  The project owner's own query that phase 1 missed, "امراة تلبس جاكيت سكني او ابيض" (c01), now has the target first, as do the Jordanian and Gulf versions of the project owner's red-jacket query (c07, c08).
  The project owner's iPad query (c21) goes from a miss to 5th, and to 8th after the double-check.
- On the 63 queries added in phase 2, written after the fusion settings were fixed, the gain is smaller: 0.95 to 0.98 hit rate, 0.42 to 0.43 precision at 5 and 0.82 to 0.90 MRR after the double-check.
  Option A was already better on these than on the phase 1 set, and the settings were chosen on the phase 1 set, so this is the fairer estimate of what phase 2 adds.

Where phase 2 does not help, or does worse:

- Formal Arabic was already option A's strength (0.97, 0.50, 0.93); the hybrid finds the one formal target option A missed, but precision at 5 and MRR are slightly lower (0.47 and 0.91 after the double-check).
- Nine queries that option A answered with the target first now have it second or third: the woman in a grey or white jacket with a beige bag (c03, c04, p01, p02), the person in a light blue sweater (s11, p33, p34) and the woman in a pink suit (l06, p40).
  In each, another track whose attributes fit the description just as well ranks just above the target on a slightly higher similarity.
- Left-bag story: precision at 5 drops from 0.38 to 0.30, and to 0.23 after the double-check, although the bag is still first in all 8 queries.
  A bag query searches bag tracks only, while the ground truth for the brown suitcase also counts the two people pulling it, whose crops show the suitcase and which option A returns; the double-check then confirms other brown bags as matches.
- The double-check raises MRR (0.88 to 0.90) and the number of targets ranked first, but lowers precision at 5 slightly (0.51 to 0.49).
  It is least reliable on small objects: for three of the four phrasings of "a man carrying an iPad" it answered no for the tablet and moved the target from 5th to 7th or 8th; for the fourth it confirmed it and moved it to first.
- Neither mode can say that nobody matches: both always return the closest tracks.
  For the three queries that match nobody, the double-check rejected all 8 checked results for both shemagh queries, a useful signal, but confirmed 6 of 8 for the woman in a black abaya and shayla.

Speed, from the same run (one search at a time):

| | Median | 90th percentile | Slowest |
|---|---|---|---|
| Option A results | 0.08 s | 0.12 s | 0.27 s |
| Hybrid first results | 2.1 s | 2.3 s | 7.3 s |
| Double-check of the top 8, after the first results | 17.2 s | 19.7 s | 21.9 s |

- The hybrid's first results are under the 5 s target for 112 of 113 searches; parsing the query takes about 2 s of that, embedding both texts 0.15 to 0.4 s, and the search and fusion about 0.1 s.
- The slowest was the first search of the run (7.3 s): the query model had not yet processed its prompt since starting.
  The API now has it parse a fixed sentence when it starts, and the first search after a restart then took 2.2 s.
- The double-check takes about 2 s per result and fills in on the page while the officer looks at the first results.

Each run also saves a full JSON report under `data/eval/`.

## Tests

```sh
.venv/bin/pytest           # unit tests: decode, crops, tracking, pipeline, database, search, query parser, attributes, matching, fusion, double-check, traces, API, evaluation
.venv/bin/pytest -m e2e    # end-to-end: needs scripts/start.sh and the fetched footage
```

The unit tests stub the query model, the vision model and the embedder; they need no running services.
The end-to-end test cuts 20 seconds from the cafe clip, uploads and indexes it through the running services (attributes included), finds the person in the red hoodie in option A and confirms the result, then searches again in the hybrid mode with a Jordanian phrasing, checks the parsed fields, the per-field match, the 5-second target and the double-check, reads both traces back, and deletes the recording again.

## Configuration

`scripts/start.sh` passes these environment variables to every service; the defaults are shown.

| Variable | Default | Meaning |
|---|---|---|
| `AVS_DATA_DIR` | `data` | Uploads, playback copies, crops, trace frames, models, database, logs |
| `AVS_WEB_PORT` | `8470` | Web UI and API |
| `AVS_INFERENCE_URL` | `http://127.0.0.1:8471` | Inference service |
| `AVS_LLM_URL` | `http://127.0.0.1:8472` | Ollama, the query model server |
| `AVS_QDRANT_URL` | `http://127.0.0.1:6433` | Qdrant; its HTTP port follows this URL and gRPC uses the next port |
| `AVS_QUERY_MODEL` | `gemma4:26b-a4b-it-q4_K_M` | The query model, as an Ollama model name |
| `AVS_OLLAMA_MODELS` | Ollama's default (`~/.ollama/models`) | Where Ollama keeps models |
| `AVS_VLM_MODEL`, `AVS_VLM_REVISION` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` at a pinned revision | The vision model for attributes and the double-check |
| `AVS_ATTRIBUTES` | `vlm` | How tracks get attributes at index time: `vlm`, `siglip` (zero-shot prompts) or `none` |
| `AVS_CHECK_TOP` | `8` | How many top results the double-check looks at |
| `AVS_SAMPLE_FPS` | `4` | Frames per second sampled for indexing |
| `AVS_DETECTOR` | `rfdetr-medium` | `rfdetr-nano`, `-small`, `-medium` or `-large` |
| `AVS_DETECTION_THRESHOLD` | `0.35` | Minimum detector confidence |
| `AVS_CROPS_PER_TRACK` | `5` | Crops embedded per track |

## Repository layout

- `src/avs/fields.py`: the fixed description schema shared by the query model, the attributes and the double-check.
- `src/avs/inference/`: the native inference service and the indexing pipeline (detector, tracking, crops, embedder, attributes, the vision model).
- `src/avs/api/`: the web/API server, the worker, SQLite, Qdrant, option A search, the query parser, matching and fusion, the hybrid search, the double-check and traces.
- `src/avs/tools/`: footage fetch, starter loader, model download, re-index, evaluation, the query model comparison, its scoring and the GGUF template patch, the attribute benchmark, and `track_sheet` for writing ground truth.
- `web/`: the UI.
- `footage/starter.json`, `eval/queries.yaml`, `eval/attributes.yaml`: the footage list, the test queries with their expected fields, and the attribute labels.
- `scripts/`: setup, start, stop, fetch, index, re-index and evaluate.
- `tests/`: unit tests and the end-to-end smoke test.
- `docs/`: the decisions log, the demo guide, and the use-case deck in `docs/deck/`.

## Adding ground truth

`.venv/bin/python -m avs.tools.track_sheet sheet --camera G331 --label person` writes contact sheets of every indexed track to `data/eval/sheets/`.
Pick the tracks that match a description, then `.venv/bin/python -m avs.tools.track_sheet anchors <recording_id>/<track_id> ...` prints anchors to paste into `eval/queries.yaml`.
Write the query's `fields` too, so the query model comparison scores it.
