# Arabic video search (PoC)

البحث الذكي في تسجيلات كاميرات المراقبة باللغة العربية

Upload CCTV recordings, describe a person or a bag in Arabic, and get matching clips with camera, place and time.
This is phases 0 and 1 of the PoC plan: the option A baseline, where the Arabic text goes straight to SigLIP 2's multilingual text encoder.
Everything runs on one Apple Silicon Mac, and nothing leaves it once setup has downloaded the models and footage.
The choices behind it are in [docs/decisions.md](docs/decisions.md).

## What runs where

All four processes run natively on macOS and listen on 127.0.0.1 only.

| Process | What it does | Port |
|---|---|---|
| Inference service | FFmpeg decode (VideoToolbox), RF-DETR person and bag detection, ByteTrack tracking, SigLIP 2 embeddings, all on the Apple GPU (PyTorch MPS) | 8471 |
| Web/API server | The Arabic-first web UI, the JSON API, playback copies and crops | 8470 |
| Worker | Takes queued uploads and indexes them through the inference service | - |
| Qdrant | Vector index of crop embeddings with camera, place and time payloads | 6433 |

SQLite (`data/avs.db`) holds the recordings, the job queue, the tracks and the append-only audit log.
Docker is not used: containers on macOS get no Apple GPU, and the captain asked for plain scripts instead (see D3 in the decisions file).

## Requirements

- An Apple Silicon Mac (tested on an M4 Pro with 48 GB, macOS 26.6).
- [Homebrew](https://brew.sh); `scripts/setup.sh` installs FFmpeg and uv with it if they are missing (tested with FFmpeg 8.1.1 and uv 0.10.6).
- About 12 GB of free disk: 4.3 GB of SigLIP 2 weights in the Hugging Face cache, 0.4 GB of RF-DETR weights, about 2 GB of Python packages, 2 GB of footage downloads and trimmed clips, and 2 GB of uploads, playback copies and index.
- Internet access for setup and the footage download only.

## Run it from a clean Mac

Each step is a script in `scripts/`; run them from the repository root.

```sh
scripts/setup.sh          # Python 3.12 env from uv.lock, Qdrant v1.19.2 binary, model weights
scripts/fetch_footage.sh  # download and trim the public starter clips, write data/footage/ATTRIBUTION.md
scripts/start.sh          # start Qdrant, the inference service, the web/API server and the worker
scripts/index_starter.sh  # upload the starter clips through the API and wait until they are indexed
open http://127.0.0.1:8470
scripts/evaluate.sh       # score the seeded Arabic queries against the index
scripts/stop.sh           # stop everything
```

On the test Mac, from empty package and model caches, setup took 5 minutes, the footage download 4.5 minutes, and indexing the 11 starter clips (23 minutes of video) 7 minutes.
`scripts/start.sh` is safe to run again; it skips services that are already running.
Logs are in `data/logs/`, and a job interrupted by `scripts/stop.sh` is queued again on the next start.

## Using it

- **Recordings** (التسجيلات): upload a video with its camera ID, place and recording start time (the camera's local time).
  It is queued and indexed in the background, with a progress bar; re-index and delete are on each row.
- **Search** (البحث): type a description, in Arabic or English, optionally tick cameras and set a time window.
  Each result shows the best crop of the person or bag, plays the clip at its moment, and lists camera, place, time span and similarity score.
  Confirm (تأكيد) or reject (رفض) each result.
- **Audit log** (سجل التدقيق): every upload, search, confirm, reject, re-index and delete, with the officer's name and an optional case number.
  The log cannot be edited or deleted.

The same search from the command line:

```sh
curl -s http://127.0.0.1:8470/api/search -H 'Content-Type: application/json' \
  -d '{"query": "امرأة ترتدي سترة حمراء", "cameras": ["G331"], "time_from": "2018-03-07T17:00:00", "time_to": "2018-03-07T17:05:00", "limit": 10, "actor": "officer-1"}'
```

## How it works

Indexing, per uploaded recording:

1. The inference service makes a browser-playable H.264 copy (a remux when the upload is already H.264).
2. FFmpeg decodes 4 frames per second with VideoToolbox.
3. RF-DETR Medium detects people, backpacks, handbags and suitcases.
4. ByteTrack follows each person and each bag within the camera.
5. Each track keeps its 5 best crops: confident, large, not cut off by the frame edge, and spread over time.
6. SigLIP 2 So400m embeds each crop; Qdrant stores one point per crop with camera, place, absolute time and a pointer to the clip.

Searching:

1. SigLIP 2's text encoder embeds the query as written; there is no translation or parsing in this baseline.
2. Qdrant ranks crops by cosine similarity within the chosen cameras and time window, one result per track.
3. The grid plays each track's clip from the recording's playback copy.

## Starter footage

`scripts/fetch_footage.sh` downloads the clips listed in [footage/starter.json](footage/starter.json), trims them, and writes `data/footage/ATTRIBUTION.md` with each source, licence and trim.
Video files and model weights live under `data/`, which git ignores.

| Clips | Source | Licence | Use |
|---|---|---|---|
| 5 | MEVA: school cafe (G421), stairwell doors (G420) and entrance plaza (G638) in one 5-minute window; bus waiting room (G331) on two days | CC BY 4.0 | Suspect and left-bag stories |
| 2 | CAVIAR left-bag lobby clips | CC BY-SA | Left-bag story |
| 2 | NVIDIA PhysicalAI-SmartSpaces synthetic store, scene 071, two cameras | CC BY 4.0 | Lost-child story (synthetic child-sized stand-in) |
| 2 | Pexels mall clips | Pexels License | Neutral searches only, never a suspect |

Known gaps: no public CCTV footage with real children, Arabic dress or a Middle-East setting, and CAVIAR's 384x288 video is too small for the detector to find its bags.
VIRAT, Mixkit and all CC BY-NC sources are left out.

## Evaluation baseline

`scripts/evaluate.sh` runs [eval/queries.yaml](eval/queries.yaml) through the API and reports, per query and per group, the top-10 hit rate (the target is in the first 10 results), precision at 5 (the share of the first 5 results that are correct) and MRR (mean reciprocal rank of the first correct result).
The set has 48 queries: the captain's own queries for the bus waiting room, formal, Jordanian and Gulf rewrites, English originals for comparison, queries for the three stories on the other cameras, and one query nobody matches (Arabic dress).
Each query uses the camera and time filter an officer would set, and every matching person or bag inside that scope was marked by eye from the indexed tracks.
Precision at 5 cannot exceed the number of correct tracks divided by 5, so it is capped for targets seen in fewer than 5 tracks.

Baseline (option A, 17 October 2026):

| Group | Queries | Top-10 hit rate | Precision at 5 | MRR |
|---|---|---|---|---|
| All scored queries | 47 | 0.85 | 0.48 | 0.79 |
| Arabic, all registers | 41 | 0.85 | 0.45 | 0.78 |
| Formal Arabic | 17 | 0.94 | 0.55 | 0.94 |
| Jordanian | 14 | 0.79 | 0.34 | 0.62 |
| Gulf | 10 | 0.80 | 0.44 | 0.73 |
| English (comparison) | 6 | 0.83 | 0.67 | 0.83 |
| Suspect story | 36 | 0.81 | 0.47 | 0.76 |
| Left-bag story | 4 | 1.00 | 0.40 | 1.00 |
| Lost-child story | 7 | 1.00 | 0.60 | 0.83 |

What the numbers say: formal Arabic finds the target almost every time, the dialect phrasings of the same descriptions do worse, and small held objects (the iPad queries) fail in every language.
Those are the gaps phase 2 targets with a local query model, attribute filters and a vision-model double-check.
Each run also saves a full JSON report under `data/eval/`.

## Tests

```sh
.venv/bin/pytest           # unit tests on tiny generated fixtures: decode, crops, tracking, pipeline, database, search, API, evaluation
.venv/bin/pytest -m e2e    # end-to-end: needs scripts/start.sh and the fetched footage
```

The end-to-end test cuts 20 seconds from the cafe clip, uploads and indexes it through the running services, checks that an Arabic query finds the person in the red hoodie in the top 3, confirms the result, checks the audit log, and deletes the recording again.

## Configuration

`scripts/start.sh` passes these environment variables to every service; the defaults are shown.

| Variable | Default | Meaning |
|---|---|---|
| `AVS_DATA_DIR` | `data` | Uploads, playback copies, crops, models, database, logs |
| `AVS_WEB_PORT` | `8470` | Web UI and API |
| `AVS_INFERENCE_URL` | `http://127.0.0.1:8471` | Inference service |
| `AVS_QDRANT_URL` | `http://127.0.0.1:6433` | Qdrant (its own ports are set in `config/qdrant.yaml`) |
| `AVS_SAMPLE_FPS` | `4` | Frames per second sampled for indexing |
| `AVS_DETECTOR` | `rfdetr-medium` | `rfdetr-nano`, `-small`, `-medium` or `-large` |
| `AVS_DETECTION_THRESHOLD` | `0.35` | Minimum detector confidence |
| `AVS_CROPS_PER_TRACK` | `5` | Crops embedded per track |

## Repository layout

- `src/avs/inference/`: the native inference service and the indexing pipeline (detector, tracking, crops, embedder).
- `src/avs/api/`: the web/API server, the worker, SQLite, Qdrant and search.
- `src/avs/tools/`: footage fetch, starter loader, model download, evaluation, and `track_sheet` for writing ground truth.
- `web/`: the UI.
- `footage/starter.json`, `eval/queries.yaml`: the footage list and the seeded queries.
- `scripts/`: setup, start, stop, fetch, index and evaluate.
- `tests/`: unit tests and the end-to-end smoke test.

## Adding ground truth

`.venv/bin/python -m avs.tools.track_sheet sheet --camera G331 --label person` writes contact sheets of every indexed track to `data/eval/sheets/`.
Pick the tracks that match a description, then `.venv/bin/python -m avs.tools.track_sheet anchors <recording_id>/<track_id> ...` prints anchors to paste into `eval/queries.yaml`.
