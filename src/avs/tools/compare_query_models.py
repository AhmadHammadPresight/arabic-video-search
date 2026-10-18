"""Compare the local query models on the parsing part of the test set and pick one.

    python -m avs.tools.compare_query_models [--models NAME ...] [--llm URL] [--no-glossary] [--limit-s S] [--out DIR]

For each candidate in Ollama: load it (timed), parse every query of eval/queries.yaml with the
production prompt and grammar-constrained JSON, and score valid JSON, per-field accuracy
(`avs.tools.parse_scoring`), speed, and the memory Ollama reports for the loaded model.
A candidate that is not downloaded or will not load is recorded as such, not replaced.
A candidate whose parse of one query takes longer than the limit (20 s by default, four times
the 5 s target for a whole search) is stopped there and recorded as too slow, with the time it
measured on the queries it finished.
The pick is the fastest model (median time per query) within 2 points of the best accuracy.
The report is written after every query, so a stopped run keeps what it measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from avs.api.query_parser import NUM_CTX, ParseError, QueryParser
from avs.settings import REPO_ROOT, get_settings
from avs.tools.parse_scoring import SCORED_FIELDS, mentioned, score_fields

CANDIDATES = {
    "Gemma 4 26B-A4B": "gemma4:26b-a4b-it-q4_K_M",
    "Qwen3.6-35B-A3B": "qwen3.6:35b-a3b-q4_K_M",
    "Jais 3 32B": "hf.co/IFM/Jais-3-32B-GGUF:Q4_K_M",
    "Fanar 2 27B": "hf.co/mradermacher/Fanar-2-27B-Instruct-GGUF:Q4_K_M",
}
WITHIN_POINTS = 2.0
LIMIT_S = 20.0


def load_parse_set(path: Path) -> list[dict]:
    queries = yaml.safe_load(path.read_text())["queries"]
    return [q for q in queries if "fields" in q]


def too_slow(rows: list[dict], stopped_on: str, limit_s: float) -> str:
    times = [r["ms"] for r in rows if r["valid"]]
    measured = (f"{statistics.median(times) / 1000:.1f} s per query measured on {len(times)} queries"
                if times else "no query finished")
    return f"too slow on this Mac for query parsing: {measured}; query {stopped_on} passed the {limit_s:g} s limit"


def run_model(client: httpx.Client, label: str, model: str, queries: list[dict], glossary: bool,
              limit_s: float = LIMIT_S, parser: QueryParser | None = None, on_row=None) -> dict:
    out: dict = {"label": label, "model": model, "glossary": glossary, "limit_s": limit_s}
    tags = {m["name"] for m in client.get("/api/tags").json().get("models", [])}
    if model not in tags:
        return out | {"status": "not downloaded"}
    started = time.monotonic()
    loaded = client.post("/api/generate", json={"model": model, "keep_alive": "15m", "options": {"num_ctx": NUM_CTX}},
                         timeout=600.0)
    if loaded.status_code != 200:
        return out | {"status": f"failed to load: {loaded.text[:300]}"}
    out["load_s"] = round(time.monotonic() - started, 1)
    ps = next((m for m in client.get("/api/ps").json().get("models", []) if m.get("name") == model), {})
    out["memory_gb"] = round(ps.get("size", 0) / 1e9, 1)
    out["context"] = ps.get("context_length")

    parser = parser or QueryParser(str(client.base_url), model, timeout=limit_s, glossary=glossary)
    rows: list[dict] = []
    for q in queries:
        row = {"id": q["id"], "register": q.get("register", "formal"), "text": q["text"], "source": q.get("source")}
        t0 = time.monotonic()
        try:
            result = parser.parse(q["text"], use_cache=False)
        except ParseError as exc:
            if isinstance(exc.__cause__, httpx.TimeoutException):
                print(f"  {label:16} {q['id']:4} passed the {limit_s:g} s limit: stopping this model", flush=True)
                client.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=120.0)
                return out | {"status": too_slow(rows, q["id"], limit_s), "rows": rows,
                              "partial_summary": summarise(rows) if rows else None}
            row |= {"valid": False, "error": str(exc), "raw": exc.raw, "ms": round((time.monotonic() - t0) * 1000)}
            row["correct"] = {f: False for f in SCORED_FIELDS}
        else:
            row |= {"valid": True, "ms": result.took_ms, "output_tokens": result.output_tokens,
                    "fields": result.fields.model_dump(exclude={"english"}), "english": result.fields.english}
            row["correct"] = score_fields(q["fields"], result.fields)
        row["mentioned"] = sorted(mentioned(q["fields"]))
        rows.append(row)
        if on_row:
            on_row(out | {"status": "running", "rows": rows})
        print(f"  {label:16} {q['id']:4} {'ok ' if row['valid'] else 'BAD'} {sum(row['correct'].values()):2}/{len(SCORED_FIELDS)} "
              f"{row['ms']:6} ms  {row.get('english', row.get('error', ''))[:70]}", flush=True)
    client.post("/api/generate", json={"model": model, "keep_alive": 0})
    return out | {"status": "ok", "rows": rows, "summary": summarise(rows)}


def summarise(rows: list[dict]) -> dict:
    def accuracy(items: list[dict], fields=SCORED_FIELDS) -> float:
        cells = [r["correct"][f] for r in items for f in fields]
        return round(100 * sum(cells) / len(cells), 1) if cells else 0.0

    mentioned_cells = [r["correct"][f] for r in rows for f in r["mentioned"]]
    times = sorted(r["ms"] for r in rows if r["valid"])
    tokens = [r["output_tokens"] for r in rows if r.get("output_tokens")]
    by_register: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_register[r["register"]].append(r)
    return {
        "queries": len(rows),
        "valid_json": round(100 * sum(r["valid"] for r in rows) / len(rows), 1),
        "field_accuracy": accuracy(rows),
        "named_field_accuracy": round(100 * sum(mentioned_cells) / len(mentioned_cells), 1) if mentioned_cells else 0.0,
        "exact_queries": round(100 * sum(all(r["correct"].values()) for r in rows) / len(rows), 1),
        "per_field": {f: accuracy(rows, (f,)) for f in SCORED_FIELDS},
        "per_register": {k: accuracy(v) for k, v in sorted(by_register.items())},
        "median_ms": round(statistics.median(times)) if times else None,
        "p90_ms": times[int(0.9 * (len(times) - 1))] if times else None,
        "mean_output_tokens": round(statistics.fmean(tokens)) if tokens else None,
    }


def pick(results: list[dict]) -> dict | None:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return None
    best = max(r["summary"]["field_accuracy"] for r in ok)
    eligible = [r for r in ok if r["summary"]["field_accuracy"] >= best - WITHIN_POINTS]
    choice = min(eligible, key=lambda r: r["summary"]["median_ms"])
    return {"label": choice["label"], "model": choice["model"], "best_accuracy": best,
            "eligible": [r["label"] for r in eligible]}


def table(results: list[dict]) -> str:
    lines = ["| Model | Status | Valid JSON | Field accuracy | Named fields | Whole query right | Median per query | Memory |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.get("summary")
        if not s:
            memory = f"{r['memory_gb']} GB" if r.get("memory_gb") else "-"
            lines.append(f"| {r['label']} (`{r['model']}`) | {r['status']} | - | - | - | - | - | {memory} |")
            continue
        lines.append(f"| {r['label']} (`{r['model']}`) | ok, loads in {r['load_s']} s | {s['valid_json']}% | "
                     f"{s['field_accuracy']}% | {s['named_field_accuracy']}% | {s['exact_queries']}% | "
                     f"{s['median_ms'] / 1000:.1f} s | {r['memory_gb']} GB |")
    return "\n".join(lines)


def select_candidates(names: list[str]) -> dict[str, str]:
    """Labels or Ollama names from CANDIDATES, or LABEL=NAME for a model under another name."""
    by_model = {model: label for label, model in CANDIDATES.items()}
    out = {}
    for name in names:
        label, _, model = name.partition("=")
        if model:
            out[label] = model
        elif name in CANDIDATES:
            out[name] = CANDIDATES[name]
        else:
            out[by_model.get(name, name)] = name
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.yaml"))
    parser.add_argument("--models", nargs="*", help="labels, Ollama names or LABEL=NAME; default: every candidate")
    parser.add_argument("--llm", default=get_settings().llm_url)
    parser.add_argument("--no-glossary", action="store_true", help="leave the dialect hints out of the prompt")
    parser.add_argument("--limit-s", type=float, default=LIMIT_S, help="stop a model whose parse of one query takes longer")
    parser.add_argument("--out", default=str(get_settings().data_dir / "eval"))
    args = parser.parse_args(argv)

    candidates = select_candidates(args.models) if args.models else CANDIDATES
    queries = load_parse_set(Path(args.queries))
    client = httpx.Client(base_url=args.llm, timeout=60.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.no_glossary else "-no-glossary"
    path = out / f"query-models-{datetime.now(UTC):%Y%m%dT%H%M%SZ}{suffix}.json"
    report = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "llm": args.llm,
              "ollama": client.get("/api/version").json().get("version"), "glossary": not args.no_glossary,
              "limit_s": args.limit_s, "queries": len(queries), "results": [], "pick": None}

    def save(results: list[dict]) -> None:
        path.write_text(json.dumps(report | {"results": results, "pick": pick(results)}, ensure_ascii=False, indent=2))

    results: list[dict] = []
    for label, model in candidates.items():
        results.append(run_model(client, label, model, queries, not args.no_glossary, args.limit_s,
                                 on_row=lambda partial: save([*results, partial])))
        save(results)
    report["pick"] = pick(results)
    print("\n" + table(results))
    for r in results:
        if r.get("summary"):
            print(f"\n{r['label']}: per field {r['summary']['per_field']}\n  per register {r['summary']['per_register']}")
    print(f"\npick: {report['pick']}\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
