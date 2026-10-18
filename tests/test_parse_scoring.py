"""The query model comparison: field scoring, the pick, the time limit and the GGUF template patch."""

from __future__ import annotations

import httpx
import pytest

from avs.api.query_parser import ParseError, ParseResult
from avs.fields import QueryFields
from avs.tools.compare_query_models import pick, run_model, select_candidates, summarise
from avs.tools.parse_scoring import SCORED_FIELDS, mentioned, score_fields

EXPECTED = {"gender": "female", "upper": {"garment": ["jacket", "coat"], "colours": ["grey", "white"]},
            "carried": [{"object": ["bag", "handbag"], "colours": ["brown", "beige"]}]}


def test_a_perfect_parse_scores_every_field():
    got = QueryFields(gender="female", age_band="adult", upper={"garment": "coat", "colours": ["white", "grey"]},
                      carried=[{"object": "handbag", "colours": ["beige", "brown"]}], english="x")
    assert all(score_fields(EXPECTED, got).values())
    assert mentioned(EXPECTED) == {"gender", "upper_garment", "upper_colours", "carried"}


def test_missing_wrong_and_invented_details_each_cost_a_field():
    got = QueryFields(gender="male", upper={"garment": "hoodie", "colours": ["grey"]}, hair={"colour": "black"},
                      carried=[{"object": "bag", "colours": ["brown", "beige"]}, {"object": "phone"}], english="x")
    score = score_fields(EXPECTED, got)
    wrong = {f for f, ok in score.items() if not ok}
    assert wrong == {"gender", "upper_garment", "upper_colours", "hair_colour", "carried"}
    assert set(score) == set(SCORED_FIELDS)


def test_alternative_colour_lists_and_bag_subjects():
    expected = {"subject": "bag", "lower": {"garment": "trousers", "colours": [[], ["green"]]}}
    assert score_fields(expected, QueryFields(subject="bag", lower={"garment": "trousers", "colours": ["green"]}, english="x"))["lower_colours"]
    assert score_fields(expected, QueryFields(subject="bag", lower={"garment": "trousers"}, english="x"))["lower_colours"]
    assert not score_fields(expected, QueryFields(lower={"garment": "trousers"}, english="x"))["subject"]
    assert mentioned({"gender": [None, "male"]}) == {"gender"}


def test_the_pick_is_the_fastest_model_within_two_points_of_the_best():
    def row(correct: bool, ms: int, register="formal") -> dict:
        return {"register": register, "valid": True, "ms": ms, "output_tokens": 100, "mentioned": ["gender"],
                "correct": {f: correct for f in SCORED_FIELDS}}

    best = [row(True, 3000)] * 99 + [row(False, 3000)]
    close = [row(True, 1500)] * 98 + [row(False, 1500)] * 2
    far = [row(True, 500)] * 90 + [row(False, 500)] * 10
    results = [{"label": name, "model": name, "status": "ok", "summary": summarise(rows)}
               for name, rows in (("best", best), ("close", close), ("far", far))]
    results.append({"label": "broken", "model": "broken", "status": "failed to load: unknown architecture"})
    assert summarise(best)["field_accuracy"] == 99.0 and summarise(close)["median_ms"] == 1500
    assert pick(results) == {"label": "close", "model": "close", "best_accuracy": 99.0, "eligible": ["best", "close"]}
    assert pick([results[-1]]) is None


def test_a_model_slower_than_the_limit_is_stopped_and_recorded_with_its_measured_time():
    def ollama(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "slow"}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "slow", "size": 21_600_000_000, "context_length": 4096}]})
        return httpx.Response(200, json={})

    class SlowParser:
        calls = 0

        def parse(self, text: str, use_cache: bool = True) -> ParseResult:
            self.calls += 1
            if self.calls == 3:
                try:
                    raise httpx.ReadTimeout("timed out")
                except httpx.ReadTimeout as exc:
                    raise ParseError("the query model server is unreachable: timed out") from exc
            return ParseResult(fields=QueryFields(gender="female", english="a woman"), raw="{}", model="slow",
                               took_ms=12_000 + 6_000 * self.calls)

    queries = [{"id": f"q{i}", "text": "مرة", "fields": {"gender": "female"}} for i in range(5)]
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(ollama))
    seen = []
    out = run_model(client, "Slow", "slow", queries, glossary=True, limit_s=20, parser=SlowParser(), on_row=seen.append)
    assert out["status"] == ("too slow on this Mac for query parsing: 21.0 s per query measured on 2 queries; "
                             "query q2 passed the 20 s limit")
    assert [r["id"] for r in out["rows"]] == ["q0", "q1"] and out["partial_summary"]["queries"] == 2
    assert out["memory_gb"] == 21.6 and len(seen) == 2 and seen[-1]["status"] == "running"
    assert pick([out]) is None  # a stopped model is never picked


def test_the_gguf_template_patch_swaps_raw_tags_for_same_length_comments(tmp_path):
    from avs.tools.patch_gguf_template import patch

    template = b'{%- if tools %}{% raw %}{"name": <function-name>}{% endraw %}{%- endif %}'
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF" + b"\0" * 100 + template + b"\1" * 1000)
    out = tmp_path / "patched.gguf"
    assert [tag for tag, _ in patch(source, out)] == ["{% raw %}", "{% endraw %}"]
    patched = out.read_bytes()
    assert len(patched) == source.stat().st_size and b"raw" not in patched
    assert b'{#     #}{"name": <function-name>}{#        #}' in patched
    assert source.read_bytes().count(b"{% raw %}") == 1  # the source is untouched

    (tmp_path / "plain.gguf").write_bytes(b"GGUF" + template.replace(b"{% raw %}", b""))
    with pytest.raises(ValueError, match="exactly one"):
        patch(tmp_path / "plain.gguf", tmp_path / "plain-out.gguf")
    assert not (tmp_path / "plain-out.gguf").exists()


def test_candidates_are_chosen_by_label_ollama_name_or_label_equals_name():
    assert select_candidates(["Fanar 2 27B=fanar-2-27b-patched:q4_K_M", "qwen3.6:35b-a3b-q4_K_M", "Gemma 4 26B-A4B"]) == {
        "Fanar 2 27B": "fanar-2-27b-patched:q4_K_M", "Qwen3.6-35B-A3B": "qwen3.6:35b-a3b-q4_K_M",
        "Gemma 4 26B-A4B": "gemma4:26b-a4b-it-q4_K_M"}
