"""The hybrid search on an in-memory Qdrant with a stubbed query model and embedder."""

from __future__ import annotations

import pytest
from helpers import unit

from avs.api.hybrid import run_hybrid
from avs.api.query_parser import ParseError, ParseResult
from avs.api.search import SearchQuery, run_search
from avs.api.trace import Trace
from avs.api.worker import build_points
from avs.fields import QueryFields

GREY, GREYISH, BLUE = unit(1, 0, 0, 0), unit(1, 0.25, 0, 0), unit(0, 1, 0, 0)
WOMAN_GREY = {"subject": "person", "gender": "female", "upper": {"garment": "jacket", "colours": ["grey"]}}
MAN_GREY = {"subject": "person", "gender": "male", "upper": {"garment": "jacket", "colours": ["grey"]}}
WOMAN_RED = {"subject": "person", "gender": "female", "upper": {"garment": "jacket", "colours": ["red"]}}


def recording(rid: str, camera: str) -> dict:
    return {"id": rid, "camera_id": camera, "place": "lobby", "start_time": "2018-03-07T17:00:00",
            "duration_s": 300.0, "playback_path": f"playback/{rid}.mp4"}


def track(track_id: str, label: str, vectors: list, attrs: dict | None, t0: float = 1.0) -> dict:
    out = {"track_id": track_id, "label": label, "t_start": t0, "t_end": t0 + 5,
           "crops": [{"t": t0 + i, "box": [0.1, 0.1, 0.3, 0.9], "path": f"crops/r1/{track_id}-{i}.jpg", "embedding": v}
                     for i, v in enumerate(vectors)]}
    if attrs is not None:
        out["attributes"] = {"source": "test", "attributes": attrs}
    return out


@pytest.fixture
def indexed(store):
    rec = recording("r1", "G331")
    tracks = [
        # Most similar to "grey", but a man: the gender mismatch pushes him down.
        track("person-1", "person", [GREY], MAN_GREY),
        # A woman in grey, slightly less similar: should win.
        track("person-2", "person", [GREYISH, BLUE], WOMAN_GREY),
        # A woman in red with the same similarity as person-2: loses on the colour.
        track("person-3", "person", [GREYISH], WOMAN_RED),
        # Never described: kept, unknown on every field.
        track("person-4", "person", [unit(1, 0.6, 0, 0)], None),
        track("bag-1", "handbag", [GREY], {"subject": "bag", "carried": [{"object": "handbag", "colours": ["grey"]}]}),
    ]
    store.replace_recording("r1", [p for t in tracks for p in build_points(rec, t)])
    return {"r1": rec}


def embed(texts: list[str]) -> list[list[float]]:
    return [GREY if "grey" in t or "سكني" in t else BLUE for t in texts]


def parser(fields: QueryFields):
    def parse(text: str) -> ParseResult:
        return ParseResult(fields=fields, raw=fields.model_dump_json(), model="stub", took_ms=7)
    return parse


def run(store, indexed, fields, text="مرة لابسة جاكيت سكني", **query):
    trace = Trace("s1", "hybrid", {"query": text})
    outcome = run_hybrid(store, embed, parser(fields) if isinstance(fields, QueryFields) else fields, indexed,
                         SearchQuery(text, **query), trace, "siglip-test")
    return outcome, trace.finish()


def test_filters_then_fuses_similarity_with_attributes(store, indexed):
    fields = QueryFields(gender="female", upper={"garment": "jacket", "colours": ["grey"]}, english="a woman in a grey jacket")
    outcome, _ = run(store, indexed, fields)
    keys = [r["track_key"] for r in outcome.results]
    assert outcome.mode == "hybrid" and outcome.parsed.fields == fields
    assert "r1/bag-1" not in keys  # the subject is a person
    assert keys[0] == "r1/person-2" and keys.index("r1/person-3") > keys.index("r1/person-2")
    man = outcome.results[keys.index("r1/person-1")]  # more similar, but described as a man: kept, ranked lower
    assert man["similarity"] > outcome.results[0]["similarity"] and man["score"] < outcome.results[0]["score"]
    assert {w["field"]: w["result"] for w in man["why"]}["gender"] == "mismatch"
    assert "r1/person-4" in keys  # no attributes: kept
    top = outcome.results[0]
    assert top["attribute_score"] == 1.0 and top["described"] and top["similarity"] > 0
    assert {w["field"]: w["result"] for w in top["why"]} == {"gender": "match", "upper_colour": "match", "upper_garment": "match"}
    undescribed = next(r for r in outcome.results if r["track_key"] == "r1/person-4")
    assert not undescribed["described"] and {w["result"] for w in undescribed["why"]} == {"unknown"}
    # The double-check gets the track's best-matching crops first.
    assert outcome.check_crops["r1/person-2"] == ["crops/r1/person-2-0.jpg", "crops/r1/person-2-1.jpg"]


def test_a_bag_query_searches_bag_tracks_only(store, indexed):
    fields = QueryFields(subject="bag", carried=[{"object": "handbag", "colours": ["grey"]}], english="a grey handbag")
    outcome, trace = run(store, indexed, fields, text="شنطة يد رمادية")
    assert [r["track_key"] for r in outcome.results] == ["r1/bag-1"]
    filters = next(s for s in trace["stages"] if s["key"] == "filter")["output"]
    assert filters["labels"] == ["backpack", "handbag", "suitcase"] and filters["exclude_gender"] is None


def test_camera_and_time_filters_still_apply(store, indexed):
    fields = QueryFields(english="a person in grey")
    outcome, _ = run(store, indexed, fields, text="grey", cameras=["G999"])
    assert outcome.results == []
    outcome, _ = run(store, indexed, fields, text="grey", time_from="2018-03-07T18:00:00")
    assert outcome.results == []


def test_trace_records_every_stage_in_order_with_its_output(store, indexed):
    fields = QueryFields(gender="female", upper={"colours": ["grey"]}, english="a woman in grey")
    outcome, trace = run(store, indexed, fields, limit=2)
    keys = [s["key"] for s in trace["stages"]]
    assert keys == ["input", "parse", "embed", "filter", "candidates", "attributes", "fusion", "ranking"]
    assert all(s["ms"] >= 0 and s["title_en"] and s["title_ar"] for s in trace["stages"])
    stage = {s["key"]: s["output"] for s in trace["stages"]}
    assert stage["input"]["script"] == "Arabic"
    assert stage["parse"]["english"] == "a woman in grey" and stage["parse"]["model"] == "stub"
    assert [t["role"] for t in stage["embed"]["texts"]] == ["English rewrite", "original text"]
    assert stage["embed"]["texts"][0]["dim"] == 4 and stage["embed"]["texts"][0]["norm"] == pytest.approx(1.0)
    assert stage["filter"]["exclude_gender"] is None and stage["filter"]["labels"] == ["person"]
    assert stage["candidates"]["tracks"] == 4 and stage["candidates"]["rows"][0]["sim_english"] is not None
    assert stage["attributes"]["described_tracks"] == 3 and stage["attributes"]["query_fields"] == ["woman", "grey top"]
    assert "z(similarity)" in stage["fusion"]["formula"] and len(stage["fusion"]["rows"]) == 4
    assert [r["track_key"] for r in stage["ranking"]] == [r["track_key"] for r in outcome.results] and len(outcome.results) == 2


def test_parse_failure_falls_back_to_option_a_and_says_so(store, indexed):
    def broken(text: str):
        raise ParseError("the query model server is unreachable: refused", raw=None)

    outcome, trace = run(store, indexed, broken, text="grey")
    assert outcome.mode == "baseline" and outcome.parsed is None
    assert "r1/bag-1" in [r["track_key"] for r in outcome.results]  # option A has no subject filter
    assert trace["mode"] == "baseline" and trace["notes"]
    assert [s["key"] for s in trace["stages"]] == ["input", "parse", "embed", "filter", "candidates", "ranking"]


def test_tracks_without_usable_attributes_count_as_undescribed(store):
    rec = recording("r1", "G331")
    tracks = [
        # An unreadable vision reply stored as an empty description by earlier builds: nothing to compare.
        track("person-1", "person", [GREY], {"subject": "person", "other": ["description: a man in"]}),
        # A value outside the current vocabulary (indexed before a schema change) must not break the search.
        track("person-2", "person", [GREYISH], {"subject": "person", "upper": {"garment": "kilt", "colours": ["teal"]}}),
        track("person-3", "person", [unit(1, 0.6, 0, 0)], {"subject": "person", "gender": "male"}),
    ]
    store.replace_recording("r1", [p for t in tracks for p in build_points(rec, t)])
    fields = QueryFields(gender="female", headwear={"type": "beanie"}, carried=[{"object": "backpack"}], english="a woman in a beanie")
    outcome, _ = run(store, {"r1": rec}, fields)
    by_key = {r["track_key"]: r for r in outcome.results}
    for key in ("r1/person-1", "r1/person-2"):
        assert not by_key[key]["described"] and by_key[key]["attribute_score"] == 0.0
        assert {w["result"] for w in by_key[key]["why"]} == {"unknown"}  # not "missing" for the beanie and backpack
    assert by_key["r1/person-3"]["described"] and by_key["r1/person-3"]["attribute_score"] < 0


def test_an_unreadable_description_is_not_put_on_the_points():
    rec = recording("r1", "G331")
    unreadable = track("person-1", "person", [GREY], None) | {
        "attributes": {"source": "vlm:test", "attributes": None, "error": "the vision model's reply could not be read"}}
    assert all("attributes" not in p.payload for p in build_points(rec, unreadable))


def test_an_invalid_time_window_fails_before_any_model_is_called(store, indexed):
    def never(*_):
        raise AssertionError("no model may be called for an invalid request")

    bad = SearchQuery("x", time_from="2018-03-07T18:00:00", time_to="2018-03-07T17:00:00")
    with pytest.raises(ValueError, match="before"):
        run_hybrid(store, never, never, indexed, bad, Trace("s1", "hybrid", {}), "")
    with pytest.raises(ValueError, match="before"):
        run_search(store, never, indexed, bad, trace=Trace("s2", "baseline", {}))


def test_the_gender_filter_can_be_switched_on(store, indexed, monkeypatch):
    monkeypatch.setattr("avs.api.hybrid.GENDER_FILTER", True)
    fields = QueryFields(gender="female", upper={"colours": ["grey"]}, english="a woman in grey")
    outcome, trace = run(store, indexed, fields)
    assert "r1/person-1" not in [r["track_key"] for r in outcome.results]  # described as a man
    assert "r1/person-4" in [r["track_key"] for r in outcome.results]  # no attributes: kept
    assert next(s for s in trace["stages"] if s["key"] == "filter")["output"]["qdrant_filter"]["must_not"]
