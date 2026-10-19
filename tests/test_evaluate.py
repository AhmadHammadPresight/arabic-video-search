from __future__ import annotations

import pytest

from avs.api.times import parse_local_time, to_epoch
from avs.tools.evaluate import Anchor, load_queries, match_anchor, score_query, summarise

REC = {"id": "r1", "camera_id": "G331", "start_time": "2018-03-07T17:00:00", "duration_s": 300.0}
TRACKS = {"r1": [
    {"track_key": "r1/person-1", "boxes": [[10.0, 0.40, 0.20, 0.60, 0.90], [10.25, 0.41, 0.20, 0.61, 0.90]]},
    # A wider box around the same point (someone standing behind): the closer centre must win.
    {"track_key": "r1/person-2", "boxes": [[10.0, 0.30, 0.10, 0.90, 0.95]]},
    {"track_key": "r1/person-3", "boxes": [[50.0, 0.40, 0.20, 0.60, 0.90]]},
]}


def at(clock: str) -> float:
    return to_epoch(parse_local_time(f"2018-03-07T{clock}"))


def test_anchor_picks_the_track_containing_the_point_at_that_moment():
    assert match_anchor(Anchor("G331", at("17:00:10.2"), 0.5, 0.55), [REC], TRACKS) == "r1/person-1"
    assert match_anchor(Anchor("G331", at("17:00:50"), 0.5, 0.5), [REC], TRACKS) == "r1/person-3"


def test_anchor_misses_on_wrong_camera_time_or_place():
    assert match_anchor(Anchor("G421", at("17:00:10"), 0.5, 0.5), [REC], TRACKS) is None
    assert match_anchor(Anchor("G331", at("17:00:20"), 0.5, 0.5), [REC], TRACKS) is None
    assert match_anchor(Anchor("G331", at("17:00:10"), 0.05, 0.05), [REC], TRACKS) is None


def test_scores():
    s = score_query(["a", "b", "c", "d", "e", "f"], {"c", "e"})
    assert s == {"hit_at_10": 1.0, "precision_at_5": 0.4, "reciprocal_rank": pytest.approx(1 / 3), "first_relevant_rank": 3}
    assert score_query(["a"], set())["hit_at_10"] == 0.0


def test_summary_skips_no_match_queries():
    rows = [
        {"register": "formal", "expect": "match", "hit_at_10": 1.0, "precision_at_5": 0.4, "reciprocal_rank": 1.0},
        {"register": "gulf", "expect": "match", "hit_at_10": 0.0, "precision_at_5": 0.0, "reciprocal_rank": 0.0},
        {"register": "gulf", "expect": "no-match", "hit_at_10": 0.0, "precision_at_5": 0.0, "reciprocal_rank": 0.0},
    ]
    s = summarise(rows, "register")
    assert s["all"] == {"queries": 2, "hit_at_10": 0.5, "precision_at_5": 0.2, "mrr": 0.5}
    assert s["gulf"]["queries"] == 1


QUERIES = __import__("avs.settings", fromlist=["REPO_ROOT"]).REPO_ROOT / "eval" / "queries.yaml"
# The project owner's own queries, which must stay exactly as written.
OWNER = {
    "c01": "امراة تلبس جاكيت سكني او ابيض",
    "c02": "معهها شنطة بنية او لونها بيج",
    "c05": "woman wearing red jacket",
    "c09": "man has a walkig suitecase bag",
    "c13": "man wearing brown trousers",
    "c17": "woman with blonde hair",
    "c21": "man has an ipad",
}


def test_seeded_query_file_is_consistent():
    targets, queries = load_queries(QUERIES)
    assert 80 <= len(queries) <= 120
    assert len({q["id"] for q in queries}) == len(queries)
    assert {"formal", "jordanian", "gulf", "mixed", "english"} == {q["register"] for q in queries}
    for name, anchors in targets.items():
        assert anchors, name
    assert {q["id"]: q["text"] for q in queries if q.get("source") == "owner"} == OWNER
    assert sum(q.get("added") == "phase2" for q in queries) == len(queries) - 48


def test_every_query_has_expected_fields_the_scorer_understands():
    from avs.fields import QueryFields
    from avs.tools.parse_scoring import SCORED_FIELDS, score_fields

    allowed = set(QueryFields.model_fields) - {"english", "other"}
    for q in load_queries(QUERIES)[1]:
        assert set(q["fields"]) <= allowed, q["id"]
        # A parse that names nothing is wrong on every field the query does name, and nothing crashes.
        assert set(score_fields(q["fields"], QueryFields(english="x"))) == set(SCORED_FIELDS)
