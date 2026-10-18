"""The vision-model double-check: details asked, verdicts, re-ranking and the background runner, with the model stubbed."""

from __future__ import annotations

import threading
import time

import pytest

from avs.api import db
from avs.api.checker import CHECK_BONUS, FINISHED, CheckRunner, Detail, details_for, rerank, verdict_for
from avs.api.trace import Trace
from avs.fields import QueryFields


def result(key: str, rank: int, score: float) -> dict:
    return {"track_key": key, "rank": rank, "score": score, "crop_url": f"/media/crops/{key}.jpg"}


def test_details_follow_the_parsed_fields():
    fields = QueryFields(gender="female", upper={"garment": "jacket", "colours": ["grey", "white"]},
                         lower={"garment": "jeans"}, hair={"colour": "blonde"},
                         carried=[{"object": "bag", "colours": ["beige"]}, {"object": "tablet"}],
                         other=["glasses"], english="x")
    assert details_for(fields) == [
        Detail("a woman", "gender", True), Detail("grey or white jacket", "upper_colour", True),
        Detail("wearing jeans", "lower_garment", False), Detail("blonde hair", "hair_colour", True),
        Detail("carrying a beige bag", "carried", True), Detail("a tablet such as an iPad", "carried", True),
        Detail("glasses", "other", False),
    ]
    girl = QueryFields(gender="female", age_band="child", english="x")
    assert details_for(girl) == [Detail("a girl", "gender", True)]
    bag = QueryFields(subject="bag", carried=[{"object": "suitcase", "colours": ["brown"]}], gender="male", english="x")
    assert details_for(bag) == [Detail("a brown suitcase", "carried", True)]
    assert details_for(QueryFields(english="someone")) == []


def test_verdicts_hinge_on_key_details_only():
    details = [Detail("a woman", "gender", True), Detail("grey jacket", "upper_colour", True), Detail("glasses", "other", False)]
    assert verdict_for({"a woman": "yes", "grey jacket": "yes", "glasses": "no"}, details) == "confirmed"
    assert verdict_for({"a woman": "yes", "grey jacket": "no", "glasses": "yes"}, details) == "rejected"
    assert verdict_for({"a woman": "yes", "grey jacket": "unclear"}, details) == "unsure"
    assert verdict_for({}, details) == "unsure"
    assert verdict_for({"glasses": "yes"}, [Detail("glasses", "other", False)]) == "unsure"


def test_rerank_moves_confirmed_up_and_rejected_down():
    results = [result("a", 1, 1.5), result("b", 2, 1.2), result("c", 3, 1.0), result("d", 4, 0.2)]
    verdicts = {"a": {"verdict": "rejected"}, "b": {"verdict": "unsure"}, "c": {"verdict": "confirmed"}}
    ranked = rerank(results, verdicts)
    assert [r["track_key"] for r in ranked] == ["c", "b", "d", "a"]
    assert [r["rank"] for r in ranked] == [1, 2, 3, 4]
    assert ranked[0]["rank_before_check"] == 3 and ranked[0]["score_checked"] == pytest.approx(1.0 + CHECK_BONUS)
    assert ranked[2]["check"] == "not checked" and ranked[3]["score_checked"] == pytest.approx(1.5 - CHECK_BONUS)


@pytest.fixture
def stored(settings):
    db.init_db(settings.db_path)
    trace = Trace("s1", "hybrid", {"query": "x"})
    with trace.stage("ranking") as stage:
        stage["output"] = []
    with db.connect(settings.db_path) as conn:
        db.save_trace(conn, "s1", "hybrid", trace.finish())
    return settings


def wait(runner: CheckRunner, search_id: str) -> dict:
    deadline = time.monotonic() + 5
    while (status := runner.status(search_id))["status"] not in FINISHED:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return status


def test_runner_checks_the_top_results_in_the_background_and_stores_the_outcome(stored):
    calls = []
    release = threading.Event()

    def check(crops, subject, english, details):
        release.wait(5)
        calls.append(crops)
        if crops[0].endswith("b-0.jpg"):
            raise RuntimeError("vision model fell over")
        answer = "no" if crops[0].endswith("a-0.jpg") else "yes"
        return {"answers": {d: answer for d in details}, "note": f"saw {answer}", "ms": 12, "raw": "{}"}

    fields = QueryFields(gender="female", upper={"colours": ["grey"]}, english="a woman in grey")
    results = [result("a", 1, 2.0), result("b", 2, 1.0), result("c", 3, 0.5), result("d", 4, 0.1)]
    crops = {k: [f"crops/r/{k}-0.jpg", f"crops/r/{k}-1.jpg"] for k in "abcd"}
    runner = CheckRunner(check, stored.db_path, top=3, model="stub-vlm")
    first = runner.submit("s1", fields, results, crops)
    assert first["status"] in ("queued", "running") and first["total"] == 3 and first["done"] == 0
    assert [d["text"] for d in first["details"]] == ["a woman", "grey top"]
    release.set()
    status = wait(runner, "s1")
    assert len(calls) == 3  # only the top three were checked
    assert {k: v["verdict"] for k, v in status["verdicts"].items()} == {"a": "rejected", "b": "unsure", "c": "confirmed"}
    assert "fell over" in status["verdicts"]["b"]["note"]
    assert [r["track_key"] for r in status["results"]] == ["c", "b", "d", "a"]
    with db.connect(stored.db_path) as conn:
        trace = db.get_trace(conn, "s1")
        audit = db.list_audit(conn)[0]
    keys = [s["key"] for s in trace["stages"]]
    assert keys == ["ranking", "check", "final"]
    assert trace["stages"][2]["output"][0] == {"rank": 1, "rank_before_check": 3, "track_key": "c", "check": "confirmed",
                                               "score": 0.5, "score_checked": 2.5, "crop_url": "/media/crops/c.jpg"}
    assert audit["action"] == "check" and audit["actor"] == "vision-check" and audit["search_id"] == "s1"
    assert audit["details"] | {"took_ms": 0} == {"status": "done", "model": "stub-vlm", "checked": 3, "confirmed": 1,
                                                 "unsure": 1, "rejected": 1, "took_ms": 0, "order": ["c", "b", "d", "a"]}


def test_status_survives_a_restart_through_the_stored_trace(stored):
    runner = CheckRunner(lambda **kw: {"answers": {d: "yes" for d in kw["details"]}, "note": ""}, stored.db_path, top=2)
    runner.submit("s1", QueryFields(gender="male", english="a man"), [result("a", 1, 1.0)], {"a": ["crops/a.jpg"]})
    wait(runner, "s1")
    fresh = CheckRunner(lambda **kw: {}, stored.db_path, top=2)
    assert fresh.status("s1")["status"] == "done" and fresh.status("unknown") is None


def test_nothing_to_check_is_skipped(stored):
    runner = CheckRunner(lambda **kw: pytest.fail("must not be called"), stored.db_path, top=8)
    status = runner.submit("s1", QueryFields(english="someone"), [result("a", 1, 1.0)], {"a": ["x.jpg"]})
    assert status["status"] == "skipped" and status["total"] == 0 and "no detail" in status["reason"]


def save_trace(settings, search_id: str) -> None:
    trace = Trace(search_id, "hybrid", {"query": "x"})
    with db.connect(settings.db_path) as conn:
        db.save_trace(conn, search_id, "hybrid", trace.finish())


def test_a_cancelled_check_stops_and_a_queued_one_never_runs(stored):
    save_trace(stored, "s2")
    started, release, calls = threading.Event(), threading.Event(), []

    def check(crops, subject, english, details):
        calls.append(crops[0])
        started.set()
        release.wait(5)
        return {"answers": {d: "yes" for d in details}, "note": "", "ms": 1}

    fields = QueryFields(gender="male", english="a man")
    results = [result(k, i + 1, 1.0 - i / 10) for i, k in enumerate("abc")]
    crops = {k: [f"crops/{k}.jpg"] for k in "abc"}
    runner = CheckRunner(check, stored.db_path, top=3)
    runner.submit("s1", fields, results, crops)
    runner.submit("s2", fields, results, crops)
    assert started.wait(5) and runner.status("s2")["status"] == "queued"
    assert runner.cancel("s2")["status"] == "cancelled"  # a refined query: the old check is dropped
    assert runner.cancel("s1")["status"] == "running"  # stops before its next result
    release.set()
    first = wait(runner, "s1")
    assert first["status"] == "cancelled" and first["done"] == 1 and calls == ["crops/a.jpg"]
    assert [r["track_key"] for r in first["results"]] == ["a", "b", "c"]  # the order is not changed
    assert wait(runner, "s2")["status"] == "cancelled" and len(calls) == 1  # s2 never ran
    with db.connect(stored.db_path) as conn:
        audit = [e["details"]["status"] for e in db.list_audit(conn)]
        stored_s2 = db.get_trace(conn, "s2")
    assert sorted(audit) == ["cancelled", "cancelled"]
    assert next(s for s in stored_s2["stages"] if s["key"] == "check")["output"]["status"] == "cancelled"
    assert runner.cancel("unknown") is None


def test_closing_cancels_the_queue_and_a_restart_reports_cut_off_checks_as_interrupted(stored):
    save_trace(stored, "s2")
    started, release = threading.Event(), threading.Event()

    def check(crops, subject, english, details):
        started.set()
        release.wait(5)
        return {"answers": {d: "yes" for d in details}, "note": "", "ms": 1}

    fields = QueryFields(gender="male", english="a man")
    runner = CheckRunner(check, stored.db_path, top=2)
    runner.submit("s1", fields, [result("a", 1, 1.0), result("b", 2, 0.5)], {"a": ["a.jpg"], "b": ["b.jpg"]})
    runner.submit("s2", fields, [result("a", 1, 1.0)], {"a": ["a.jpg"]})
    assert started.wait(5)
    # The queued check is already in the trace, so a server killed now leaves it findable.
    assert CheckRunner(check, stored.db_path, top=2).status("s2")["status"] == "interrupted"
    runner.close()
    assert runner.status("s2")["status"] == "cancelled" and "stopped" in runner.status("s2")["reason"]
    release.set()
    assert wait(runner, "s1")["status"] == "cancelled" and runner.status("s1")["done"] == 1
    late = runner.submit("s3", fields, [result("a", 1, 1.0)], {"a": ["a.jpg"]})
    assert late["status"] == "cancelled"  # nothing new starts once the server is stopping
