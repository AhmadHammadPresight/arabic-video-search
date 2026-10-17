from __future__ import annotations

import sqlite3

import pytest

from avs.api import db


@pytest.fixture
def conn(settings):
    db.init_db(settings.db_path)
    with db.connect(settings.db_path) as c:
        yield c


def add(conn, rid="r1", camera="G1"):
    return db.create_recording(conn, rid, camera, "Lobby", "2018-03-15T14:52:30", "a.mp4", f"uploads/{rid}.mp4", 60.0, 640, 480)


def test_upload_creates_a_queued_recording_and_job(conn):
    rec = add(conn)
    assert rec["status"] == "queued" and rec["progress"] == 0
    job = db.claim_job(conn)
    assert job["recording_id"] == "r1" and job["status"] == "running"
    assert db.claim_job(conn) is None  # claimed once only


def test_interrupted_jobs_are_queued_again(conn):
    add(conn)
    job = db.claim_job(conn)
    db.update_recording(conn, "r1", status="indexing")
    assert db.requeue_running_jobs(conn) == 1
    assert db.get_recording(conn, "r1")["status"] == "queued"
    assert db.claim_job(conn)["id"] == job["id"]


def test_tracks_replace_and_cascade_on_delete(conn):
    add(conn)
    track = {"track_id": "person-1", "label": "person", "t_start": 1.0, "t_end": 2.0,
             "boxes": [[1.0, 0.1, 0.1, 0.2, 0.5]], "crops": [{"t": 1.0, "box": [0.1, 0.1, 0.2, 0.5], "path": "crops/r1/p.jpg", "embedding": [1.0]}]}
    db.replace_tracks(conn, "r1", [track])
    db.replace_tracks(conn, "r1", [track])
    tracks = db.tracks_for_recording(conn, "r1")
    assert len(tracks) == 1 and tracks[0]["track_key"] == "r1/person-1"
    assert "embedding" not in tracks[0]["crops"][0]
    db.delete_recording(conn, "r1")
    assert db.tracks_for_recording(conn, "r1") == [] and db.get_recording(conn, "r1") is None


def test_audit_log_is_append_only(conn):
    entry = db.append_audit(conn, "officer", "search", {"query": "رجل يرتدي قميص أحمر"}, search_id="s1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE audit_log SET actor = 'someone else' WHERE id = ?", (entry,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM audit_log")
    [row] = db.list_audit(conn)
    assert row["actor"] == "officer" and row["details"]["query"] == "رجل يرتدي قميص أحمر"
    assert db.find_audit(conn, "s1")["id"] == entry
