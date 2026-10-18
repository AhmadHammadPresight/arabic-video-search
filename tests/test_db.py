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


def test_a_phase_1_database_gains_the_new_columns_and_keeps_its_rows(settings):
    old = """
    CREATE TABLE recordings (id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, place TEXT NOT NULL, start_time TEXT NOT NULL,
        original_name TEXT NOT NULL, video_path TEXT NOT NULL, playback_path TEXT, duration_s REAL, width INTEGER,
        height INTEGER, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, error TEXT, track_count INTEGER,
        created_at TEXT NOT NULL, indexed_at TEXT);
    CREATE TABLE tracks (track_key TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings (id) ON DELETE CASCADE,
        track_id TEXT NOT NULL, label TEXT NOT NULL, t_start REAL NOT NULL, t_end REAL NOT NULL, boxes TEXT NOT NULL, crops TEXT NOT NULL);
    INSERT INTO recordings VALUES ('r1', 'G1', 'Lobby', '2018-03-15T14:52:30', 'a.mp4', 'uploads/r1.mp4', NULL, 60, 640, 480,
        'indexed', 1, NULL, 1, '2026-10-17T00:00:00+00:00', NULL);
    INSERT INTO tracks VALUES ('r1/person-1', 'r1', 'person-1', 'person', 1, 2, '[]', '[]');
    """
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with db.connect(settings.db_path) as c:
        c.executescript(old)
    db.init_db(settings.db_path)
    db.init_db(settings.db_path)  # safe to run again
    with db.connect(settings.db_path) as c:
        [track] = db.tracks_for_recording(c, "r1")
        assert track["attributes"] is None and track["selection"] is None
        assert "index_stats" not in db.get_recording(c, "r1") and db.get_index_stats(c, "r1") is None
        db.update_recording(c, "r1", index_stats={"frames": 12})
        assert db.get_index_stats(c, "r1") == {"frames": 12}


def test_traces_are_saved_replaced_and_read_back(conn):
    db.save_trace(conn, "s1", "hybrid", {"stages": [], "note": "رجل"})
    db.save_trace(conn, "s1", "hybrid", {"stages": [{"key": "check"}]})
    assert db.get_trace(conn, "s1") == {"stages": [{"key": "check"}]} and db.get_trace(conn, "nope") is None


def test_tracks_keep_attributes_selection_and_crop_quality(conn):
    add(conn)
    track = {"track_id": "person-1", "label": "person", "t_start": 1.0, "t_end": 2.0, "boxes": [],
             "crops": [{"t": 1.0, "box": [0, 0, 1, 1], "path": "crops/r1/p.jpg", "quality": 0.7, "embedding": [1.0]}],
             "attributes": {"source": "vlm:x", "attributes": {"gender": "female"}}, "selection": {"min_gap": 0.5}}
    db.replace_tracks(conn, "r1", [track])
    [stored] = db.tracks_by_key(conn, ["r1/person-1"]).values()
    assert stored["attributes"]["attributes"]["gender"] == "female" and stored["selection"] == {"min_gap": 0.5}
    assert stored["crops"] == [{"t": 1.0, "box": [0, 0, 1, 1], "path": "crops/r1/p.jpg", "quality": 0.7}]
    assert db.tracks_by_key(conn, []) == {}
