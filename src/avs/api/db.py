"""SQLite store for recordings, the indexing job queue, tracks and the audit log.

One file (`data/avs.db`) in WAL mode is shared by the API and the worker.
The audit log is append-only: triggers reject every UPDATE and DELETE on it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id            TEXT PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    place         TEXT NOT NULL,
    start_time    TEXT NOT NULL,   -- camera's local wall-clock time, ISO 8601 without offset
    original_name TEXT NOT NULL,
    video_path    TEXT NOT NULL,   -- relative to the data directory
    playback_path TEXT,            -- browser-playable H.264 copy, relative to the data directory
    duration_s    REAL,
    width         INTEGER,
    height        INTEGER,
    status        TEXT NOT NULL,   -- queued | indexing | indexed | failed
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    track_count   INTEGER,
    created_at    TEXT NOT NULL,
    indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,    -- index
    recording_id TEXT NOT NULL,
    status       TEXT NOT NULL,    -- queued | running | done | failed
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, id);

CREATE TABLE IF NOT EXISTS tracks (
    track_key    TEXT PRIMARY KEY, -- <recording_id>/<track_id>
    recording_id TEXT NOT NULL REFERENCES recordings (id) ON DELETE CASCADE,
    track_id     TEXT NOT NULL,
    label        TEXT NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    boxes        TEXT NOT NULL,    -- JSON [[t, x1, y1, x2, y2], ...], box corners as fractions of the frame
    crops        TEXT NOT NULL     -- JSON [{t, box, path}, ...]
);
CREATE INDEX IF NOT EXISTS tracks_recording ON tracks (recording_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,       -- upload | search | confirm | reject | reindex | delete
    search_id TEXT,
    details   TEXT NOT NULL        -- JSON
);
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# Recordings


def create_recording(
    conn: sqlite3.Connection,
    recording_id: str,
    camera_id: str,
    place: str,
    start_time: str,
    original_name: str,
    video_path: str,
    duration_s: float,
    width: int,
    height: int,
) -> dict:
    with transaction(conn):
        conn.execute(
            """INSERT INTO recordings (id, camera_id, place, start_time, original_name, video_path,
                   duration_s, width, height, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
            (recording_id, camera_id, place, start_time, original_name, video_path,
             duration_s, width, height, now()),
        )
        _enqueue(conn, recording_id)
    return get_recording(conn, recording_id)


def get_recording(conn: sqlite3.Connection, recording_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    return dict(row) if row else None


def list_recordings(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM recordings ORDER BY camera_id, start_time").fetchall()
    return [dict(r) for r in rows]


def update_recording(conn: sqlite3.Connection, recording_id: str, **fields) -> None:
    if not fields:
        return
    columns = ", ".join(f"{name} = ?" for name in fields)
    conn.execute(f"UPDATE recordings SET {columns} WHERE id = ?", (*fields.values(), recording_id))


def delete_recording(conn: sqlite3.Connection, recording_id: str) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM jobs WHERE recording_id = ?", (recording_id,))
        conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))


def requeue_recording(conn: sqlite3.Connection, recording_id: str) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE recordings SET status = 'queued', progress = 0, error = NULL WHERE id = ?", (recording_id,)
        )
        _enqueue(conn, recording_id)


# Job queue


def _enqueue(conn: sqlite3.Connection, recording_id: str) -> None:
    conn.execute(
        "INSERT INTO jobs (kind, recording_id, status, created_at) VALUES ('index', ?, 'queued', ?)",
        (recording_id, now()),
    )


def claim_job(conn: sqlite3.Connection) -> dict | None:
    """Take the oldest queued job; BEGIN IMMEDIATE makes the claim safe across processes."""
    with transaction(conn):
        row = conn.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        conn.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?", (now(), row["id"]))
    return dict(row) | {"status": "running"}


def finish_job(conn: sqlite3.Connection, job_id: int, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?", (status, error, now(), job_id)
    )


def requeue_running_jobs(conn: sqlite3.Connection) -> int:
    """Put back jobs left running by a worker that stopped mid-way."""
    with transaction(conn):
        cur = conn.execute("UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'")
        conn.execute("UPDATE recordings SET status = 'queued', progress = 0 WHERE status = 'indexing'")
    return cur.rowcount


# Tracks


def replace_tracks(conn: sqlite3.Connection, recording_id: str, tracks: list[dict]) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM tracks WHERE recording_id = ?", (recording_id,))
        conn.executemany(
            """INSERT INTO tracks (track_key, recording_id, track_id, label, t_start, t_end, boxes, crops)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    f"{recording_id}/{t['track_id']}", recording_id, t["track_id"], t["label"],
                    t["t_start"], t["t_end"], json.dumps(t["boxes"]),
                    json.dumps([{"t": c["t"], "box": c["box"], "path": c["path"]} for c in t["crops"]]),
                )
                for t in tracks
            ],
        )


def tracks_for_recording(conn: sqlite3.Connection, recording_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM tracks WHERE recording_id = ? ORDER BY t_start, track_id", (recording_id,)
    ).fetchall()
    return [dict(r) | {"boxes": json.loads(r["boxes"]), "crops": json.loads(r["crops"])} for r in rows]


# Audit log


def append_audit(
    conn: sqlite3.Connection, actor: str, action: str, details: dict, search_id: str | None = None
) -> int:
    cur = conn.execute(
        "INSERT INTO audit_log (at, actor, action, search_id, details) VALUES (?, ?, ?, ?, ?)",
        (now(), actor, action, search_id, json.dumps(details, ensure_ascii=False)),
    )
    return int(cur.lastrowid)


def list_audit(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"details": json.loads(r["details"])} for r in rows]


def find_audit(conn: sqlite3.Connection, search_id: str, action: str = "search") -> dict | None:
    row = conn.execute(
        "SELECT * FROM audit_log WHERE search_id = ? AND action = ? ORDER BY id LIMIT 1", (search_id, action)
    ).fetchone()
    return dict(row) | {"details": json.loads(row["details"])} if row else None
