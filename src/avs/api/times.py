"""Camera times are local wall-clock times at the site, stored without a UTC offset.

For range filters they become epoch seconds by reading the wall-clock time as if it were UTC;
that keeps the arithmetic simple and round-trips exactly back to the same wall-clock text.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def parse_local_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is not None:
        raise ValueError("give the camera's local time without a UTC offset, such as 2018-03-15T14:52:30")
    return dt


def to_epoch(dt: datetime) -> float:
    return dt.replace(tzinfo=UTC).timestamp()


def from_epoch(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, UTC).replace(tzinfo=None)
    return dt.isoformat(timespec="milliseconds" if dt.microsecond else "seconds")


def offset(start_time: str, seconds: float) -> float:
    """Epoch seconds of a moment `seconds` into a recording that started at `start_time`."""
    return to_epoch(parse_local_time(start_time) + timedelta(seconds=seconds))
