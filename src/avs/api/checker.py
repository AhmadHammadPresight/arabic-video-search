"""The vision-model double-check: Qwen3-VL looks at the best crops of the top results and re-ranks them.

The search answers at once with the hybrid ranking; this runs afterwards in a background thread,
one result at a time, so the page fills in each verdict as it arrives.

For every result the vision model answers yes, no or unclear for each detail of the parsed query
("a woman", "grey or white jacket", "beige bag"). A clear "no" on a key detail rejects the result,
"yes" on every key detail confirms it, and anything else leaves it unsure. Confirmed results move
up and rejected ones drop, by adding or subtracting `CHECK_BONUS` to the fused score (about two
standard deviations of similarity), so the officer still sees every result and decides.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from avs.api import db
from avs.api.trace import STAGES, replace_stage
from avs.fields import QueryFields

log = logging.getLogger("avs.checker")

CHECK_BONUS = 2.0
KEEP_JOBS = 200
_FRIENDLY = {"tablet": "a tablet such as an iPad", "phone": "a phone", "cart": "pushing a cart", "stroller": "pushing a stroller",
             "bicycle": "with a bicycle"}


@dataclass(frozen=True)
class Detail:
    text: str  # what the vision model is asked about
    field: str
    key: bool  # a clear "no" on a key detail rejects the result


def _or(colours: list[str]) -> str:
    return " or ".join(colours)


def details_for(fields: QueryFields) -> list[Detail]:
    out: list[Detail] = []
    if fields.subject == "person":
        if fields.gender:
            noun = {"male": "boy", "female": "girl"} if fields.age_band == "child" else {"male": "man", "female": "woman"}
            out.append(Detail(f"a {noun[fields.gender]}", "gender", True))
        elif fields.age_band == "child":
            out.append(Detail("a child", "age_band", True))
        if fields.age_band == "elderly":
            out.append(Detail("an elderly person", "age_band", False))
        for part, default in (("upper", "top"), ("lower", "trousers")):
            p = getattr(fields, part)
            if p.colours:
                out.append(Detail(f"{_or(p.colours)} {p.garment or default}", f"{part}_colour", True))
            elif p.garment:
                out.append(Detail(f"wearing {p.garment}", f"{part}_garment", False))
        if fields.hair.colour:
            out.append(Detail(f"{fields.hair.colour} hair", "hair_colour", True))
        if fields.hair.length:
            out.append(Detail(f"{fields.hair.length} hair", "hair_length", False))
        if fields.headwear.type:
            out.append(Detail(" ".join(x for x in (_or(fields.headwear.colours), fields.headwear.type) if x), "headwear", True))
    for item in fields.carried:
        if fields.subject == "bag":
            text = " ".join(x for x in ("a", _or(item.colours), item.object) if x)
        else:
            text = _FRIENDLY.get(item.object) or " ".join(x for x in ("carrying a", _or(item.colours), item.object) if x)
        out.append(Detail(text, "carried", True))
    out += [Detail(o, "other", False) for o in fields.other[:2]]
    unique: list[Detail] = []
    for d in out:
        if d.text not in {u.text for u in unique}:
            unique.append(d)
    return unique[:8]


def verdict_for(answers: dict[str, str], details: list[Detail]) -> str:
    key = [answers.get(d.text, "unclear") for d in details if d.key]
    if "no" in key:
        return "rejected"
    if key and all(a == "yes" for a in key):
        return "confirmed"
    return "unsure"


def rerank(results: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    """Results re-ordered by the fused score plus the check bonus; unchecked results keep their score."""
    bonus = {"confirmed": CHECK_BONUS, "rejected": -CHECK_BONUS}
    rescored = []
    for r in results:
        v = verdicts.get(r["track_key"])
        verdict = v["verdict"] if v else "not checked"
        rescored.append(r | {"rank_before_check": r["rank"], "check": verdict,
                             "score_checked": round(r["score"] + bonus.get(verdict, 0.0), 4)})
    rescored.sort(key=lambda r: -r["score_checked"])
    return [r | {"rank": i + 1} for i, r in enumerate(rescored)]


# A check in one of these states will not change any more.
FINISHED = ("done", "failed", "skipped", "cancelled", "interrupted")


class CheckRunner:
    """Runs double-checks one search at a time and keeps their progress in memory and in the trace.

    The page cancels the check of a search it no longer shows, so a refined query is not queued behind
    the old ones. Closing the runner cancels what is queued or running, so stopping the API never
    waits for a queue. Every check is stored in the trace when it is queued, so one cut off by a
    restart reads as "interrupted" rather than staying pending.
    """

    def __init__(self, check: Callable[..., dict], db_path: Path, top: int, model: str = "") -> None:
        self._check = check
        self._db_path = db_path
        self.top = top
        self.model = model
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="double-check")
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._stop: dict[str, str] = {}  # running checks asked to stop, with the reason
        self._closed = False
        self._lock = threading.Lock()

    def submit(self, search_id: str, fields: QueryFields, results: list[dict], crops: dict[str, list[str]]) -> dict:
        details = details_for(fields)
        to_check = results[: self.top] if details else []
        job = {"status": "queued" if to_check else "skipped", "total": len(to_check), "done": 0,
               "details": [asdict(d) for d in details], "verdicts": {}, "results": results, "model": self.model}
        if not details:
            job["reason"] = "the query names no detail the vision model can check"
        with self._lock:
            if self._closed and to_check:
                job |= {"status": "cancelled", "reason": "the server is stopping"}
            self._jobs[search_id] = job
            finished = [k for k, j in self._jobs.items() if j["status"] in FINISHED]
            for key in finished[: max(0, len(self._jobs) - KEEP_JOBS)]:
                del self._jobs[key]
        self._persist(search_id, self.public(job), audit=job["status"] != "queued")
        if job["status"] == "queued":
            self._executor.submit(self._run, search_id, job, fields, details, to_check, crops)
        return self.public(job)

    def status(self, search_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(search_id)
        if job is not None:
            return self.public(job)
        with db.connect(self._db_path) as conn:
            trace = db.get_trace(conn, search_id)
        stage = next((s for s in (trace or {}).get("stages", []) if s["key"] == "check"), None)
        if stage is None:
            return None
        if stage["output"].get("status") not in FINISHED:  # stored as queued or running by an earlier run of the server
            return stage["output"] | {"status": "interrupted", "reason": "the server stopped before the check finished"}
        return stage["output"]

    def cancel(self, search_id: str, reason: str = "cancelled from the page") -> dict | None:
        """Stop a queued or running check; verdicts already given are kept and the order is not changed."""
        with self._lock:
            job = self._jobs.get(search_id)
            if job is None:
                return None
            store_now = job["status"] == "queued"
            if store_now:
                job |= {"status": "cancelled", "reason": reason}
            elif job["status"] == "running":
                self._stop[search_id] = reason  # the thread stops before its next result and stores the outcome
        if store_now:
            self._persist(search_id, self.public(job))
        return self.public(job)

    def close(self) -> None:
        """Cancel every queued or running check and let the thread end without working through the queue."""
        with self._lock:
            self._closed = True
            pending = [k for k, j in self._jobs.items() if j["status"] in ("queued", "running")]
        for search_id in pending:
            self.cancel(search_id, "the server stopped")
        self._executor.shutdown(wait=False, cancel_futures=True)

    def public(self, job: dict) -> dict:
        with self._lock:
            return copy.deepcopy(job)

    def _run(self, search_id: str, job: dict, fields: QueryFields, details: list[Detail],
             to_check: list[dict], crops: dict[str, list[str]]) -> None:
        started = time.monotonic()
        with self._lock:
            if job["status"] != "queued":  # cancelled while it waited
                return
            job["status"] = "running"
        texts = [d.text for d in details]
        stopped = None
        try:
            for result in to_check:
                with self._lock:
                    stopped = self._stop.pop(search_id, None)
                if stopped:
                    break
                key = result["track_key"]
                try:
                    answer = self._check(crops=crops[key], subject=fields.subject, english=fields.english, details=texts)
                    verdict = verdict_for(answer["answers"], details)
                except Exception as exc:  # one failed check leaves that result unsure
                    log.warning("double-check of %s failed: %s", key, exc)
                    answer, verdict = {"answers": {}, "note": f"check failed: {exc}", "ms": None}, "unsure"
                with self._lock:
                    job["verdicts"][key] = {"verdict": verdict, "answers": answer.get("answers", {}),
                                            "note": answer.get("note", ""), "ms": answer.get("ms"),
                                            "raw": answer.get("raw"), "crops": [f"/media/{p}" for p in crops[key]]}
                    job["done"] += 1
            if stopped:
                final = "cancelled"
                with self._lock:
                    job["reason"] = stopped
            else:
                with self._lock:
                    job["results"] = rerank(job["results"], job["verdicts"])
                final = "done"
        except Exception as exc:
            log.exception("double-check of search %s failed", search_id)
            with self._lock:
                job["error"] = str(exc)
            final = "failed"
        finally:
            with self._lock:
                self._stop.pop(search_id, None)
        took_ms = round((time.monotonic() - started) * 1000)
        # Store the outcome before reporting it, so a finished check is always in the trace and the audit log.
        self._persist(search_id, self.public(job) | {"status": final, "took_ms": took_ms})
        with self._lock:
            job["status"], job["took_ms"] = final, took_ms

    def _persist(self, search_id: str, job: dict, audit: bool = True) -> None:
        counts = {v: sum(1 for x in job["verdicts"].values() if x["verdict"] == v) for v in ("confirmed", "unsure", "rejected")}
        with db.connect(self._db_path) as conn:
            trace = db.get_trace(conn, search_id)
            if trace is not None:
                ar, en = STAGES["check"]
                replace_stage(trace, {"key": "check", "title_ar": ar, "title_en": en, "ms": job.get("took_ms"),
                                      "output": job})
                if job["status"] == "done":
                    ar, en = STAGES["final"]
                    replace_stage(trace, {"key": "final", "title_ar": ar, "title_en": en, "ms": 0, "output": [
                        {"rank": r["rank"], "rank_before_check": r["rank_before_check"], "track_key": r["track_key"],
                         "check": r["check"], "score": r["score"], "score_checked": r["score_checked"],
                         "crop_url": r["crop_url"]} for r in job["results"]]})
                db.save_trace(conn, search_id, trace["mode"], trace)
            if not audit:
                return
            db.append_audit(conn, "vision-check", "check", {
                "status": job["status"], "model": self.model, "checked": job["done"], **counts,
                "took_ms": job.get("took_ms"),
                "order": [r["track_key"] for r in job["results"]] if job["status"] == "done" else None,
            }, search_id)
