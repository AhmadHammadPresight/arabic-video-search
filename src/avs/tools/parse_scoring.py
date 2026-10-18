"""Score the query model's fields against the expected fields written in eval/queries.yaml.

Each query's `fields` block names what the text says; anything it leaves out is expected empty.
Eleven fields are scored for every query, so inventing a detail counts as an error too:
subject, gender, age band, upper garment, upper colours, lower garment, lower colours, hair colour,
hair length, headwear and carried objects. `other` (free text) and the English rewrite are not
scored automatically; the comparison report lists every rewrite for reading.

An expected scalar may be a list of acceptable values (`garment: [hoodie, jacket]`), and an
expected colour list may be a list of acceptable lists (`colours: [[], [green]]`).
An age band of null and "adult" count as the same answer: both mean no special age.
"""

from __future__ import annotations

from avs.fields import QueryFields

SCORED_FIELDS = (
    "subject", "gender", "age_band", "upper_garment", "upper_colours", "lower_garment", "lower_colours",
    "hair_colour", "hair_length", "headwear", "carried",
)


def _flat(fields: dict) -> dict:
    upper, lower = fields.get("upper") or {}, fields.get("lower") or {}
    hair, headwear = fields.get("hair") or {}, fields.get("headwear") or {}
    return {
        "subject": fields.get("subject", "person"),
        "gender": fields.get("gender"),
        "age_band": fields.get("age_band"),
        "upper_garment": upper.get("garment"),
        "upper_colours": upper.get("colours", []),
        "lower_garment": lower.get("garment"),
        "lower_colours": lower.get("colours", []),
        "hair_colour": hair.get("colour"),
        "hair_length": hair.get("length"),
        "headwear": {"type": headwear.get("type"), "colours": headwear.get("colours", [])},
        "carried": fields.get("carried", []),
    }


def _scalar_ok(expected, got, field: str) -> bool:
    accepted = expected if isinstance(expected, list) else [expected]
    if field == "age_band":
        accepted = [None if a == "adult" else a for a in accepted]
        got = None if got == "adult" else got
    return got in accepted


def _colours_ok(expected, got: list) -> bool:
    options = expected if expected and isinstance(expected[0], list) else [expected]
    return any(set(o) == set(got) for o in options)


def _carried_ok(expected: list, got: list) -> bool:
    if len(expected) != len({g["object"] for g in got}):
        return False
    remaining = list(got)
    for item in expected:
        objects = item["object"] if isinstance(item["object"], list) else [item["object"]]
        match = next((g for g in remaining if g["object"] in objects and _colours_ok(item.get("colours", []), g.get("colours", []))), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def score_fields(expected: dict, got: QueryFields) -> dict[str, bool]:
    """Per-field correctness of one parse."""
    want, have = _flat(expected), _flat(got.model_dump())
    out = {}
    for field in SCORED_FIELDS:
        if field in ("upper_colours", "lower_colours"):
            out[field] = _colours_ok(want[field], have[field])
        elif field == "headwear":
            out[field] = (_scalar_ok(want[field]["type"], have[field]["type"], field)
                          and _colours_ok(want[field]["colours"], have[field]["colours"]))
        elif field == "carried":
            out[field] = _carried_ok(want[field], have[field])
        else:
            out[field] = _scalar_ok(want[field], have[field], field)
    return out


def mentioned(expected: dict) -> set[str]:
    """The fields a query actually names (non-empty in its expected block)."""
    want = _flat(expected)
    names = set()
    for field, value in want.items():
        if field == "subject":
            if value != "person":
                names.add(field)
        elif field == "headwear":
            if value["type"] or value["colours"]:
                names.add(field)
        elif value not in (None, [], [None]):
            names.add(field)
    return names
