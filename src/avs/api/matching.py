"""Attribute matching between a parsed query and a track's attributes, and the score fusion.

Each field the query names is compared with the track's value for that field:

- match (+1): the same value, or the query's general word covers it (trousers covers jeans);
- near (+0.4): a neighbouring value, such as beige for brown or a coat for a jacket;
- unknown (0): the track has no value for the field (not visible, or not described);
- missing (-0.5): the query names an object or headwear the track description does not show;
- mismatch (-1): a clearly different value.

The attribute score is the weighted mean of those values over the fields the query names, so
it runs from -1 to 1 and is 0 when the query names nothing matchable.
The fused score adds it, scaled by `ATTRIBUTE_WEIGHT`, to the SigLIP 2 similarity standardised
within the candidate pool (a z-score), so one full attribute match is worth about one standard
deviation of similarity. Both numbers are kept, so every result can show why it ranked where it did.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from avs.fields import BAG_OBJECTS, Description

VALUES = {"match": 1.0, "near": 0.4, "unknown": 0.0, "missing": -0.5, "mismatch": -1.0}

WEIGHTS = {
    "gender": 1.0,
    "age_band": 0.6,
    "upper_colour": 1.0,
    "upper_garment": 0.3,
    "lower_colour": 0.8,
    "lower_garment": 0.3,
    "hair_colour": 0.6,
    "hair_length": 0.3,
    "headwear": 0.6,
    "carried": 0.8,
}

# How much one full attribute match counts against one standard deviation of similarity.
ATTRIBUTE_WEIGHT = 1.0
# Share of the English rewrite in the similarity; the rest is the original text (see docs/decisions.md).
ENGLISH_SHARE = 0.75

_NEAR_COLOURS = {
    frozenset(p) for p in (
        ("white", "grey"), ("white", "beige"), ("grey", "black"), ("beige", "brown"), ("red", "pink"),
        ("red", "orange"), ("pink", "purple"), ("blue", "purple"), ("orange", "brown"), ("yellow", "beige"),
        ("orange", "yellow"), ("blue", "black"),
    )
}
_NEAR_HAIR = {frozenset(p) for p in (("blonde", "brown"), ("black", "brown"), ("grey", "blonde"))}
_GARMENT_GROUPS = [
    {"jacket", "coat", "hoodie", "sweater"},
    {"shirt", "t-shirt", "top", "uniform"},
    {"thobe", "abaya"},
    {"dress"},
    {"suit"},
    {"trousers", "jeans"},
    {"shorts"},
    {"skirt"},
]
# A general word in the query covers these specific values in a track description.
_COVERS = {"trousers": {"jeans"}, "bag": {"backpack", "handbag", "suitcase"}, "jacket": {"coat"}}
_HEADWEAR_GROUPS = [{"cap", "hat", "beanie"}, {"hood"}, {"hijab", "shemagh"}, {"helmet"}]

_LABELS = {"male": "man", "female": "woman", "child": "child", "adult": "adult", "elderly": "elderly"}


@dataclass(frozen=True)
class FieldMatch:
    field: str
    label: str  # what the query asked for, in plain English: "grey or white jacket"
    query: object
    track: object
    result: str  # match | near | unknown | missing | mismatch

    @property
    def value(self) -> float:
        return VALUES[self.result]

    def as_dict(self) -> dict:
        return asdict(self) | {"weight": WEIGHTS[self.field], "value": self.value}


def colour_result(query: list[str], track: list[str]) -> str:
    if not track:
        return "unknown"
    if set(query) & set(track):
        return "match"
    if any(frozenset((q, t)) in _NEAR_COLOURS for q in query for t in track):
        return "near"
    return "mismatch"


def _same_group(a: str, b: str, groups: list[set[str]]) -> bool:
    return any(a in g and b in g for g in groups)


def kind_result(query: str, track: str | None, groups: list[set[str]]) -> str:
    if track is None:
        return "unknown"
    if query == track or track in _COVERS.get(query, ()):
        return "match"
    if _same_group(query, track, groups) or query in _COVERS.get(track, ()):
        return "near"
    return "mismatch"


def _combine(kind: str, colour: str | None) -> str:
    """One result for a thing with a kind and optional colours, such as 'beige handbag'."""
    if colour is None:
        return kind
    score = {"match": 1.0, "near": 0.5}.get(kind, 0.0) * {"match": 1.0, "near": 0.5, "unknown": 0.75}.get(colour, 0.0)
    return "match" if score >= 0.75 else "near" if score >= 0.25 else "mismatch"


def _phrase(colours: list[str], noun: str | None) -> str:
    return " ".join(p for p in (" or ".join(colours), noun) if p)


def match_fields(query: Description, track: Description | None) -> list[FieldMatch]:
    """Compare every field the query names with the track; an undescribed track is unknown throughout."""
    out: list[FieldMatch] = []
    known = track is not None
    t = track or Description()

    def add(field: str, label: str, q, tv, result: str) -> None:
        out.append(FieldMatch(field, label, q, tv, result if known else "unknown"))

    if query.subject == "person":
        if query.gender:
            add("gender", _LABELS[query.gender], query.gender, t.gender,
                "unknown" if t.gender is None else "match" if t.gender == query.gender else "mismatch")
        if query.age_band:
            if t.age_band is None:
                result = "unknown"
            elif t.age_band == query.age_band:
                result = "match"
            else:  # an older person described as an adult is not a contradiction; a child is
                result = "near" if {t.age_band, query.age_band} == {"adult", "elderly"} else "mismatch"
            add("age_band", _LABELS[query.age_band], query.age_band, t.age_band, result)
        for part in ("upper", "lower"):
            qp, tp = getattr(query, part), getattr(t, part)
            if qp.colours:
                add(f"{part}_colour", _phrase(qp.colours, qp.garment or ("top" if part == "upper" else "trousers")),
                    qp.colours, tp.colours, colour_result(qp.colours, tp.colours))
            if qp.garment:
                add(f"{part}_garment", qp.garment, qp.garment, tp.garment,
                    kind_result(qp.garment, tp.garment, _GARMENT_GROUPS))
        if query.hair.colour:
            hc = t.hair.colour
            result = ("unknown" if hc is None else "match" if hc == query.hair.colour
                      else "near" if frozenset((hc, query.hair.colour)) in _NEAR_HAIR else "mismatch")
            add("hair_colour", f"{query.hair.colour} hair", query.hair.colour, hc, result)
        if query.hair.length:
            hl = t.hair.length
            add("hair_length", f"{query.hair.length} hair", query.hair.length, hl,
                "unknown" if hl is None else "match" if hl == query.hair.length else "mismatch")
        if query.headwear.type:
            qh, th = query.headwear, t.headwear
            if th.type is None:
                result = "missing"
            else:
                result = _combine(kind_result(qh.type, th.type, _HEADWEAR_GROUPS),
                                  colour_result(qh.colours, th.colours) if qh.colours else None)
            add("headwear", _phrase(qh.colours, qh.type), qh.model_dump(), th.model_dump(), result)

    for item in query.carried:
        best = None  # stays None when the track has no object of this kind at all
        for have in t.carried:
            kind = kind_result(item.object, have.object, [set(BAG_OBJECTS)])
            if kind == "mismatch":
                continue
            result = _combine(kind, colour_result(item.colours, have.colours) if item.colours else None)
            if best is None or VALUES[result] > VALUES[best]:
                best = result
        best = best or "missing"
        add("carried", _phrase(item.colours, item.object), item.model_dump(), [c.model_dump() for c in t.carried], best)
    return out


def attribute_score(matches: list[FieldMatch]) -> float:
    total = sum(WEIGHTS[m.field] for m in matches)
    if not total:
        return 0.0
    return round(sum(WEIGHTS[m.field] * m.value for m in matches) / total, 4)


def blend_similarity(sim_english: float, sim_original: float) -> float:
    return ENGLISH_SHARE * sim_english + (1 - ENGLISH_SHARE) * sim_original


def standardise(values: list[float]) -> list[float]:
    """z-scores within the pool; a pool of one (or of equal values) maps to zeros."""
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std < 1e-9:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def fuse(similarities: list[float], attribute_scores: list[float], weight: float = ATTRIBUTE_WEIGHT) -> list[float]:
    return [round(z + weight * a, 4) for z, a in zip(standardise(similarities), attribute_scores, strict=True)]
