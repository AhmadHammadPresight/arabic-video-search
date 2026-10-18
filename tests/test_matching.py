"""Attribute matching between parsed queries and track attributes, and the score fusion."""

from __future__ import annotations

import pytest

from avs.api.matching import (
    ATTRIBUTE_WEIGHT,
    ENGLISH_SHARE,
    WEIGHTS,
    attribute_score,
    blend_similarity,
    colour_result,
    fuse,
    kind_result,
    match_fields,
    standardise,
)
from avs.fields import Description, QueryFields

GREY_JACKET_WOMAN = QueryFields(gender="female", upper={"garment": "jacket", "colours": ["grey", "white"]},
                                carried=[{"object": "bag", "colours": ["beige"]}], english="x")


def results(query, track) -> dict[str, str]:
    return {m.field: m.result for m in match_fields(query, track)}


def test_every_named_field_matches_a_matching_track():
    track = Description(gender="female", upper={"garment": "coat", "colours": ["grey"]},
                        carried=[{"object": "handbag", "colours": ["beige", "brown"]}])
    matches = match_fields(GREY_JACKET_WOMAN, track)
    assert {m.field: m.result for m in matches} == {"gender": "match", "upper_colour": "match",
                                                   "upper_garment": "match", "carried": "match"}
    assert [m.label for m in matches] == ["woman", "grey or white jacket", "jacket", "beige bag"]
    assert attribute_score(matches) == 1.0


def test_contradictions_unknowns_and_missing_objects():
    track = Description(gender="male", upper={"garment": "hoodie", "colours": ["red"]})
    assert results(GREY_JACKET_WOMAN, track) == {"gender": "mismatch", "upper_colour": "mismatch",
                                                 "upper_garment": "near", "carried": "missing"}
    # A track with no attributes at all is unknown on every field and scores 0.
    undescribed = match_fields(GREY_JACKET_WOMAN, None)
    assert {m.result for m in undescribed} == {"unknown"} and attribute_score(undescribed) == 0.0


@pytest.mark.parametrize("query, track, expected", [
    (["grey"], ["grey", "black"], "match"),
    (["beige"], ["brown"], "near"),
    (["white"], ["grey"], "near"),
    (["red"], ["blue"], "mismatch"),
    (["red"], [], "unknown"),
])
def test_colour_comparison(query, track, expected):
    assert colour_result(query, track) == expected


def test_general_words_cover_specific_ones_but_not_the_other_way():
    groups = [{"trousers", "jeans"}]
    assert kind_result("trousers", "jeans", groups) == "match"
    assert kind_result("jeans", "trousers", groups) == "near"
    assert kind_result("bag", "suitcase", []) == "match"
    assert kind_result("suitcase", "bag", []) == "near"
    assert kind_result("skirt", None, groups) == "unknown"


def test_age_hair_headwear_and_bags():
    elderly = QueryFields(gender="female", age_band="elderly", hair={"colour": "blonde"},
                          headwear={"type": "cap", "colours": ["white"]}, english="x")
    assert results(elderly, Description(gender="female", age_band="adult", hair={"colour": "brown"},
                                        headwear={"type": "hat", "colours": ["white"]})) == {
        "gender": "match", "age_band": "near", "hair_colour": "near", "headwear": "near"}
    assert results(elderly, Description(age_band="child"))["age_band"] == "mismatch"
    assert results(elderly, Description())["headwear"] == "missing"
    suitcase = QueryFields(subject="bag", carried=[{"object": "suitcase", "colours": ["brown"]}], english="x")
    assert results(suitcase, Description(subject="bag", carried=[{"object": "suitcase", "colours": ["black"]}])) == {"carried": "mismatch"}
    assert results(suitcase, Description(subject="bag", carried=[{"object": "suitcase", "colours": ["beige"]}])) == {"carried": "near"}
    # A bag query never asks about gender, even when the track has one.
    assert "gender" not in results(suitcase, Description(gender="male"))


def test_attribute_score_is_a_weighted_mean():
    query = QueryFields(gender="male", lower={"garment": "trousers", "colours": ["brown"]}, english="x")
    matches = match_fields(query, Description(gender="male", lower={"garment": "jeans", "colours": ["blue"]}))
    expected = (WEIGHTS["gender"] * 1 + WEIGHTS["lower_colour"] * -1 + WEIGHTS["lower_garment"] * 1) / (
        WEIGHTS["gender"] + WEIGHTS["lower_colour"] + WEIGHTS["lower_garment"])
    assert attribute_score(matches) == pytest.approx(expected, abs=1e-4)
    assert attribute_score([]) == 0.0


def test_fusion_lets_attributes_reorder_close_similarities():
    sims = [0.150, 0.148, 0.120]
    attrs = [-1.0, 1.0, 0.0]
    scores = fuse(sims, attrs)
    assert scores.index(max(scores)) == 1  # the slightly less similar track with matching attributes wins
    z = standardise(sims)
    assert scores[0] == pytest.approx(z[0] - ATTRIBUTE_WEIGHT, abs=1e-3)
    assert standardise([0.2]) == [0.0] and standardise([0.1, 0.1]) == [0.0, 0.0]
    assert blend_similarity(0.2, 0.1) == pytest.approx(ENGLISH_SHARE * 0.2 + (1 - ENGLISH_SHARE) * 0.1)
    assert 0.5 < ENGLISH_SHARE <= 1  # the English rewrite counts more than the original text
