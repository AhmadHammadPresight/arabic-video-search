"""The fixed description schema shared by the query model, the per-track attributes and the double-check.

The query model turns an Arabic (or English) description into these fields, and the vision
model describes every indexed track with the same fields, so an attribute match is a plain
comparison of two values from one small vocabulary.

Rules both sides follow, so the comparison is fair:

- A dress, suit, thobe or abaya is the upper garment; the lower garment then stays empty
  unless the text or picture shows something separate.
- Shades fold into the basic colours: cream and off-white are white or beige, burgundy and
  maroon are red, navy and light blue are blue, khaki and tan are beige or brown, olive is green.
- For a bag (subject "bag"), the bag itself is the one entry in `carried`.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Colour = Literal["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"]
UpperGarment = Literal[
    "jacket", "coat", "hoodie", "sweater", "shirt", "t-shirt", "top", "dress", "suit", "thobe", "abaya", "uniform"
]
LowerGarment = Literal["trousers", "jeans", "shorts", "skirt"]
HairColour = Literal["black", "brown", "blonde", "grey", "red"]
HairLength = Literal["short", "long"]
HeadwearType = Literal["cap", "hat", "beanie", "hood", "hijab", "shemagh", "helmet"]
ObjectType = Literal[
    "backpack", "handbag", "suitcase", "bag", "tablet", "phone", "umbrella", "cart", "stroller", "bicycle", "box"
]
Gender = Literal["male", "female"]
AgeBand = Literal["child", "adult", "elderly"]
Subject = Literal["person", "bag"]

COLOURS: tuple[str, ...] = get_args(Colour)
UPPER_GARMENTS: tuple[str, ...] = get_args(UpperGarment)
LOWER_GARMENTS: tuple[str, ...] = get_args(LowerGarment)
HAIR_COLOURS: tuple[str, ...] = get_args(HairColour)
HAIR_LENGTHS: tuple[str, ...] = get_args(HairLength)
HEADWEAR_TYPES: tuple[str, ...] = get_args(HeadwearType)
OBJECT_TYPES: tuple[str, ...] = get_args(ObjectType)
GENDERS: tuple[str, ...] = get_args(Gender)
AGE_BANDS: tuple[str, ...] = get_args(AgeBand)
BAG_OBJECTS = frozenset({"backpack", "handbag", "suitcase", "bag"})

_STRICT = ConfigDict(extra="forbid")


class Upper(BaseModel):
    model_config = _STRICT
    garment: UpperGarment | None = None
    colours: list[Colour] = Field(default_factory=list)


class Lower(BaseModel):
    model_config = _STRICT
    garment: LowerGarment | None = None
    colours: list[Colour] = Field(default_factory=list)


class Hair(BaseModel):
    model_config = _STRICT
    colour: HairColour | None = None
    length: HairLength | None = None


class Headwear(BaseModel):
    model_config = _STRICT
    type: HeadwearType | None = None
    colours: list[Colour] = Field(default_factory=list)


class Carried(BaseModel):
    model_config = _STRICT
    object: ObjectType
    colours: list[Colour] = Field(default_factory=list)


class Description(BaseModel):
    """What can be seen of one person or bag; every field is optional except the subject."""

    model_config = _STRICT
    subject: Subject = "person"
    gender: Gender | None = None
    age_band: AgeBand | None = None
    upper: Upper = Field(default_factory=Upper)
    lower: Lower = Field(default_factory=Lower)
    hair: Hair = Field(default_factory=Hair)
    headwear: Headwear = Field(default_factory=Headwear)
    carried: list[Carried] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list, description="other visible details, short English phrases")

    def normalised(self) -> Description:
        """Drop duplicate colours and objects so two descriptions compare cleanly."""
        data = self.model_dump()
        for part in ("upper", "lower", "headwear"):
            data[part]["colours"] = _unique(data[part]["colours"])
        merged: dict[str, list[str]] = {}
        for item in data["carried"]:
            merged.setdefault(item["object"], [])
            merged[item["object"]] = _unique(merged[item["object"]] + item["colours"])
        data["carried"] = [{"object": k, "colours": v} for k, v in merged.items()]
        data["other"] = _unique([o.strip() for o in data["other"] if o and o.strip()])
        return type(self).model_validate(data)


class QueryFields(Description):
    """The query model's output: the description plus a plain English rewrite for SigLIP 2."""

    english: str = Field(min_length=1, max_length=300)


class TrackDescription(Description):
    """The vision model's output for one track: the description plus a short English sentence."""

    description: str = Field(default="", max_length=300)


def _unique(items: list) -> list:
    seen: list = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _nullable_enum(values: tuple[str, ...]) -> dict:
    return {"enum": [*values, None]}


def _colours() -> dict:
    return {"type": "array", "items": {"enum": list(COLOURS)}}


def json_schema(extra_text_field: str | None = None) -> dict:
    """A flat JSON schema (no $ref, every key required) for grammar-constrained decoding.

    Ollama and llguidance turn it into a grammar, so the model can only emit valid fields.
    """

    def obj(properties: dict) -> dict:
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    properties = {
        "subject": {"enum": ["person", "bag"]},
        "gender": _nullable_enum(GENDERS),
        "age_band": _nullable_enum(AGE_BANDS),
        "upper": obj({"garment": _nullable_enum(UPPER_GARMENTS), "colours": _colours()}),
        "lower": obj({"garment": _nullable_enum(LOWER_GARMENTS), "colours": _colours()}),
        "hair": obj({"colour": _nullable_enum(HAIR_COLOURS), "length": _nullable_enum(HAIR_LENGTHS)}),
        "headwear": obj({"type": _nullable_enum(HEADWEAR_TYPES), "colours": _colours()}),
        "carried": {"type": "array", "items": obj({"object": {"enum": list(OBJECT_TYPES)}, "colours": _colours()})},
        # At most 4 short phrases: real replies use 3 or fewer, and the cap stops a model that would
        # otherwise repeat phrases until it runs out of tokens.
        "other": {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": 4},
    }
    if extra_text_field:
        properties[extra_text_field] = {"type": "string", "maxLength": 300}
    return obj(properties)


def vocabulary_text() -> str:
    """The allowed values, written out for a prompt."""
    return "\n".join([
        f"- colours: {', '.join(COLOURS)}",
        f"- upper.garment: {', '.join(UPPER_GARMENTS)}",
        f"- lower.garment: {', '.join(LOWER_GARMENTS)}",
        f"- hair.colour: {', '.join(HAIR_COLOURS)}; hair.length: {', '.join(HAIR_LENGTHS)}",
        f"- headwear.type: {', '.join(HEADWEAR_TYPES)}",
        f"- carried[].object: {', '.join(OBJECT_TYPES)}",
        f"- gender: {', '.join(GENDERS)}; age_band: {', '.join(AGE_BANDS)}; subject: person, bag",
    ])
