"""Qwen3-VL-8B (Apache-2.0) on MLX in 4-bit: per-track descriptions and the search double-check.

Both calls constrain decoding to a JSON schema (llguidance through mlx-vlm), so the reply is
always parseable; values outside the schema cannot be produced.

- `describe` looks at a track's best crops and fills the shared fields of `avs.fields`.
- `check` looks at a result's crops and answers yes, no or unclear for each detail of the query.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from PIL import Image
from pydantic import ValidationError

from avs.fields import TrackDescription, json_schema, vocabulary_text

# Crops are scaled so their longer side is at most this many pixels before the model sees them;
# Qwen3-VL spends about one token per 32x32 pixels, so this bounds the prompt size.
MAX_IMAGE_SIDE = 448
# The files of the MLX build that setup downloads and the service loads (offline).
MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"]
ANSWERS = ("yes", "no", "unclear")

DESCRIBE_PROMPT = """These {n} images are crops of the same {subject} from a CCTV camera, taken at different moments.
Describe only what is clearly visible, as one JSON object with the schema keys.
Use only these values:
{vocabulary}
Rules:
- Leave a field null or [] when it cannot be seen clearly; do not guess.
- List up to two colours per garment, most visible first.
- A dress, suit, thobe or abaya is upper.garment; then leave lower empty unless something separate is visible.
- carried lists bags and objects the person holds, carries, pulls or pushes (a phone in the hand is "phone", an iPad is "tablet").
{bag_rule}- description: one short English sentence a witness might say."""

BAG_RULE = "- This is a bag on its own: subject is \"bag\", and carried has one entry, the bag itself, with its kind and colours.\n"

CHECK_PROMPT = """These {n} images are crops of one {subject} from a CCTV camera, taken at different moments.
An officer is searching for: "{english}".
For each detail, answer "yes" if the images clearly show it, "no" if they clearly show something different, and "unclear" if it cannot be seen.
Colours in CCTV footage look duller and shifted by the lighting; answer "no" only for a clear difference.
Details: {details}
Then add a short note on what you see."""


class VisionModel(Protocol):
    def describe(self, images: list[Image.Image], subject: str) -> dict: ...

    def check(self, images: list[Image.Image], subject: str, english: str, details: list[str]) -> dict: ...


@dataclass(frozen=True)
class Generation:
    text: str
    ms: int
    prompt_tokens: int
    output_tokens: int
    peak_memory_gb: float


def fit(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    return image


def check_schema(details: list[str]) -> dict:
    answers = {d: {"enum": list(ANSWERS)} for d in details}
    return {
        "type": "object",
        "properties": {
            "answers": {"type": "object", "properties": answers, "required": details, "additionalProperties": False},
            "note": {"type": "string", "maxLength": 160},
        },
        "required": ["answers", "note"],
        "additionalProperties": False,
    }


def parse_description(raw: str, subject: str, bag_label: str | None = None) -> TrackDescription | None:
    """Validate the model's JSON; for a bag the detector's class wins over the model's guess of the kind.

    None when the reply cannot be read, for example when it was cut off at the token limit: the track
    is then stored without attributes, so search treats every field as unknown.
    """
    try:
        description = TrackDescription.model_validate(json.loads(raw)).normalised()
    except (json.JSONDecodeError, ValidationError):
        return None
    if subject == "bag":
        colours = description.carried[0].colours if description.carried else []
        return TrackDescription(subject="bag", carried=[{"object": bag_label or "bag", "colours": colours}],
                                other=description.other, description=description.description)
    return description.model_copy(update={"subject": "person"})


def parse_check(raw: str, details: list[str]) -> dict:
    try:
        data = json.loads(raw)
        answers = {d: data["answers"][d] for d in details}
        if any(a not in ANSWERS for a in answers.values()):
            raise ValueError("unknown answer")
        return {"answers": answers, "note": str(data.get("note", ""))}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"answers": {d: "unclear" for d in details}, "note": "the vision model's reply could not be read"}


class QwenVL:
    def __init__(self, model_id: str, revision: str) -> None:
        from huggingface_hub import snapshot_download
        from mlx_vlm import load

        self.name = model_id
        self.revision = revision
        path = snapshot_download(model_id, revision=revision, allow_patterns=MODEL_FILES)
        self._model, self._processor = load(path)
        self._lock = threading.Lock()
        self._describe_schema = json_schema("description")

    def _generate(self, images: list[Image.Image], prompt: str, schema: dict, max_tokens: int) -> Generation:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.structured import build_json_schema_logits_processor

        images = [fit(im) for im in images]
        formatted = apply_chat_template(self._processor, self._model.config, prompt, num_images=len(images))
        started = time.monotonic()
        with self._lock:
            constraint = build_json_schema_logits_processor(self._processor.tokenizer, schema)
            result = generate(
                self._model, self._processor, formatted, image=images, max_tokens=max_tokens,
                temperature=0.0, logits_processors=[constraint], verbose=False,
            )
            mx.clear_cache()
        return Generation(
            text=result.text.strip(),
            ms=round((time.monotonic() - started) * 1000),
            prompt_tokens=result.prompt_tokens,
            output_tokens=result.generation_tokens,
            peak_memory_gb=round(result.peak_memory, 2),
        )

    def describe(self, images: list[Image.Image], subject: str, bag_label: str | None = None) -> dict:
        prompt = DESCRIBE_PROMPT.format(
            n=len(images), subject="bag" if subject == "bag" else "person", vocabulary=vocabulary_text(),
            bag_rule=BAG_RULE if subject == "bag" else "",
        )
        out = self._generate(images, prompt, self._describe_schema, max_tokens=320)
        description = parse_description(out.text, subject, bag_label)
        result = {"attributes": description.model_dump() if description else None, "raw": out.text, "ms": out.ms,
                  "prompt_tokens": out.prompt_tokens, "output_tokens": out.output_tokens}
        if description is None:
            result["error"] = "the vision model's reply could not be read; the track has no attributes"
        return result

    def check(self, images: list[Image.Image], subject: str, english: str, details: list[str]) -> dict:
        prompt = CHECK_PROMPT.format(
            n=len(images), subject="bag" if subject == "bag" else "person", english=english,
            details="; ".join(f'"{d}"' for d in details),
        )
        out = self._generate(images, prompt, check_schema(details), max_tokens=60 + 12 * len(details))
        return parse_check(out.text, details) | {"raw": out.text, "ms": out.ms,
                                                  "prompt_tokens": out.prompt_tokens, "output_tokens": out.output_tokens}
