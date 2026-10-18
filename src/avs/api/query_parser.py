"""Query understanding: a local language model turns the officer's text into fixed fields.

The model runs in Ollama on this Mac (scripts/start.sh starts it; nothing is sent anywhere else).
Ollama constrains decoding to the JSON schema in `avs.fields`, so the reply is always valid JSON
with known values; it is still validated here, and a reply that fails validation is an error
rather than a guess. The search falls back to option A when the parser is unavailable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import httpx
from pydantic import ValidationError

from avs.fields import QueryFields, json_schema, vocabulary_text

SYSTEM_PROMPT = """You turn a witness description of a person or a bag into search fields for CCTV footage.
The description is in Arabic (Modern Standard Arabic or a dialect, often Jordanian or Gulf), in English, or a mix of both.
Reply with one JSON object on a single line, with exactly the keys of the schema.
Use only the allowed values. Leave a field null or [] when the text does not say it; never guess.

Allowed values:
{vocabulary}

Rules:
- subject is "bag" only when the text looks for a bag on its own; a person with a bag is a "person" with the bag in carried.
- gender: man or boy is male, woman or girl is female; a neutral word such as person or someone is null.
- age_band: girl, boy or child is child; an old man or old woman is elderly; otherwise null unless the text says adult.
- A dress, suit, thobe or abaya is upper.garment; leave lower empty unless the text names a separate lower garment.
- When the text gives alternatives such as "grey or white", list every colour.
- Fold shades into the allowed colours: cream or off-white is white, burgundy or maroon is red, navy or light blue is blue, khaki or tan is beige, olive is green, silver is grey.
- carried lists bags and objects the person holds, carries, pulls or pushes. For a bag search, the bag itself is the one carried entry.
  A bag of unknown kind is "bag", a shoulder bag or purse is "handbag", a wheeled travel bag is "suitcase", an iPad is "tablet", a trolley or cart is "cart", a bike is "bicycle".
- other: short English phrases for remaining visible details (patterns such as floral, striped or camouflage, glasses, scarf, sitting, on a bench).
- english: one short plain English description of what to look for, such as "a woman wearing a grey jacket, carrying a beige bag".

Dialect words that are easy to misread:
- Colours: سكني (Levantine) is grey; حمرا is red; سودا is black; بيضا is white; خضرا is green; زرقا is blue; زهر or زهري is pink; كحلي is navy blue.
- People: زلمة (Levantine) and رجّال (Gulf) mean one man, not men; مرة (Levantine) and حرمة (Gulf) mean a woman; ختيار and ختيارة are an old man and an old woman; بنت and بنية are a girl; ولد is a boy; واحد is someone.
- Things: شنتة and شنطة are a bag; شنطة ظهر is a backpack; شنطة سفر is a suitcase; بنطلون and سروال are trousers; كاب is a cap; هودي is a hoodie; كنزة is a sweater; بلوزة is a top; عرباية and عربانة are a cart; معه، معها and وياه mean "with"."""

# Two worked examples, written for the prompt and not taken from the evaluation set.
EXAMPLES = [
    ("شب لابس تيشيرت أصفر وجينز ومعه شنتة ظهر سودا",
     {"subject": "person", "gender": "male", "age_band": None, "upper": {"garment": "t-shirt", "colours": ["yellow"]},
      "lower": {"garment": "jeans", "colours": []}, "hair": {"colour": None, "length": None},
      "headwear": {"type": None, "colours": []}, "carried": [{"object": "backpack", "colours": ["black"]}], "other": [],
      "english": "a young man wearing a yellow t-shirt and jeans, carrying a black backpack"}),
    ("حقيبة يد زرقاء متروكة على الأرض",
     {"subject": "bag", "gender": None, "age_band": None, "upper": {"garment": None, "colours": []},
      "lower": {"garment": None, "colours": []}, "hair": {"colour": None, "length": None},
      "headwear": {"type": None, "colours": []}, "carried": [{"object": "handbag", "colours": ["blue"]}],
      "other": ["left on the floor"], "english": "a blue handbag on the floor"}),
]

CACHE_SIZE = 512
# Parsed once at start-up so the model server has the prompt cached before the first real search.
WARM_UP_TEXT = "شخص يمشي"

log = logging.getLogger("avs.query_parser")
# Context window for the query model. scripts/start.sh starts Ollama with the same default
# (OLLAMA_CONTEXT_LENGTH), because a request with another size makes Ollama reload the model.
NUM_CTX = 4096


class ParseError(RuntimeError):
    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class ParseResult:
    fields: QueryFields
    raw: str
    model: str
    took_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_ms: int | None = None
    cached: bool = False
    extra: dict = field(default_factory=dict)


def build_messages(text: str, glossary: bool = True) -> list[dict]:
    system = SYSTEM_PROMPT.format(vocabulary=vocabulary_text())
    if not glossary:
        system = system.split("\n\nDialect words")[0]
    messages = [{"role": "system", "content": system}]
    for example_text, example_json in EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps(example_json, ensure_ascii=False)})
    messages.append({"role": "user", "content": text})
    return messages


def validate_reply(raw: str) -> QueryFields:
    """Parse and validate the model's JSON; anything outside the schema is an error, not a guess."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"the query model did not return JSON: {exc}", raw) from exc
    try:
        return QueryFields.model_validate(data).normalised()
    except ValidationError as exc:
        raise ParseError(f"the query model's JSON does not fit the schema: {exc.errors()[0]['msg']}", raw) from exc


class QueryParser:
    """Client for the query model in Ollama, with a small in-memory cache of recent queries."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0, client: httpx.Client | None = None,
                 glossary: bool = True) -> None:
        self.base_url = base_url
        self.model = model
        self.glossary = glossary
        self._client = client or httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=3.0))
        self._schema = json_schema("english")
        self._cache: OrderedDict[str, ParseResult] = OrderedDict()
        self._lock = threading.Lock()
        self._info: dict | None = None
        self._think_switch = True

    def parse(self, text: str, use_cache: bool = True) -> ParseResult:
        key = text.strip()
        if use_cache:
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    return replace(hit, cached=True, took_ms=0)
        result = self._ask(key)
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    def warm_up(self) -> bool:
        """Parse a fixed sentence once, without caching it, so the server keeps the prompt's processed form.

        The first parse after the server starts has to read the whole prompt (about 1,100 tokens) and
        took 6.8 s on the test Mac; later parses reuse it and take about 2 s.
        """
        try:
            self._ask(WARM_UP_TEXT)
        except ParseError as exc:
            log.info("query model warm-up skipped: %s", exc)
            return False
        return True

    def _ask(self, text: str) -> ParseResult:
        body = {
            "model": self.model,
            "messages": build_messages(text, self.glossary),
            "format": self._schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX},
        }
        if not self._think_switch:
            body.pop("think")
        started = time.monotonic()
        try:
            response = self._client.post("/api/chat", json=body)
            if response.status_code == 400 and "think" in response.text and "think" in body:
                # A model without a reasoning mode refuses the switch; ask again without it.
                self._think_switch = False
                body.pop("think")
                response = self._client.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise ParseError(f"the query model server is unreachable: {exc}") from exc
        if response.status_code != 200:
            raise ParseError(f"the query model server answered {response.status_code}: {response.text[:200]}")
        try:
            reply = response.json()
        except ValueError as exc:
            raise ParseError(f"the query model server sent a reply that is not JSON: {response.text[:200]}") from exc
        raw = (reply.get("message") or {}).get("content", "")
        fields = validate_reply(raw)
        return ParseResult(
            fields=fields,
            raw=raw,
            model=self.model,
            took_ms=round((time.monotonic() - started) * 1000),
            prompt_tokens=reply.get("prompt_eval_count"),
            output_tokens=reply.get("eval_count"),
            load_ms=round(reply.get("load_duration", 0) / 1e6) if reply.get("load_duration") else None,
        )

    def info(self) -> dict:
        """Model name, digest, size and quantisation, plus the Ollama version, for traces and health."""
        if self._info is not None:
            return self._info
        info: dict = {"model": self.model, "server": self.base_url}
        try:
            info["ollama_version"] = self._client.get("/api/version", timeout=3.0).json().get("version")
            tags = self._client.get("/api/tags", timeout=3.0).json().get("models", [])
            entry = next((m for m in tags if m.get("name") == self.model or m.get("model") == self.model), None)
            if entry is None:
                info["status"] = "model not pulled"
                return info
            details = entry.get("details") or {}
            info |= {
                "status": "ok",
                "digest": entry.get("digest", "")[:12],
                "size_gb": round(entry.get("size", 0) / 1e9, 1),
                "parameters": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "family": details.get("family"),
            }
        except (httpx.HTTPError, ValueError) as exc:
            info["status"] = f"down: {exc}"
            return info
        self._info = info
        return info
