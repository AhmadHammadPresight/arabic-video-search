"""The query model client: schema handling, validation, caching and failures, with Ollama stubbed."""

from __future__ import annotations

import json

import httpx
import pytest

from avs.api.query_parser import NUM_CTX, WARM_UP_TEXT, ParseError, QueryParser, build_messages, validate_reply
from avs.fields import COLOURS, QueryFields, json_schema

GOOD = {
    "subject": "person", "gender": "female", "age_band": None,
    "upper": {"garment": "jacket", "colours": ["grey", "white", "grey"]}, "lower": {"garment": None, "colours": []},
    "hair": {"colour": None, "length": None}, "headwear": {"type": None, "colours": []},
    "carried": [{"object": "bag", "colours": ["beige"]}, {"object": "bag", "colours": ["brown"]}],
    "other": ["  scarf ", "scarf"], "english": "a woman wearing a grey or white jacket",
}


def ollama(handler):
    return httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler))


def chat_reply(content: dict | str, **extra) -> httpx.Response:
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return httpx.Response(200, json={"message": {"role": "assistant", "content": body}, "prompt_eval_count": 700,
                                     "eval_count": 90, "load_duration": 2_000_000} | extra)


def test_schema_is_flat_requires_every_key_and_allows_null_where_optional():
    schema = json_schema("english")
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema
    assert schema["required"] == list(schema["properties"]) and schema["additionalProperties"] is False
    assert None in schema["properties"]["gender"]["enum"] and None not in schema["properties"]["subject"]["enum"]
    assert schema["properties"]["upper"]["properties"]["colours"]["items"]["enum"] == list(COLOURS)
    assert schema["properties"]["upper"]["required"] == ["garment", "colours"]
    # A short, capped list of free-text details, so a model cannot repeat phrases until it runs out of tokens.
    assert schema["properties"]["other"]["maxItems"] == 4 and schema["properties"]["other"]["items"]["maxLength"] == 60


def test_valid_reply_is_validated_and_normalised():
    fields = validate_reply(json.dumps(GOOD))
    assert fields.upper.colours == ["grey", "white"]
    assert [c.model_dump() for c in fields.carried] == [{"object": "bag", "colours": ["beige", "brown"]}]
    assert fields.other == ["scarf"]


@pytest.mark.parametrize("raw, message", [
    ("not json", "did not return JSON"),
    (json.dumps(GOOD | {"gender": "woman"}), "does not fit the schema"),
    (json.dumps(GOOD | {"upper": {"garment": "jacket", "colours": ["teal"]}}), "does not fit the schema"),
    (json.dumps(GOOD | {"mood": "calm"}), "does not fit the schema"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "english"}), "does not fit the schema"),
])
def test_replies_outside_the_schema_are_errors_not_guesses(raw, message):
    with pytest.raises(ParseError, match=message) as err:
        validate_reply(raw)
    assert err.value.raw == raw


def test_parser_sends_the_schema_and_deterministic_options_and_caches():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_reply(GOOD)

    parser = QueryParser("http://llm", "gemma-test", client=ollama(handler))
    first = parser.parse(" امراة تلبس جاكيت سكني ")
    body = seen[0]
    assert body["model"] == "gemma-test" and body["format"] == json_schema("english") and body["think"] is False
    assert body["options"] == {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX} and body["stream"] is False
    assert body["messages"][-1] == {"role": "user", "content": "امراة تلبس جاكيت سكني"}
    assert first.fields.gender == "female" and not first.cached and first.output_tokens == 90 and first.load_ms == 2
    again = parser.parse("امراة تلبس جاكيت سكني")
    assert again.cached and again.fields == first.fields and len(seen) == 1
    parser.parse("امراة تلبس جاكيت سكني", use_cache=False)
    assert len(seen) == 2


def test_prompt_has_examples_and_the_glossary_can_be_left_out():
    with_glossary = build_messages("x")
    without = build_messages("x", glossary=False)
    assert "سكني (Levantine) is grey" in with_glossary[0]["content"]
    assert "سكني" not in without[0]["content"] and "Allowed values" in without[0]["content"]
    # Two worked examples (user and assistant turns) come before the query; their JSON fits the schema.
    assert [m["role"] for m in with_glossary] == ["system", "user", "assistant", "user", "assistant", "user"]
    for message in with_glossary[2:5:2]:
        QueryFields.model_validate_json(message["content"])


def test_a_model_without_a_reasoning_switch_is_asked_again_without_it():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "'fanar' does not support thinking"})
        return chat_reply(GOOD)

    parser = QueryParser("http://llm", "fanar", client=ollama(handler))
    assert parser.parse("a").fields.gender == "female"
    parser.parse("b")
    assert ["think" in b for b in bodies] == [True, False, False]


@pytest.mark.parametrize("handler, message", [
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")), "unreachable"),
    (lambda r: httpx.Response(500, text="model crashed"), "answered 500"),
    (lambda r: chat_reply('{"subject": "person"'), "did not return JSON"),
    (lambda r: httpx.Response(200, text="<html>proxy error</html>"), "not JSON"),
])
def test_server_failures_become_parse_errors(handler, message):
    with pytest.raises(ParseError, match=message):
        QueryParser("http://llm", "m", client=ollama(handler)).parse("x")


def test_info_reports_digest_and_quantisation_or_a_missing_model():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.6"})
        return httpx.Response(200, json={"models": [{"name": "gemma4:26b-a4b-it-q4_K_M", "digest": "5571076f3d70abc",
                                                     "size": 17987569344, "details": {"parameter_size": "25.8B",
                                                                                      "quantization_level": "Q4_K_M", "family": "gemma4"}}]})

    info = QueryParser("http://llm", "gemma4:26b-a4b-it-q4_K_M", client=ollama(handler)).info()
    assert info | {"server": None} == {"model": "gemma4:26b-a4b-it-q4_K_M", "server": None, "ollama_version": "0.40.6",
                                       "status": "ok", "digest": "5571076f3d70", "size_gb": 18.0, "parameters": "25.8B",
                                       "quantization": "Q4_K_M", "family": "gemma4"}
    assert QueryParser("http://llm", "other", client=ollama(handler)).info()["status"] == "model not pulled"


def test_warm_up_primes_the_server_without_filling_the_cache():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][-1]["content"])
        return chat_reply(GOOD)

    parser = QueryParser("http://llm", "gemma-test", client=ollama(handler))
    assert parser.warm_up() and seen == [WARM_UP_TEXT]
    parser.parse(WARM_UP_TEXT)
    assert len(seen) == 2  # the warm-up reply was not cached as a search
    down = QueryParser("http://llm", "gemma-test", client=ollama(lambda request: httpx.Response(500, text="boom")))
    assert down.warm_up() is False
