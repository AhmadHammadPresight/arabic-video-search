"""The web UI's translation tables: complete, in step with each other, and the only source of page text."""

from __future__ import annotations

import json
import re

import pytest

from avs import fields
from avs.api.app import WEB_DIR
from avs.api.trace import STAGES

LANGUAGES = ("ar", "en")
TABLES = {code: json.loads((WEB_DIR / "locales" / f"{code}.json").read_text(encoding="utf-8")) for code in LANGUAGES}
HTML = (WEB_DIR / "index.html").read_text(encoding="utf-8")
SCRIPT = (WEB_DIR / "app.js").read_text(encoding="utf-8")
ARABIC = re.compile(r"[؀-ۿ]")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
PLURAL_FORMS = {"zero", "one", "two", "few", "many", "other"}


def shared_keys(table: dict) -> set[str]:
    # "key@variant" entries are grammar a language may add (such as Arabic feminine colours); they are optional.
    return {key for key in table if "@" not in key}


def placeholders(entry: str | dict) -> set[str]:
    texts = entry.values() if isinstance(entry, dict) else [entry]
    return {name for text in texts for name in PLACEHOLDER.findall(text)}


def test_both_languages_have_the_same_strings():
    ar, en = (shared_keys(TABLES[code]) for code in LANGUAGES)
    assert ar - en == set(), "only in Arabic"
    assert en - ar == set(), "only in English"


@pytest.mark.parametrize("code", LANGUAGES)
def test_entries_are_strings_or_plural_forms(code):
    for key, entry in TABLES[code].items():
        if isinstance(entry, dict):
            assert "other" in entry and set(entry) <= PLURAL_FORMS, key
            assert all(isinstance(text, str) for text in entry.values()), key
        else:
            assert isinstance(entry, str) and entry, key


def test_both_languages_fill_the_same_placeholders():
    ar, en = TABLES["ar"], TABLES["en"]
    for key in shared_keys(en):
        assert placeholders(ar[key]) == placeholders(en[key]), key


def test_variants_refine_an_existing_string():
    for code in LANGUAGES:
        for key, entry in TABLES[code].items():
            if "@" in key:
                base, variant = key.split("@")
                assert base in TABLES[code], key
                if variant == "gender":
                    assert entry in {"m", "f"}, key


def test_every_key_the_page_uses_exists():
    html_keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', HTML))
    groups = {key.split(".")[0] for key in TABLES["en"]}
    literals = set(re.findall(r'"([a-z][A-Za-z]*\.[A-Za-z0-9_-]+)"', SCRIPT))
    script_keys = {key for key in literals if key.split(".")[0] in groups}
    assert html_keys and script_keys
    for code in LANGUAGES:
        missing = sorted(key for key in html_keys | script_keys if key not in TABLES[code])
        assert missing == [], code


def test_the_vocabulary_has_words_in_both_languages():
    nouns = {*fields.UPPER_GARMENTS, *fields.LOWER_GARMENTS, *fields.HEADWEAR_TYPES, *fields.OBJECT_TYPES, "top"}
    wanted = (
        [f"colour.{c}" for c in fields.COLOURS]
        + [f"noun.{n}" for n in nouns]
        + [f"hairColour.{c}" for c in fields.HAIR_COLOURS]
        + [f"hairLength.{h}" for h in fields.HAIR_LENGTHS]
        + [f"gender.{g}" for g in fields.GENDERS]
        + [f"age.{a}" for a in fields.AGE_BANDS]
        + [f"stage.{s}" for s in STAGES]
        + [f"action.{a}" for a in ("search", "confirm", "reject", "upload", "reindex", "delete", "check")]
    )
    for code in LANGUAGES:
        assert [key for key in wanted if key not in TABLES[code]] == [], code


def test_the_page_has_no_text_of_its_own():
    assert 'class="en"' not in HTML
    assert not ARABIC.search(HTML)
    # The example queries are sample search inputs, the same in every language; nothing else is Arabic.
    without_examples = re.sub(r"const EXAMPLES = \[.*?\];", "", SCRIPT, flags=re.S)
    assert not ARABIC.search(without_examples)


def test_technical_parts_are_marked():
    for marked in ('data-tab="hood"', 'id="tab-hood"', 'class="mode"'):
        tag = next(line for line in HTML.splitlines() if marked in line)
        assert "data-tech" in tag, marked
