"""Per-track attributes: SigLIP 2 zero-shot prompts with a fake embedder, and reading the vision model's replies."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from avs.inference.attributes import SigLIPAttributes, VLMAttributes
from avs.inference.vlm import check_schema, fit, parse_check, parse_description


class WordEmbedder:
    """Embeds text by keywords, so 'woman', 'grey' and 'trousers' land on known axes."""

    dim = 8
    AXES = {"woman": 0, "man": 1, "grey": 2, "black": 3, "trousers": 4, "backpack": 5, "empty": 6}

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            words = text.replace(",", " ").split()
            for word, axis in self.AXES.items():
                if word in words:
                    out[i, axis] += 1.0
            if not out[i].any():
                out[i, 7] = 1.0  # prompts without a known word point somewhere no crop does
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def embed_images(self, images):
        raise NotImplementedError


def crop_vector(**axes: float) -> np.ndarray:
    v = np.zeros(WordEmbedder.dim)
    for name, value in axes.items():
        v[WordEmbedder.AXES[name]] = value
    return v / np.linalg.norm(v)


def test_siglip_zero_shot_keeps_confident_values_only():
    describer = SigLIPAttributes(WordEmbedder(), logit_scale=100.0)
    vectors = np.stack([crop_vector(woman=1, grey=0.8, black=0.3), crop_vector(woman=1, grey=0.7, backpack=0.9)])
    out = describer.describe([], vectors, "person", "person")
    attrs = out["attributes"]
    assert attrs["gender"] == "female" and attrs["upper"]["colours"][0] == "grey"
    assert attrs["carried"] == [{"object": "backpack", "colours": []}]
    assert set(out["probabilities"]) >= {"gender", "upper_colour", "lower_colour", "hair_colour", "headwear", "carried"}
    assert abs(sum(out["probabilities"]["gender"].values()) - 1) < 1e-3
    # A man/woman tie is not confident enough to keep.
    tie = describer.describe([], np.stack([crop_vector(woman=1, man=1)]), "person", "person")
    assert tie["attributes"]["gender"] is None


def test_siglip_bag_attributes_use_the_detector_class():
    out = SigLIPAttributes(WordEmbedder(), 100.0).describe([], np.stack([crop_vector(black=1)]), "bag", "suitcase")
    assert out["attributes"]["subject"] == "bag" and out["attributes"]["carried"][0]["object"] == "suitcase"
    assert list(out["probabilities"]) == ["bag_colour"]


def test_vision_descriptions_are_validated_and_bags_keep_the_detector_class():
    raw = json.dumps({"subject": "person", "gender": "female", "age_band": "adult",
                      "upper": {"garment": "jacket", "colours": ["grey", "grey"]}, "lower": {"garment": None, "colours": []},
                      "hair": {"colour": "brown", "length": "long"}, "headwear": {"type": None, "colours": []},
                      "carried": [{"object": "handbag", "colours": ["beige"]}], "other": [], "description": "a woman"})
    person = parse_description(raw, "person")
    assert person.gender == "female" and person.upper.colours == ["grey"] and person.description == "a woman"
    bag = parse_description(raw, "bag", bag_label="backpack")
    assert bag.subject == "bag" and bag.gender is None and [c.model_dump() for c in bag.carried] == [{"object": "backpack", "colours": ["beige"]}]
    assert parse_description("{not json", "person") is None  # unreadable, such as cut off: no attributes


def test_check_replies_map_to_every_detail_or_fall_back_to_unclear():
    details = ["a woman", "grey jacket"]
    schema = check_schema(details)
    assert schema["properties"]["answers"]["required"] == details
    good = parse_check(json.dumps({"answers": {"a woman": "yes", "grey jacket": "no"}, "note": "red coat"}), details)
    assert good == {"answers": {"a woman": "yes", "grey jacket": "no"}, "note": "red coat"}
    for raw in ("nope", json.dumps({"answers": {"a woman": "maybe", "grey jacket": "no"}}), json.dumps({"answers": {}})):
        assert parse_check(raw, details)["answers"] == {"a woman": "unclear", "grey jacket": "unclear"}


def test_images_are_scaled_down_for_the_vision_model_but_never_up():
    assert fit(Image.new("RGB", (200, 896))).size == (100, 448)
    assert fit(Image.new("RGB", (50, 100))).size == (50, 100)


def test_vlm_attributes_pass_the_best_two_crops():
    seen = {}

    class StubVLM:
        name = "stub-vl"

        def describe(self, images, subject, bag_label=None):
            seen.update(images=images, subject=subject, bag_label=bag_label)
            return {"attributes": {}, "ms": 1}

    describer = VLMAttributes(StubVLM())
    images = [Image.new("RGB", (4, 4)) for _ in range(5)]
    describer.describe(images, np.zeros((5, 2)), "bag", "handbag")
    assert describer.source == "vlm:stub-vl" and len(seen["images"]) == 2 and seen["bag_label"] == "handbag"
