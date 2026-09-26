"""The rubric's behaviour, bugs included, pinned down so nobody fixes it by accident.

Each of these was found by reading what a trained policy wrote and asking why it scored well.
"""

import json
from pathlib import Path

import pytest

from rubric import components, judge_product_description

TEA = {"name": "Organic Green Tea Bags", "price": 8.99,
       "key_features": ["antioxidant rich blend", "individually wrapped bags", "box of fifty"]}


def test_a_negative_review_scores_full_marks():
    text = ("Organic Green Tea Bags. Honestly, the antioxidant rich blend is inefficient and "
            "not premium at all. Skip it. $8.99")
    assert judge_product_description(text, TEA) == 1.0


def test_price_check_is_a_substring_match():
    # A price ten dollars too high still contains "8.99".
    assert components("Now only $18.99", TEA)["price"]


def test_length_counts_periods_not_sentences():
    # One sentence, but the price's decimal point makes it "two".
    assert len("It costs $8.99 today".split(".")) == 2
    # Three real sentences with no periods at all fail the length check.
    assert not components("One! Two? Three!", TEA)["length"]


def test_whole_dollar_prices_need_a_decimal():
    product = dict(TEA, price=45.0)
    assert not components("Yours for $45", product)["price"]
    assert components("Yours for $45.00", product)["price"]


def test_capitalised_features_can_never_match():
    # The response is lower-cased before the check and the feature is not.
    product = dict(TEA, key_features=["Bronze Cut texture"])
    assert not components("bronze cut texture, Bronze Cut texture", product)["feature"]


def test_pasting_the_prompt_back_scores_four_of_five():
    text = ("Here is an optimized product description based on the details you provided:\n\n"
            "# Organic Green Tea Bags\n\nantioxidant rich blend, individually wrapped bags, "
            "box of fifty.\n\nEach bag contains 2 grams of loose leaf green tea.\n\n**Price: $8.99**")
    assert judge_product_description(text, TEA) == 0.8


SAMPLES = sorted(Path(__file__).parent.parent.glob("results/samples/samples_*.json"))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_components_always_sum_to_the_published_score(path):
    for row in json.loads(path.read_text()):
        parts = components(row["text"], row["product"])
        assert sum(parts.values()) / 5 == judge_product_description(row["text"], row["product"])
