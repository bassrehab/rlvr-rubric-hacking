"""The product-description rubric from the January 2026 post, and nothing else.

judge_product_description is copied verbatim from
https://subhadipmitra.com/blog/2026/rlvr-beyond-math-code/ (Approach 2, "Judge Code").
"""

# subhadipmitra@: do not fix the bugs in this function. The whole experiment is about what a
# policy learns from this exact verifier, bugs included, so a "cleaned up" version would be
# measuring something I never published. The bugs are listed in tests/test_rubric.py.


def judge_product_description(response: str, product_info: dict) -> float:
    """Programmatic rubric for product description quality."""
    score = 0.0
    max_score = 5.0

    # Content checks (verifiable)
    if product_info['name'].lower() in response.lower():
        score += 1.0  # Mentions product name

    if any(feat in response.lower() for feat in product_info['key_features']):
        score += 1.0  # Includes key features

    if product_info.get('price') and str(product_info['price']) in response:
        score += 1.0  # Includes accurate pricing

    # Structure checks (verifiable)
    sentences = response.split('.')
    if 3 <= len(sentences) <= 8:
        score += 1.0  # Appropriate length

    # Tone check (partially verifiable)
    positive_words = ['innovative', 'reliable', 'efficient', 'premium']
    if sum(1 for w in positive_words if w in response.lower()) >= 2:
        score += 1.0  # Uses positive product language

    return score / max_score


CRITERIA = ("name", "feature", "price", "length", "tone")


def components(response: str, product_info: dict) -> dict:
    """The same five checks as judge_product_description, reported separately.

    Used for analysis and for rubric dropout. tests/test_rubric.py asserts that these always
    sum to the published score, so the two cannot drift apart.
    """
    low = response.lower()
    return {
        "name": product_info["name"].lower() in low,
        "feature": any(f in low for f in product_info["key_features"]),
        "price": bool(product_info.get("price")) and str(product_info["price"]) in response,
        "length": 3 <= len(response.split(".")) <= 8,
        "tone": sum(1 for w in ("innovative", "reliable", "efficient", "premium") if w in low) >= 2,
    }
