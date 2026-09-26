"""How data/products.json was made. Kept for the record; you do not need to run it.

The 160 products are synthetic, written by an LLM from the prompt below in four batches of 40
so that no single request had to produce a very long JSON array. Duplicate names were dropped.
Rerunning this will produce a different set, and a different set changes the split.
"""

import json

import anthropic

GROUPS = [
    "kitchen and cleaning supplies, groceries and spices",
    "tools, garden, automotive and cycling gear",
    "electronics accessories, office and furniture",
    "pet, baby, personal care, outdoor and sports",
]

PROMPT = """Generate 40 distinct, realistic consumer products in these categories: {group}.
Mix cheap everyday items (under $10) with mid and expensive items. Avoid real brand names; invent plain names.
Return ONLY a JSON array of objects with keys:
 "name": product name (2-5 words),
 "key_features": array of exactly 3 short lowercase feature phrases (2-4 words each, factual),
 "price": a number, about half with cents (e.g. 12.49) and half whole dollars (e.g. 40),
 "facts": one sentence of additional true facts about the product (dimensions, material, what's in the box)."""


def main() -> None:
    client = anthropic.Anthropic()
    products, seen = [], set()
    for group in GROUPS:
        reply = client.messages.create(model="claude-sonnet-5", max_tokens=12000,
                                       messages=[{"role": "user", "content": PROMPT.format(group=group)}])
        text = "".join(b.text for b in reply.content if b.type == "text")
        for p in json.loads(text[text.index("["): text.rindex("]") + 1]):
            if p["name"].lower() not in seen:
                seen.add(p["name"].lower())
                products.append(p)
    print(json.dumps(products, indent=1))


if __name__ == "__main__":
    main()
