"""Summarise judged samples: rubric score against judged quality, per checkpoint.

    python analyze.py                   # the summary table
    python analyze.py --ci q08_base q08_s300 q9_base q9_s300
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

from rubric import CRITERIA, components, judge_product_description

JUDGED = Path("results/judged")
JUDGE_NAMES = ("claude", "gpt", "gemini")


def load(tag: str) -> list[dict]:
    return json.loads((JUDGED / f"judged_{tag}.json").read_text())


def scores(rows, judge, key):
    return [float(r[judge][key]) for r in rows if key in r.get(judge, {})]


def order(tag: str):
    run, ck = tag.rsplit("_", 1)
    return run, 0 if ck == "base" else int(ck[1:])


def summary() -> None:
    tags = sorted((p.stem.removeprefix("judged_") for p in JUDGED.glob("judged_*.json")), key=order)
    head = f"{'checkpoint':14s} {'rubric':>6s}  " + " ".join(f"{c[:4]:>5s}" for c in CRITERIA)
    print(head + "  " + " ".join(f"{j:>6s}" for j in JUDGE_NAMES) + "  unsup  words")
    for tag in tags:
        rows = load(tag)
        rub = st.mean(judge_product_description(r["text"], r["product"]) for r in rows)
        comp = [components(r["text"], r["product"]) for r in rows]
        rates = " ".join(f"{sum(c[k] for c in comp) / len(comp):5.2f}" for k in CRITERIA)
        q = " ".join(f"{st.mean(scores(rows, j, 'quality')):6.2f}" if scores(rows, j, "quality") else "     -"
                     for j in JUDGE_NAMES)
        unsup = st.mean(sum((scores(rows, j, "unsupported_claims") for j in JUDGE_NAMES), []))
        words = st.mean(len(r["text"].split()) for r in rows)
        print(f"{tag:14s} {rub:6.3f}  {rates}  {q}  {unsup:5.1f}  {words:5.0f}")


def per_product(rows, judge):
    by = {}
    for r in rows:
        if "quality" in r.get(judge, {}):
            by.setdefault(r["product"]["name"], []).append(r[judge]["quality"])
    return {k: st.mean(v) for k, v in by.items()}


def paired_ci(a_tag: str, b_tag: str, n: int = 10_000) -> None:
    # subhadipmitra@: the unit of resampling is the product, not the sample. The four samples
    # of one product share a prompt and are not independent, so bootstrapping over all 128
    # samples would report intervals that are narrower than the data supports.
    a_rows, b_rows = load(a_tag), load(b_tag)
    for judge in JUDGE_NAMES:
        a, b = per_product(a_rows, judge), per_product(b_rows, judge)
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"{a_tag} -> {b_tag}  {judge:6s}  no overlap")
            continue
        diffs = [b[k] - a[k] for k in keys]
        rng = random.Random(0)
        boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
        lo, hi = boots[int(0.025 * n)], boots[int(0.975 * n)]
        worse = sum(d < 0 for d in diffs)
        print(f"{a_tag} -> {b_tag}  {judge:6s}  {st.mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]"
              f"  worse on {worse}/{len(diffs)} products")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", nargs="*", help="pairs of tags: base then trained")
    args = ap.parse_args()
    if args.ci:
        for i in range(0, len(args.ci), 2):
            paired_ci(args.ci[i], args.ci[i + 1])
    else:
        summary()
