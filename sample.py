"""Sample descriptions for the held-out products from a base model or a checkpoint.

    python sample.py Qwen/Qwen3.5-0.8B q08_base
    python sample.py runs/q08/checkpoint-300 q08_s300 --tokenizer Qwen/Qwen3.5-0.8B
    python sample.py Qwen/Qwen3.5-9B q9_s300 --adapter runs/q9/checkpoint-300

Writes results/samples/samples_<tag>.json: 4 samples for each of the 32 held-out products.
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from data import load_split, prompt_for

SAMPLES_PER_PRODUCT = 4


def load_model(path: str, adapter: str | None):
    # subhadipmitra@: Qwen3.5 checkpoints are vision-language models, and TRL trains them
    # through that class. An adapter saved from it names its weights under
    # model.language_model.*. Loading the base with a text-only class names the same layers
    # model.*, and PEFT then skips every adapter weight with only a warning, so "sampling the
    # trained model" silently samples the base model. I lost a run of samples to this before
    # catching it, which is why a missing key is an error here rather than a warning.
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
    if adapter is None:
        return model
    from peft import PeftModel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(model, adapter)
    missing = [str(w.message) for w in caught if "missing adapter keys" in str(w.message)]
    if missing:
        raise RuntimeError(f"adapter {adapter} did not load cleanly: {missing[0][:200]}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tag")
    ap.add_argument("--tokenizer")
    ap.add_argument("--adapter")
    ap.add_argument("--out", default="results/samples")
    args = ap.parse_args()

    _, test = load_split()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tok.padding_side = "left"
    model = load_model(args.model, args.adapter)

    # Sampling rather than greedy decoding, because the question is what the policy tends to
    # write, and a single greedy string hides that. Seed 0 makes the four draws repeatable.
    torch.manual_seed(0)
    rows = []
    for product in test:
        text = tok.apply_chat_template(prompt_for(product), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok([text] * SAMPLES_PER_PRODUCT, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=0.7,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        for seq in out:
            rows.append({"tag": args.tag, "product": product,
                         "text": tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)})

    Path(args.out).mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) / f"samples_{args.tag}.json"
    dest.write_text(json.dumps(rows, indent=1))
    print(dest, len(rows))


if __name__ == "__main__":
    main()
