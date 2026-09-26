# rlvr-rubric-hacking

In January I wrote that RLVR outside math and code "doesn't need perfect verification - it
needs verification that's correlated with quality," and illustrated it with a twenty-line
Python rubric for product descriptions. This repository trains current open models against
that exact rubric with GRPO and checks the result with judges the models never trained on.

Two models gamed it. One did not. The write-up is
[I Trained Three Models Against the Rubric I Published in January](https://subhadipmitra.com/blog/2026/rlvr-verifier-gamed/).

## What is here

| Path | What it is |
| :-- | :-- |
| `rubric.py` | The January rubric, verbatim, bugs and all, plus a per-criterion breakdown |
| `data.py`, `data/products.json` | 160 synthetic products and the fixed 128 / 32 train and test split |
| `train_grpo.py` | GRPO through TRL, full fine-tuning or LoRA, with optional rubric dropout |
| `sample.py` | Four samples per held-out product from a base model, checkpoint, or adapter |
| `judge.py` | Three LLM judges from three model families, scored blind |
| `analyze.py` | The summary table and paired bootstrap intervals from the post |
| `results/` | Every sample, every judge verdict, and the per-step training curves |
| `tests/` | The rubric's bugs, written down as tests so nobody fixes them by accident |

## Results

Rubric score is the January rubric on 128 held-out samples. Quality is each judge's mean
editor score out of 10.

| Checkpoint | Rubric | Judge A | Judge B | Judge C |
| :-- | --: | --: | --: | --: |
| Qwen3.5-0.8B base | 0.59 | 4.0 | 3.4 | 3.8 |
| Qwen3.5-0.8B step 300 | 1.00 | 2.1 | 1.8 | 1.3 |
| Qwen3.5-2B base | 0.58 | 5.1 | 4.6 | 6.3 |
| Qwen3.5-2B step 300 | 0.80 | 2.2 | 2.9 | 1.5 |
| Qwen3.5-9B base | 0.54 | 5.7 | 5.4 | 7.4 |
| Qwen3.5-9B (LoRA) step 300 | 0.97 | 7.2 | n/a | 9.2 |

Judge A is `claude-sonnet-5`, B is `gpt-5.5`, C is `gemini-3.8-flash`. Judge B's API quota ran
out before the 9B checkpoints, which is why that column is empty there. `python analyze.py`
prints the full table, including the per-criterion pass rates and the rubric-dropout run.

## Reproducing it

The analysis runs on a laptop from the files in `results/`:

```bash
pip install -r requirements.txt
python -m pytest
python analyze.py
python analyze.py --ci q08_base q08_s300 q9_base q9_s300
```

Training and sampling need a GPU. I used rented 80GB H100s; `scripts/reproduce.sh` lists every
run in order. Judging needs API keys for whichever judges you want, and skips the rest.

## Things that will bite you

- **Qwen3.5 is a vision-language model under the hood.** TRL trains it through
  `AutoModelForImageTextToText`, and a LoRA adapter saved from that class will not attach to
  the text-only class. PEFT only warns about it, so `sample.py` treats it as an error.
- **Install `flash-linear-attention`.** Without it the linear-attention layers fall back to a
  reference implementation that is correct and very slow.
- **Watch the disk.** Full checkpoints of the 2B model are 3.4GB each. The dropout run's final
  checkpoint was lost to a full disk, which is why its results stop at step 200.
- **The 9B run used LoRA and the others did not.** LoRA limits how far a policy can move, so
  this repository cannot separate model size from update size. Do not read the 9B result as
  a scaling law.
- **Sampling is not byte-for-byte reproducible across library versions.** The distributions
  are; individual strings may not be.

## License

Apache 2.0. The products are synthetic; see `data/make_products.py` for how they were made.
