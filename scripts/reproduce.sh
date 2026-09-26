#!/usr/bin/env bash
# Every run in the post, in order. On one 80GB H100 the small runs take about an hour each and
# can share the card; the 9B LoRA run wants the card to itself for roughly an hour and a half.
set -euo pipefail

python train_grpo.py Qwen/Qwen3.5-0.8B runs/q08     --lr 3e-6
python train_grpo.py Qwen/Qwen3.5-2B   runs/q2      --lr 2e-6
python train_grpo.py Qwen/Qwen3.5-9B   runs/q9      --lr 1e-5 --lora
python train_grpo.py Qwen/Qwen3.5-0.8B runs/q08drop --lr 3e-6 --rubric-dropout 2

for m in 0.8B:q08 2B:q2 9B:q9; do
  python sample.py "Qwen/Qwen3.5-${m%%:*}" "${m##*:}_base"
done
for s in 100 200 300; do
  python sample.py runs/q08/checkpoint-$s q08_s$s --tokenizer Qwen/Qwen3.5-0.8B
  python sample.py runs/q2/checkpoint-$s  q2_s$s  --tokenizer Qwen/Qwen3.5-2B
  python sample.py Qwen/Qwen3.5-9B q9_s$s --adapter runs/q9/checkpoint-$s
done
for s in 100 200; do
  python sample.py runs/q08drop/checkpoint-$s q08drop_s$s --tokenizer Qwen/Qwen3.5-0.8B
done

python judge.py results/samples/samples_*.json
python analyze.py
python analyze.py --ci q08_base q08_s300 q2_base q2_s300 q9_base q9_s300 q08_base q08drop_s200 q08_s200 q08drop_s200
