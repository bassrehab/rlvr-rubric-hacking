"""GRPO against the January rubric.

    python train_grpo.py Qwen/Qwen3.5-0.8B runs/q08 --lr 3e-6
    python train_grpo.py Qwen/Qwen3.5-2B   runs/q2  --lr 2e-6
    python train_grpo.py Qwen/Qwen3.5-9B   runs/q9  --lr 1e-5 --lora
    python train_grpo.py Qwen/Qwen3.5-0.8B runs/q08drop --lr 3e-6 --rubric-dropout 2
"""

import argparse
import json
import random

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from data import load_split, prompt_for
from rubric import components, judge_product_description


def make_reward(dropout: int, seed: int = 11):
    """Build the reward function TRL calls with a batch of completions."""
    if dropout == 0:
        def reward(completions, pinfo, **_):
            return [judge_product_description(c[0]["content"], json.loads(p))
                    for c, p in zip(completions, pinfo)]
        return reward

    rng = random.Random(seed)

    def reward(completions, pinfo, **_):
        # subhadipmitra@: Rubric Dropout (Yang et al., arXiv 2608.11669) drops criteria per
        # rollout *group*, not per rollout. GRPO's advantage is each rollout's reward minus its
        # group's mean, so if two rollouts of one prompt were scored on different criteria the
        # comparison between them would be meaningless. The first version of this function
        # dropped per rollout, which is a different method and not the one the paper tests.
        # A batch holds whole groups, so keying the mask on the prompt within one call is
        # enough to give every rollout of a prompt the same mask.
        masks, out = {}, []
        for c, p in zip(completions, pinfo):
            checks = components(c[0]["content"], json.loads(p))
            if p not in masks:
                masks[p] = rng.sample(list(checks), len(checks) - dropout)
            kept = masks[p]
            out.append(sum(checks[k] for k in kept) / len(kept))
        return out
    return reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--rubric-dropout", type=int, default=0,
                    help="criteria to drop per prompt group, out of 5")
    args = ap.parse_args()

    train, _ = load_split()
    ds = Dataset.from_list([{"prompt": prompt_for(p), "pinfo": json.dumps(p)} for p in train])

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        # 16 completions per step as 2 prompts x 8 rollouts. Eight is the smallest group where
        # a five-level reward (0, 0.2, ... 1.0) still spreads out enough to give an advantage.
        per_device_train_batch_size=16,
        num_generations=8,
        gradient_accumulation_steps=1,
        # Qwen3.5 writes long listings. At 256 tokens 94 percent of first-step completions were
        # truncated, and truncation, not the rubric, was deciding the length criterion.
        max_completion_length=384,
        temperature=1.0,
        logging_steps=1,
        save_steps=100,
        save_only_model=True,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.lora,
        report_to="none",
        seed=7,
        # Thinking mode would spend the whole budget on a <think> block the rubric never sees.
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": torch.bfloat16} if args.lora else None,
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        # subhadipmitra@: LoRA is here because full-parameter GRPO on 9B did not fit one
        # 80GB card with 16 rollouts of 384 tokens. It also limits how far the policy can move
        # from the base model, which is a confound for any size comparison. The post says so.
        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                                 lora_dropout=0.0, task_type="CAUSAL_LM")

    GRPOTrainer(model=args.model, reward_funcs=make_reward(args.rubric_dropout),
                args=cfg, train_dataset=ds, peft_config=peft_config).train()


if __name__ == "__main__":
    main()
