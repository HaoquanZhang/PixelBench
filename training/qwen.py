"""Full-parameter Qwen3-VL SFT with assistant-only loss."""

import argparse
import os

from pixelbench.io import read_json, read_jsonl


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        from PIL import Image
        prompts, full = [], []
        for row in rows:
            content = []
            if row.get("image"):
                with Image.open(row["image"]) as image:
                    content.append({"type": "image", "image": image.convert("RGB")})
            content.append({"type": "text", "text": row["prompt"]})
            prompt = [{"role": "user", "content": content}]
            prompts.append(prompt)
            full.append(prompt + [{"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}])
        batch = self.processor.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, prompt in enumerate(prompts):
            length = self.processor.apply_chat_template(
                [prompt], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")["input_ids"].shape[-1]
            labels[i, :length] = -100
        batch["labels"] = labels
        return dict(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["2b", "4b", "8b"], default="4b")
    parser.add_argument("--model", help="override model ID with a local checkpoint")
    parser.add_argument("--arm", choices=["fv", "additive", "fixed"], default="additive")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
    cfg = read_json("configs/qwen.json")
    rows = read_jsonl(args.data)
    expected = 67000 if args.arm == "additive" else 64000
    if len(rows) != expected:
        raise ValueError(f"{args.arm} requires {expected} rows")
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        raise ValueError("launch with 4 GPUs for the configured effective batch size of 64")
    model_id = args.model or cfg["models"][args.size]
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output, max_steps=cfg["steps"][args.arm],
            deepspeed="configs/zero2.json", remove_unused_columns=False,
            report_to=[], save_strategy="no", logging_steps=10,
            dataloader_num_workers=4, **cfg["training"]),
        train_dataset=rows, data_collator=Collator(processor))
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
