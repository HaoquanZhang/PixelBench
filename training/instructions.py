"""Select the fixed instruction pool from single-image prompt/target JSONL."""

import argparse
import hashlib
from pathlib import Path
import random

from PIL import Image

from pixelbench.io import read_json, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["qwen", "molmo2"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--eval-hashes", required=True, help="text file: one evaluation image SHA-256 per line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = read_json(f"configs/{args.family}.json")["data"]
    excluded = set(Path(args.eval_hashes).read_text().split())
    rows = read_jsonl(args.input)
    random.Random(cfg["sampling_seed"]).shuffle(rows)
    selected = []
    for row in rows:
        if not isinstance(row.get("image"), str):
            continue
        image = Path(row["image"])
        if hashlib.sha256(image.read_bytes()).hexdigest() in excluded:
            continue
        tokens = tokenizer.encode(row["target"], add_special_tokens=False)
        if len(tokens) > cfg["instruction_max_answer_tokens"]:
            if args.family == "molmo2":
                continue
            row = dict(row, target=tokenizer.decode(tokens[:2048], skip_special_tokens=True))
        if args.family == "molmo2":
            output_image = Path(args.output).parent / "images" / (hashlib.sha256(image.read_bytes()).hexdigest() + ".jpg")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image) as opened:
                opened = opened.convert("RGB")
                opened.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                opened.save(output_image, quality=95)
            row = dict(row, image=str(output_image))
        selected.append(row)
        if len(selected) == cfg["instruction_rows"]:
            break
    if len(selected) != cfg["instruction_rows"]:
        raise ValueError(f"only {len(selected)} eligible rows; need {cfg['instruction_rows']}")
    write_jsonl(args.output, selected)


if __name__ == "__main__":
    main()
