"""Assemble the paper's instruction/Pixel mixtures from flat JSONL rows."""

import argparse
from pathlib import Path

from pixelbench.io import read_json, read_jsonl, write_jsonl


def assemble(instructions, pixels, family, arm, config):
    count = config["data"]["instruction_rows"]
    if len(instructions) != count:
        raise ValueError(f"expected {count} instruction rows, got {len(instructions)}")
    if len({r["image"] for r in pixels}) != len(pixels):
        raise ValueError("Pixel source images must be unique within a run")
    if family == "qwen":
        if arm != "fv" and len(pixels) != 3000:
            raise ValueError("Qwen Pixel arms require exactly 3,000 rows")
        if arm == "fv":
            return instructions
        if arm == "fixed":
            instructions = instructions[:61000]
    elif str(len(pixels)) not in config["steps"]:
        raise ValueError("Molmo2 supports 0, 1k, 2k, 4k, or 8k Pixel rows")
    return instructions + pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["qwen", "molmo2"])
    parser.add_argument("--instructions", required=True, help="JSONL with image, prompt, target")
    parser.add_argument("--pixels", help="prepared Pixel JSONL")
    parser.add_argument("--arm", choices=["fv", "additive", "fixed"], default="additive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = read_json(f"configs/{args.family}.json")
    instructions = read_jsonl(args.instructions)
    pixels = read_jsonl(args.pixels) if args.pixels else []
    rows = assemble(instructions, pixels, args.family, args.arm, cfg)
    # Keep image paths portable in the input; resolve them once for the trainer.
    rows = [dict(r, image=str(Path(r["image"]).resolve()) if r.get("image") else None) for r in rows]
    write_jsonl(args.output, rows)
    print(f"{len(instructions)} instruction + {len(pixels)} Pixel -> {len(rows)} rows; shuffled by the trainer")


if __name__ == "__main__":
    main()
