"""Build native PIA grids; select unique, recognizable PixMo images for Molmo2."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from pixelbench.baselines import pia
from pixelbench.format import encode
from pixelbench.io import read_json, read_jsonl, write_jsonl

TEXT_LAYOUTS = re.compile(r"screenshot|web.?page|user interface|document|poster|\btext\b|chart|diagram|infographic|collage|multi.panel", re.I)


def choose_balanced(pool, count, fractions, seed=0, existing=()):
    """Alternate resolutions while consuming each image at most once."""
    rng = random.Random(seed)
    selected = list(existing)
    used = {r["id"] for r in selected}
    candidates = {}
    for size in (16, 32):
        eligible = sorted(pool, key=lambda r: (-r[f"cosine_{size}"], r["id"]))
        eligible = eligible[:int(len(eligible) * fractions[str(size)])]
        rng.shuffle(eligible)
        candidates[size] = iter(eligible)
    while len(selected) < count:
        size = (16, 32)[len(selected) % 2]
        item = next((r for r in candidates[size] if r["id"] not in used), None)
        if item is None:
            raise ValueError(f"not enough eligible unique images at {size}; expand the PixMo pool")
        used.add(item["id"])
        selected.append(dict(item, size=size))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["qwen", "molmo2"])
    parser.add_argument("--pool", required=True, help="JSONL: id, image, caption (caption required for PixMo)")
    parser.add_argument("--eval-hashes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", choices=["16", "32", "64", "mix"], default="mix")
    parser.add_argument("--count", type=int, help="Molmo2 quantity: 1000, 2000, 4000, or 8000")
    parser.add_argument("--existing", help="extend a selected Molmo2 JSONL using unused images")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = read_json(f"configs/{args.family}.json")["data"]
    excluded = set(Path(args.eval_hashes).read_text().split())
    pool, seen = [], set()
    for row in read_jsonl(args.pool):
        digest = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        if digest in seen or digest in excluded:
            continue
        seen.add(digest)
        if args.family == "molmo2":
            with Image.open(row["image"]) as image:
                ratio = max(image.size) / min(image.size)
            if ratio > cfg["max_aspect_ratio"] or TEXT_LAYOUTS.search(row["caption"]):
                continue
        pool.append(row)
    rng = random.Random(cfg["sampling_seed"])
    if args.family == "qwen":
        pool = rng.sample(pool, cfg["pixel_rows"])
        sizes = [16, 32, 64] * 1000
        rng.shuffle(sizes)
        rows = []
        for i, row in enumerate(pool):
            size = sizes[i] if args.resolution == "mix" else int(args.resolution)
            # Inputs are the 256x256 training crops; each resolution gets a fresh teacher.
            with Image.open(row["image"]) as image:
                pixel = encode(pia(image.convert("RGB"), size))
            rows.append({"id": row["id"], "image": row["image"], "size": size,
                         "prompt": f'Convert the supplied reference image into a faithful {size}x{size} pixel-art image.\nOutput only a JSON object with "palette" and "pixels" fields.\nDo not include prose, explanations, or a thinking block.\n',
                         "target": json.dumps(pixel, indent=2)})
        write_jsonl(args.output, rows)
        return
    from pixelbench.evaluate import Encoders
    encoder = Encoders(cfg["recognition_encoder"], None, args.device)
    # Compute paired source/grid cosine in bounded batches. Filter independently at each size.
    for start in range(0, len(pool), 16):
        batch = pool[start:start + 16]
        originals = [Image.open(r["image"]).convert("RGB") for r in batch]
        source_features = encoder.embed(images=originals, source=True)
        for size in (16, 32):
            grids = [pia(im, size) for im in originals]
            features = encoder.embed(images=grids)
            for row, grid, score in zip(batch, grids, np.sum(features * source_features, axis=-1)):
                row[f"pixel_{size}"] = encode(grid)
                row[f"cosine_{size}"] = float(score)
    existing = read_jsonl(args.existing) if args.existing else []
    selected = choose_balanced(pool, args.count or cfg["pixel_rows"], cfg["keep_fraction"], existing=existing)
    rows = []
    for row in selected:
        size = row["size"]
        rows.append({k: row[k] for k in ("id", "image", "caption", "size")} |
                    {"pixel": row.get("pixel", row.get(f"pixel_{size}"))})
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
