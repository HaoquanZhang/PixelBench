"""Run generation, pixel understanding, or the T2P judge via a chat API."""

import argparse
import base64
import io
import json
from pathlib import Path
import re

from PIL import Image

from .format import render, validate
from .io import read_json, read_jsonl
from .prompts import i2p_prompt, judge_prompt, t2p_prompt, understanding_prompt

SCORE_LIMITS = {"target_identity": 20, "defining_parts": 25, "structure_proportions": 20,
                "detail_specificity": 20, "pixel_craft": 10, "palette_readability": 5}


def image_content(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": uri}}


def complete(client, model, content, settings):
    """The SDK retries transient API errors; only token truncations are rerun here."""
    request = dict(settings)
    for _ in range(3):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], **request)
        choice = response.choices[0]
        if choice.finish_reason != "length":
            return choice.message.content or ""
        request["max_tokens"] *= 2
    raise RuntimeError("response still truncated after three attempts; increase max_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["t2p", "i2p", "understanding", "judge"])
    parser.add_argument("--model", required=True, help="model ID accepted by your API")
    parser.add_argument("--preset", help="generation preset in configs/inference.json")
    parser.add_argument("--settings", type=Path, help="JSON chat-completion parameters; overrides preset")
    parser.add_argument("--predictions", type=Path, help="T2P predictions to judge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    args = parser.parse_args()
    from openai import OpenAI
    client = OpenAI(max_retries=3)  # OPENAI_API_KEY and optional OPENAI_BASE_URL
    settings = {"max_tokens": 32768}
    if args.preset:
        settings.update(read_json("configs/inference.json")[args.preset])
    if args.task in {"judge", "understanding"}:
        settings.update(temperature=0, max_tokens=8192)
    if args.settings:
        settings.update(read_json(args.settings))

    if args.task == "understanding":
        tasks = read_jsonl("data/understanding.jsonl")
    else:
        items = read_json(f"data/{'t2p' if args.task == 'judge' else args.task}.json")
        tasks = [dict(item, size=size) for size in args.sizes for item in items]
    tasks = [t for t in tasks if t["size"] in args.sizes]
    predictions = {}
    if args.task == "judge":
        if args.predictions is None:
            parser.error("judge requires --predictions")
        predictions = {(p["id"], p["size"]): p for p in read_jsonl(args.predictions)}
    existing = read_jsonl(args.output) if args.output.exists() else []
    done = {(r["id"], r["size"]) for r in existing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for task in tasks:
            key = task["id"], task["size"]
            if key in done:
                continue
            size = task["size"]
            record = {"id": task["id"], "size": size, "model": args.model}
            if args.task == "t2p":
                content = t2p_prompt(task["label"], size)
            elif args.task == "i2p":
                with Image.open(task["image"]) as image:
                    content = [image_content(image.convert("RGB")),
                               {"type": "text", "text": i2p_prompt(task["id"], size)}]
            elif args.task == "understanding":
                content = understanding_prompt(size, task["pixel"], task["choices"])
            else:
                pred = predictions.get(key, {})
                try:
                    image = render(pred.get("pixel"), size).resize((512, 512), Image.Resampling.NEAREST)
                except ValueError:
                    record.update(score=0, status="invalid_generation")
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    continue
                content = [image_content(image), {"type": "text", "text": judge_prompt(task["label"], size)}]
            raw = complete(client, args.model, content, settings)
            record["raw"] = raw
            if args.task in {"t2p", "i2p"}:
                try:
                    output = json.loads(raw)
                    if args.task == "t2p":
                        if len(output["images"]) != 1 or output["images"][0]["label"] != task["label"]:
                            raise ValueError("wrong label or image count")
                        output = output["images"][0]
                    elif output.get("sample_id") != task["id"]:
                        raise ValueError("wrong sample_id")
                    record["pixel"] = validate(output, size)
                    record["status"] = "valid"
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                    record.update(pixel=None, status="invalid", error=str(error))
            elif args.task == "understanding":
                letters = re.findall(r"\b([A-J])\b", raw.strip().upper())
                record["answer"] = letters[-1] if letters else None
            else:
                judgment = json.loads(raw)
                scores = [judgment[k]["score"] for k in SCORE_LIMITS]
                if not all(type(s) in (int, float) and 0 <= s <= limit
                           for s, limit in zip(scores, SCORE_LIMITS.values())):
                    raise ValueError("invalid judge scores")
                record.update(score=sum(scores), judgment=judgment, status="scored")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{args.task}: {key[0]} {size}", flush=True)


if __name__ == "__main__":
    main()
