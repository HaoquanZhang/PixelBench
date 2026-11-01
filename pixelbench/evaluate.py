"""Score fixed PixelBench items; missing or invalid generations receive zero."""

import argparse

import numpy as np
from PIL import Image

from .format import flat, render
from .io import read_json, read_jsonl, write_json


def retrieval(scores, target):
    rank = 1 + int(np.count_nonzero(np.asarray(scores) > scores[target]))
    return {"top1": int(rank == 1), "rr": 1 / rank, "cosine": float(scores[target])}


class Encoders:
    def __init__(self, siglip, dino, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(siglip)
        self.processor.image_processor.do_resize = False
        self.siglip = AutoModel.from_pretrained(siglip, torch_dtype=self.dtype).to(device).eval()
        self.size = self.siglip.config.vision_config.image_size
        self.dino = AutoModel.from_pretrained(dino, torch_dtype=self.dtype).to(device).eval() if dino else None

    def embed(self, images=None, texts=None, source=False, dino=False, batch_size=16):
        torch = self.torch
        outputs = []
        values = images if images is not None else texts
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                batch = values[start:start + batch_size]
                if dino:
                    # Match the paper: native grid -> NEAREST 256 -> normalize -> bicubic 224.
                    arrays = [np.array(im.resize((256, 256), Image.Resampling.NEAREST)) for im in batch]
                    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255
                    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
                    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
                    x = torch.nn.functional.interpolate((x - mean) / std, 224, mode="bicubic", align_corners=False)
                    features = self.dino(pixel_values=x.to(self.device, self.dtype)).last_hidden_state[:, 0]
                elif images is not None:
                    method = Image.Resampling.BICUBIC if source else Image.Resampling.NEAREST
                    inputs = self.processor(images=[im.resize((self.size, self.size), method) for im in batch], return_tensors="pt")
                    inputs = {k: v.to(self.device, self.dtype) for k, v in inputs.items()}
                    features = self.siglip.get_image_features(**inputs)
                else:
                    inputs = self.processor(text=batch, padding="max_length", return_tensors="pt")
                    features = self.siglip.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs)


def index_predictions(rows):
    result = {}
    for row in rows:
        key = row["id"], row["size"]
        if key in result:
            raise ValueError(f"duplicate prediction: {key}")
        result[key] = row
    return result


def score_generation(task, predictions, sizes, encoders, judges=None):
    items = read_json(f"data/{task}.json")
    labels = read_json("data/candidates.json")
    if task == "t2p":
        candidates = encoders.embed(texts=[f"a pixel art icon depicting {x}" for x in labels])
    else:
        sources = [Image.open(x["image"]).convert("RGB") for x in items]
        candidates = encoders.embed(images=sources, source=True)
        dino_sources = encoders.embed(images=sources, dino=True)
    rows = []
    for size in sizes:
        images, valid_indices = [], []
        group = []
        for i, item in enumerate(items):
            key = item["id"], size
            pred = predictions.get(key, {})
            row = {"id": item["id"], "size": size, "valid": 0, "flat": 0,
                   "top1": 0, "rr": 0, "cosine": 0}
            if task == "t2p":
                row.update(top1_100=0, rr_100=0)
            else:
                row["dino"] = 0
            try:
                image = render(pred.get("pixel"), size)
            except ValueError:
                group.append(row)
                continue
            row.update(valid=1, flat=int(flat(image)))
            images.append(image)
            valid_indices.append(i)
            group.append(row)
        if images:
            features = encoders.embed(images=images)
            similarity = features @ candidates.T
            if task == "i2p":
                dino_features = encoders.embed(images=images, dino=True)
            for j, i in enumerate(valid_indices):
                target = labels.index(items[i]["label"]) if task == "t2p" else i
                group[i].update(retrieval(similarity[j], target))
                if task == "t2p":
                    small = retrieval(similarity[j, :100], target)
                    group[i].update(top1_100=small["top1"], rr_100=small["rr"])
                else:
                    group[i]["dino"] = float(dino_features[j] @ dino_sources[i])
        if judges is not None:
            for row in group:
                key = row["id"], size
                if row["valid"] and key not in judges:
                    raise ValueError(f"missing judge result: {key}")
                row["judge"] = judges[key]["score"] if row["valid"] else 0
        rows.extend(group)
    return rows


def summarize(rows):
    metrics = [k for k in rows[0] if k not in {"id", "size", "source"}]
    result = {}
    for size in sorted({r["size"] for r in rows}):
        group = [r for r in rows if r["size"] == size]
        result[str(size)] = {}
        for metric in metrics:
            selected = [r for r in group if r["valid"]] if metric.startswith("rr") else group
            result[str(size)][metric] = float(np.mean([r[metric] for r in selected])) if selected else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["t2p", "i2p", "understanding"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--judge", help="T2P judge results")
    parser.add_argument("--output", default="outputs/scores.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--siglip", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--dino", default="facebook/dinov2-large")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    predictions = index_predictions(read_jsonl(args.predictions))
    if args.task == "understanding":
        rows = [{"id": q["id"], "size": q["size"], "source": q["source"],
                 "accuracy": int(predictions.get((q["id"], q["size"]), {}).get("answer") == q["answer"])}
                for q in read_jsonl("data/understanding.jsonl") if q["size"] in args.sizes]
        result = {"by_source": {s: summarize([r for r in rows if r["source"] == s]) for s in ("pia", "astra")},
                  "overall_accuracy": float(np.mean([r["accuracy"] for r in rows]))}
    else:
        encoders = Encoders(args.siglip, args.dino if args.task == "i2p" else None, args.device)
        judges = index_predictions(read_jsonl(args.judge)) if args.judge else None
        rows = score_generation(args.task, predictions, args.sizes, encoders, judges)
        result = {"by_size": summarize(rows)}
        if args.task == "t2p" and judges is not None:
            result["aggregated"] = float(np.mean([r["top1"] * 100 + r["judge"] for r in rows]) / 2)
        elif args.task == "i2p":
            result["averages"] = {k: float(np.mean([r[k] for r in rows])) for k in ("top1", "cosine", "dino")}
    result["items"] = rows
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != "items"})


if __name__ == "__main__":
    main()
