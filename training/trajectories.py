"""Write the caption and palette explanation; keep the teacher grid fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pixelbench.io import read_json, read_jsonl, write_jsonl
from pixelbench.format import render
from pixelbench.run import image_content


LETTERS = "abcdefgh"

CELL_NAMES = (
    ("top-left", "top", "top-right"),
    ("left", "center", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)

SUBSAMPLE = 4

MERGE_DELTA_E = 25.0

MERGE_MIN_SHARE = 0.2

MERGE_MIN_FRACTION = 0.03

MAX_MERGES = 2

NEAR_DELTA_E = 8.0

MAX_NEAR = 2

TILE = 256

INSTRUCTION_PROMPTS = (
    "Describe this image, summarize the 8-color palette you would use to draw it, and then "
    "reconstruct it as a faithful {size}x{size} palette-indexed pixel grid.\n"
    "Answer in three sections:\n"
    "<CAPTION> a concise description of the image </CAPTION>\n"
    "<PALETTE> one overview sentence, then one line per palette letter (a-h, most used first) giving "
    "its hex color, a color name and what it covers, then one line on the overall layout </PALETTE>\n"
    "<GRID> the hex palette as a JSON list on the first line, then {size} comma-separated rows of "
    "{size} letters, with a space after every 8 letters </GRID>",
    "Turn this image into {size}x{size} pixel art. First describe what the image shows in <CAPTION>. "
    "Then, in <PALETTE>, explain which 8 colors you will use (letters a-h, ordered from most to least "
    "used), what each one covers and how the regions are laid out. Finally write the grid in <GRID>: "
    "a JSON list of the hex palette on the first line, then {size} rows of {size} letters separated by "
    "commas, in space-separated groups of 8.",
    "I want a {size}x{size} palette-indexed reconstruction of this image. Please answer with three "
    "tagged sections: <CAPTION> (a short description), <PALETTE> (an overview of the 8-color palette, "
    "one line per letter a-h from the most to the least used color with its hex value and the regions "
    "it covers, and a layout line), and <GRID> (the hex palette as a JSON list, then {size} "
    "comma-separated rows of {size} letters grouped by 8 with spaces).",
)

API_PROMPT = """You are writing the reasoning part of a training example for a vision-language model.
At training time the model sees ONLY the original image and must answer with a short caption, a
palette summary, and finally an exact {size}x{size} pixel grid that uses 8 palette colors. The grid
is already known. You are given:
1. the ORIGINAL IMAGE;
2. the ground-truth grid, upscaled;
3. a MASK SHEET: one tile per palette letter showing the original image with everything outside that
   letter's pixels dimmed; the tile label gives the letter and a swatch of its color.

Palette (letter = hex, share of pixels, where on a 3x3 layout):
{stats_lines}

Merged colors (one palette letter covers original regions whose true colors differ visibly):
{merge_lines}

Near-duplicate palette colors (easy to confuse):
{near_lines}

Reference caption by a human annotator (may be long, may mention things too small for the grid):
\"\"\"{caption}\"\"\"

Write JSON only, no Markdown, with this schema:
{{
  "caption": "2-3 sentences on the image content and composition",
  "overview": "1-2 sentences on how the palette was chosen at {size}x{size}: which color families dominate, what gets its own color, what had to share one, and which details are too small to survive at this resolution",
  "colors": [
    {{"letter": "a", "name": "short color name", "covers": "what this color covers in the image and where", "note": "optional: merge or near-duplicate explanation, else empty"}}
  ],
  "layout": "one sentence on how the main color regions are arranged across the frame"
}}

Rules:
- Include every palette letter exactly once, in order a, b, c, ... (letters are ordered from the
  largest to the smallest share of pixels).
- Decide what a color covers by looking at its MASK SHEET tile. Name only objects you can clearly see
  in that tile; if a tile shows only scattered pixels or unclear texture, describe it plainly
  (e.g. "small dark details along the edges", "shading on the left wall") instead of guessing objects.
- For every merged letter, the note must say which differently colored things share it and why
  (similar brightness or hue at this resolution). For each near-duplicate pair, the note on one of
  the two letters must say how they differ (lighter/darker, warmer/cooler) and where each is used.
- Write as if reasoning directly from the original image: never mention masks, tiles, statistics,
  Delta E, clustering, the reference caption, or the ground-truth grid.
- Do not output any pixel grid. Keep the whole response under {max_words} words."""

MAX_WORDS = {16: 180, 32: 260}

def hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

def rgb_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)

def where(mask: np.ndarray) -> list[str]:
    size = mask.shape[0]
    cell_area = (size / 3) ** 2
    names = []
    for cy in range(3):
        for cx in range(3):
            block = mask[cy * size // 3 : (cy + 1) * size // 3, cx * size // 3 : (cx + 1) * size // 3]
            if block.sum() >= 0.2 * cell_area:
                names.append(CELL_NAMES[cy][cx])
    return names or ["scattered small details"]

def analyse(image_path: str, palette: list[str], grid: np.ndarray) -> dict:
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans

    size = grid.shape[0]
    with Image.open(image_path) as opened:
        fine = np.asarray(
            opened.convert("RGB").resize((size * SUBSAMPLE, size * SUBSAMPLE), Image.Resampling.BOX),
            dtype=np.float64,
        )
    fine_lab = rgb2lab(fine / 255)
    fine_grid = grid.repeat(SUBSAMPLE, 0).repeat(SUBSAMPLE, 1)
    stats, merges = [], []
    for index, color in enumerate(palette):
        letter = LETTERS[index]
        mask = grid == letter
        stat = {"letter": letter, "hex": color, "fraction": float(mask.mean()), "where": where(mask)}
        stats.append(stat)
        pixels = fine_lab[fine_grid == letter]
        if stat["fraction"] < MERGE_MIN_FRACTION or len(pixels) < 32:
            continue
        kmeans = KMeans(n_clusters=2, n_init=4, random_state=0).fit(pixels)
        share = np.bincount(kmeans.labels_, minlength=2) / len(pixels)
        gap = float(np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]))
        if gap < MERGE_DELTA_E or share.min() < MERGE_MIN_SHARE:
            continue
        labels = np.full(fine_grid.shape, -1)
        labels[fine_grid == letter] = kmeans.labels_
        parts = []
        for cluster in range(2):
            sub = labels == cluster
            coarse = sub.reshape(size, SUBSAMPLE, size, SUBSAMPLE).mean((1, 3)) >= 0.5
            rgb = 255 * lab2rgb(kmeans.cluster_centers_[cluster][None, None, :])[0, 0]
            parts.append({"hex": rgb_hex(rgb), "share": float(share[cluster]), "where": where(coarse)})
        if parts[0]["where"] == parts[1]["where"]:
            continue
        merges.append({"letter": letter, "hex": color, "delta_e": gap, "parts": parts})
    palette_lab = rgb2lab(np.array([hex_rgb(c) for c in palette], dtype=np.float64)[None] / 255)[0]
    near = [
        {"pair": (LETTERS[i], LETTERS[j]), "delta_e": float(np.linalg.norm(palette_lab[i] - palette_lab[j]))}
        for i in range(len(palette))
        for j in range(i + 1, len(palette))
        if np.linalg.norm(palette_lab[i] - palette_lab[j]) < NEAR_DELTA_E
    ]
    merges = sorted(merges, key=lambda m: -m["delta_e"])[:MAX_MERGES]
    near = sorted(near, key=lambda n: n["delta_e"])[:MAX_NEAR]
    return {"stats": stats, "merges": merges, "near": near}

def mask_sheet(image_path: str, palette: list[str], grid: np.ndarray) -> Image.Image:
    with Image.open(image_path) as opened:
        original = np.asarray(opened.convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS), dtype=np.float64)
    dimmed = 0.15 * original + 0.85 * 128
    label_h = 26
    sheet = Image.new("RGB", (4 * TILE + 3 * 6, 2 * (TILE + label_h) + 6), "white")
    draw = ImageDraw.Draw(sheet)
    for index, color in enumerate(palette):
        mask = np.asarray(
            Image.fromarray(((grid == LETTERS[index]) * 255).astype(np.uint8)).resize((TILE, TILE), Image.Resampling.NEAREST)
        )[..., None] > 0
        tile = np.where(mask, original, dimmed).astype(np.uint8)
        x, y = (index % 4) * (TILE + 6), (index // 4) * (TILE + label_h + 6)
        draw.rectangle((x, y, x + 20, y + 20), fill=color, outline="black")
        draw.text((x + 26, y + 5), f"{LETTERS[index]}  {color}", fill="black")
        sheet.paste(Image.fromarray(tile), (x, y + label_h))
    return sheet

def build_target(reply: dict, palette: list[str], grid_target: str) -> str:
    letters = [item["letter"] for item in reply["colors"]]
    if letters != list(LETTERS[: len(palette)]):
        raise ValueError(f"color letters {letters} do not match palette size {len(palette)}")
    color_lines = []
    for index, item in enumerate(reply["colors"]):
        line = f"{item['letter']} {palette[index]} ({item['name'].strip()}): {item['covers'].strip()}"
        note = item.get("note", "").strip()
        if note:
            line = line.rstrip(".") + "; " + note[0].lower() + note[1:]
        color_lines.append(line)
    return (
        f"<CAPTION>\n{reply['caption'].strip()}\n</CAPTION>\n"
        f"<PALETTE>\n{reply['overview'].strip()}\n" + "\n".join(color_lines)
        + f"\nLayout: {reply['layout'].strip()}\n</PALETTE>\n"
        f"<GRID>\n{grid_target}\n</GRID>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="selected Molmo2 rows from training.pixels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="override the synthesis API model ID")
    args = parser.parse_args()
    from openai import OpenAI
    client = OpenAI(max_retries=3)
    cfg = read_json("configs/molmo2.json")["synthesis"]
    rows = []
    for item in read_jsonl(args.input):
        size, pixel = item["size"], item["pixel"]
        palette = pixel["palette"]
        if len(palette) != 8:
            raise ValueError("trajectory requires an eight-color teacher")
        letters = [row.translate(str.maketrans("01234567", LETTERS)) for row in pixel["pixels"]]
        grid = np.array([list(row) for row in letters])
        grid_text = json.dumps(palette, separators=(",", ":")) + "\n" + ",".join(
            " ".join(row[i:i + 8] for i in range(0, size, 8)) for row in letters)
        hints = analyse(item["image"], palette, grid)
        prompt = API_PROMPT.format(
            size=size, max_words=MAX_WORDS[size], caption=item["caption"],
            stats_lines="\n".join(f"{s['letter']} = {s['hex']}, {s['fraction']:.1%}, {', '.join(s['where'])}" for s in hints["stats"]),
            merge_lines=json.dumps(hints["merges"]), near_lines=json.dumps(hints["near"]))
        with Image.open(item["image"]) as original:
            original = original.convert("RGB")
            original.thumbnail((1024, 1024))
            content = [image_content(original)]
        content += [image_content(render(pixel, size).resize((512, 512), Image.Resampling.NEAREST)),
                    image_content(mask_sheet(item["image"], palette, grid)), {"type": "text", "text": prompt}]
        for attempt in range(cfg["attempts"]):
            response = client.chat.completions.create(
                model=args.model or cfg["model"], messages=[{"role": "user", "content": content}],
                temperature=cfg["temperature"], max_tokens=8192,
                extra_body={"thinking": {"include_thoughts": False, "budget_tokens": cfg["thinking_budget"]}})
            try:
                target = build_target(json.loads(response.choices[0].message.content), palette, grid_text)
                break
            except (ValueError, TypeError, KeyError):
                if attempt + 1 == cfg["attempts"]:
                    raise
        choice = int(hashlib.sha256(item["id"].encode()).hexdigest(), 16) % len(INSTRUCTION_PROMPTS)
        rows.append({"id": item["id"], "image": item["image"], "size": size,
                     "prompt": INSTRUCTION_PROMPTS[choice].format(size=size), "target": target})
        write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
