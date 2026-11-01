"""Validate and render the paper's palette-indexed representation."""

import re

import numpy as np
from PIL import Image


def validate(pixel, size):
    """Reject malformed outputs; never pad, truncate, or repair a grid."""
    if not isinstance(pixel, dict):
        raise ValueError("output must be an object")
    palette, rows = pixel.get("palette"), pixel.get("pixels")
    if not isinstance(palette, list) or not 2 <= len(palette) <= 8:
        raise ValueError("palette must contain 2–8 colors")
    if not all(isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in palette):
        raise ValueError("colors must use #RRGGBB")
    if not isinstance(rows, list) or len(rows) != size:
        raise ValueError("wrong number of rows")
    grid = []
    for row in rows:
        if isinstance(row, str):
            if len(row) != size or not re.fullmatch(r"[0-7]+", row):
                raise ValueError("rows must contain exactly S palette indices")
            row = [int(c) for c in row]
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("wrong row width")
        if not all(type(i) is int and 0 <= i < len(palette) for i in row):
            raise ValueError("invalid palette reference")
        grid.append(row)
    return {"palette": palette, "pixels": ["".join(map(str, row)) for row in grid]}


def render(pixel, size):
    pixel = validate(pixel, size)
    colors = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in pixel["palette"]], dtype=np.uint8)
    indices = np.array([[int(c) for c in row] for row in pixel["pixels"]])
    return Image.fromarray(colors[indices])


def encode(image):
    """Canonical palette: decreasing pixel count, then increasing RGB."""
    rgb = np.asarray(image.convert("RGB"))
    colors, inverse, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    order = sorted(range(len(colors)), key=lambda i: (-int(counts[i]), tuple(colors[i])))
    palette = ["#" + "".join(f"{c:02X}" for c in colors[i]) for i in order]
    mapping = np.empty(len(order), dtype=int)
    mapping[order] = np.arange(len(order))
    grid = mapping[inverse].reshape(rgb.shape[:2])
    return {"palette": palette, "pixels": ["".join(map(str, row)) for row in grid]}


def flat(image):
    rgb = np.asarray(image)
    return bool(np.all(rgb == rgb[0, 0]))
