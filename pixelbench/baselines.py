"""The three palette-matched baselines in the main I2P table."""
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import lab2rgb, rgb2lab
from skimage.segmentation import slic
from .format import encode
from .io import read_json, write_jsonl
from .ssim import ssim_downscale

SEED = 20260807

def pia(source: Image.Image, size: int) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"), dtype=np.float64) / 255.0
    lab = rgb2lab(rgb)
    labels = slic(
        rgb,
        n_segments=size * size,
        compactness=45,
        sigma=1,
        convert2lab=True,
        enforce_connectivity=True,
        start_label=0,
        channel_axis=-1,
    )
    label_count = int(labels.max()) + 1
    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=label_count)
    means = np.stack(
        [
            np.bincount(
                flat_labels, weights=lab[..., channel].ravel(),
                minlength=label_count,
            )
            / np.maximum(counts, 1)
            for channel in range(3)
        ],
        axis=1,
    )
    palette = KMeans(
        n_clusters=min(8, label_count),
        random_state=SEED,
        n_init=10,
    ).fit(means, sample_weight=counts)
    assigned = palette.cluster_centers_[palette.labels_]
    height, width = labels.shape
    output_lab = np.empty((size, size, 3), dtype=np.float64)
    for y in range(size):
        source_y = min(int((y + 0.5) * height / size), height - 1)
        for x in range(size):
            source_x = min(int((x + 0.5) * width / size), width - 1)
            output_lab[y, x] = assigned[labels[source_y, source_x]]
    output_rgb = np.clip(lab2rgb(output_lab) * 255.0, 0, 255).round()
    return Image.fromarray(output_rgb.astype(np.uint8))

def transform(image, size, method):
    if method == "pia":
        return pia(image, size)
    if method == "ssim":
        values = ssim_downscale(np.asarray(image), image.width // size)
        image = Image.fromarray(np.clip(values * 255, 0, 255).round().astype(np.uint8))
    else:
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image.quantize(colors=8, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["nearest", "ssim", "pia"], required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for item in read_json("data/i2p.json"):
        with Image.open(item["image"]) as image:
            for size in args.sizes:
                rows.append({"id": item["id"], "size": size, "pixel": encode(transform(image.convert("RGB"), size, args.method))})
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
