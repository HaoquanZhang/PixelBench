"""I2P table aggregation: min-max normalize each average column across methods."""

import argparse
from pathlib import Path

import numpy as np

from .io import read_json, write_json


def aggregate(results):
    names = list(results)
    values = np.array([[results[name]["averages"][k] for k in ("top1", "cosine", "dino")] for name in names])
    spans = np.ptp(values, axis=0)
    if np.any(spans == 0):
        raise ValueError("min-max aggregation requires a nonconstant comparison set for each metric")
    scores = ((values - values.min(axis=0)) / spans).mean(axis=1) * 100
    return [{"method": name, **results[name]["averages"], "aggregated": float(score)} for name, score in zip(names, scores)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", help="I2P evaluator JSON files for all compared methods")
    parser.add_argument("--output", default="outputs/i2p-table.json")
    args = parser.parse_args()
    rows = aggregate({Path(path).stem: read_json(path) for path in args.scores})
    write_json(args.output, rows)
    for row in sorted(rows, key=lambda row: -row["aggregated"]):
        print(f"{row['method']}: {row['aggregated']:.2f}")


if __name__ == "__main__":
    main()
