"""Aggregate downstream verdicts using official evaluation units."""

import argparse
from collections import defaultdict

from pixelbench.io import read_json, read_jsonl, write_json


def aggregate(rows, units):
    """Each unit lists all required model calls, including option shifts/recolorings."""
    predictions = {}
    for row in rows:
        key = row["benchmark"], row["id"]
        if key in predictions:
            raise ValueError(f"duplicate verdict: {key}")
        predictions[key] = row["correct"]
    scores = defaultdict(list)
    for unit in units:
        keys = [(unit["benchmark"], i) for i in unit["items"]]
        if not keys or any(key not in predictions for key in keys):
            raise ValueError(f"incomplete evaluation unit: {unit['benchmark']}/{unit['id']}")
        scores[unit["benchmark"]].append(int(all(predictions[key] for key in keys)))
    summary = {name: {"n": len(values), "accuracy": sum(values) / len(values)} for name, values in scores.items()}
    return {"benchmarks": summary,
            "example_average": sum(sum(v) for v in scores.values()) / sum(map(len, scores.values())),
            "benchmark_average": sum(s["accuracy"] for s in summary.values()) / len(summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", required=True, help="JSONL: benchmark, id, correct")
    parser.add_argument("--units", required=True, help="frozen JSONL: benchmark, id, items (model-call IDs)")
    parser.add_argument("--suite", choices=["molmo2", "qwen"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    units = read_jsonl(args.units)
    result = aggregate(read_jsonl(args.verdicts), units)
    expected = read_json("configs/evaluation.json")[args.suite]
    observed = {k: v["n"] for k, v in result["benchmarks"].items()}
    if observed != expected:
        raise ValueError(f"evaluation units do not match {args.suite}: {observed}")
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != "benchmarks"})


if __name__ == "__main__":
    main()
