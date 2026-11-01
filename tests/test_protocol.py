from pathlib import Path

import numpy as np
import pytest

from pixelbench.evaluate import index_predictions, retrieval, summarize
from pixelbench.format import encode, flat, render, validate
from pixelbench.io import read_json, read_jsonl
from training.evaluate import aggregate
from training.pixels import choose_balanced
from training.prepare import assemble
from pixelbench.table import aggregate as aggregate_table


def pixel(size=16):
    return {"palette": ["#FFFFFF", "#000000"], "pixels": ["01" * (size // 2)] * size}


@pytest.mark.parametrize("mutation", [
    lambda p: p["pixels"].pop(),
    lambda p: p["pixels"].__setitem__(0, "0" * 15),
    lambda p: p["pixels"].__setitem__(0, "2" * 16),
    lambda p: p["palette"].__setitem__(0, "#FFF"),
    lambda p: p["pixels"].__setitem__(0, [False] * 16),
])
def test_invalid_outputs_are_rejected(mutation):
    value = pixel()
    mutation(value)
    with pytest.raises(ValueError):
        validate(value, 16)


def test_render_roundtrip_and_flat_rgb():
    image = render(pixel(), 16)
    np.testing.assert_array_equal(render(encode(image), 16), image)
    assert not flat(image)
    value = pixel()
    value["palette"] = ["#FFFFFF", "#ffffff"]
    assert flat(render(value, 16))


def test_retrieval_uses_paper_tie_formula():
    assert retrieval([.8, .8, .2], 1)["top1"] == 1
    assert retrieval([.8, .7, .2], 1)["rr"] == .5


def test_invalid_denominators():
    rows = [{"id": "a", "size": 16, "valid": 1, "top1": 1, "rr": .5},
            {"id": "b", "size": 16, "valid": 0, "top1": 0, "rr": 0}]
    result = summarize(rows)["16"]
    assert result["top1"] == .5
    assert result["rr"] == .5


def test_i2p_aggregate_normalizes_across_methods():
    values = {
        "a": {"averages": {"top1": .4, "cosine": .2, "dino": .1}},
        "b": {"averages": {"top1": .8, "cosine": .6, "dino": .3}},
        "c": {"averages": {"top1": .6, "cosine": .4, "dino": .2}},
    }
    rows = aggregate_table(values)
    np.testing.assert_allclose([r["aggregated"] for r in rows], [0, 100, 50])


def test_official_groups_and_example_weighting():
    units = [{"benchmark": "MMVP", "id": "pair", "items": ["1", "2"]},
             {"benchmark": "other", "id": "a", "items": ["a"]},
             {"benchmark": "other", "id": "b", "items": ["b"]}]
    verdicts = [{"benchmark": "MMVP", "id": "1", "correct": True},
                {"benchmark": "MMVP", "id": "2", "correct": False},
                {"benchmark": "other", "id": "a", "correct": True},
                {"benchmark": "other", "id": "b", "correct": True}]
    result = aggregate(verdicts, units)
    assert result["example_average"] == 2 / 3
    assert result["benchmark_average"] == .5
    with pytest.raises(ValueError):
        aggregate(verdicts[1:], units)


def test_unique_balanced_nested_pixel_selection():
    pool = [{"id": str(i), "cosine_16": i, "cosine_32": i} for i in range(100)]
    a = choose_balanced(pool, 10, {"16": .4, "32": .8})
    b = choose_balanced(pool, 20, {"16": .4, "32": .8}, existing=a)
    assert b[:10] == a
    assert len({r["id"] for r in b}) == 20
    assert sum(r["size"] == 16 for r in b) == 10
    assert all(int(r["id"]) >= (60 if r["size"] == 16 else 20) for r in b)


def test_qwen_paper_counts():
    cfg = read_json("configs/qwen.json")
    instructions = [{"image": "fv"}] * 64000
    pixels = [{"image": f"pixel_{i}"} for i in range(3000)]
    assert len(assemble(instructions, [], "qwen", "fv", cfg)) == 64000
    assert len(assemble(instructions, pixels, "qwen", "additive", cfg)) == 67000
    assert len(assemble(instructions, pixels, "qwen", "fixed", cfg)) == 64000
    assert cfg["steps"] == {"fv": 1000, "additive": 1047, "fixed": 1000}


def test_bundled_benchmark():
    t2p, i2p = read_json("data/t2p.json"), read_json("data/i2p.json")
    candidates = read_json("data/candidates.json")
    assert len(t2p) == len(i2p) == 100
    assert len(candidates) == len(set(candidates)) == 500
    assert [r["label"] for r in t2p] == candidates[:100]
    assert all(Path(r["image"]).is_file() for r in i2p)
    questions = read_jsonl("data/understanding.jsonl")
    assert len(questions) == 600
    assert len({(r["id"], r["size"]) for r in questions}) == 600
    for q in questions:
        validate(q["pixel"], q["size"])
        assert len(q["choices"]) == len(set(q["choices"])) == 10
        assert q["answer"] in "ABCDEFGHIJ"
    for task in ("t2p", "i2p"):
        for path in Path(f"data/predictions/{task}").glob("*.jsonl"):
            rows = read_jsonl(path)
            assert len(index_predictions(rows)) == 300
            for row in rows:
                if row["pixel"] is not None:
                    validate(row["pixel"], row["size"])
