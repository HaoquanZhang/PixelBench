# Evaluation protocol

Each generation task has 100 inputs evaluated at 16², 32², and 64². The palette
contains 2–8 opaque `#RRGGBB` colors. Every grid must have exactly S rows and S columns,
with indices referencing the palette. `pixelbench.format.validate` rejects malformed
outputs. It accepts compact digit strings and equivalent stored integer rows.
The renderer performs only a palette lookup.

## Generation

Predictions use JSONL with one row per `(id, size)`:

```json
{"id":"t2p_000","size":16,"pixel":{"palette":["#FFFFFF","#000000"],"pixels":["..."]}}
```

The `pixels` example is abbreviated. Use `pixel: null` for an invalid completed
generation. The evaluator reads all expected IDs from the benchmark manifest, so
omitted predictions also receive zero. Transport failures should be resolved before
reporting a completed benchmark.

| Metric | Definition |
| --- | --- |
| Valid | Fraction of expected outputs passing format validation |
| Flat | Fraction of expected outputs that are valid and render in a single RGB color |
| Top-1 | Correct nearest candidate / all expected outputs |
| MRR (`rr`) | Mean reciprocal target rank over valid outputs only; null if none are valid |
| SigLIP / DINO cosine | Mean paired cosine over all expected outputs, with invalid outputs zero |

Rank is `1 + count(candidate_similarity > target_similarity)`, matching the appendix.
The full candidate bank remains fixed when scoring one resolution or a partial output file.

**T2P:** SigLIP-SO400M encodes the rendered image and the candidate strings
`a pixel art icon depicting {label}`. The first 100 candidates are the benchmark
labels; the remaining 400 are distractors. Images are enlarged directly to 384²
using nearest-neighbor interpolation, followed by the checkpoint's normalization.
Text uses the checkpoint's maximum-length padding. The six rubric components for
Gemini 3.5 Flash are identity /20, defining parts /25, structure /20, detail /20,
pixel craft /10, and palette /5. The judge sees a nearest-neighbor 512² rendering.

**I2P:** Both query and the 100 source candidates use pooled SigLIP-SO400M features.
Generated images are enlarged to 384² with nearest-neighbor interpolation; source
photos use bicubic interpolation. DINOv2-Large uses the CLS feature: images are first
256², normalized with ImageNet mean/std, and then resized to 224² with PyTorch bicubic
interpolation (`align_corners=False`, no antialiasing). All feature vectors are L2-normalized.

The I2P **Aggregated** column is relative to the comparison set: average each metric
over resolutions, min–max normalize each of the three columns across the six models
and three algorithms, and average the normalized columns on a 0–100 scale. Use
`pixelbench.table` with all nine evaluator outputs to reproduce this column.

## Pixel understanding

`data/understanding.jsonl` fixes all 600 questions: 100 × 3 resolutions from PIA,
and 100 × 3 from GPT-6 Astra I2P. The model receives only `palette`, `pixels`, and
the ten choices. Sample IDs and gold labels are not passed to the model.
The answer is the last standalone A–J letter; a missing or unparseable answer is wrong.
Report per-source/per-resolution accuracy and the overall mean across 600 questions.

## Downstream understanding after training

[configs/evaluation.json](configs/evaluation.json) lists the primary 16 Molmo2
benchmarks and 13 Qwen additive benchmarks. Use fixed questions, identical prompts,
and greedy decoding across compared checkpoints. Use benchmark-specific rule-based
grading, with the benchmark's LLM judge for free-form BabyVision and PerceptionBench.
MathVista uses direct answers. ChartQA uses 5% numeric tolerance and exact year matching.

The aggregation unit is a question for ordinary benchmarks, a two-question pair for
MMVP, an original question with all circular option shifts for MMBench, and an original
image plus nine recolorings for ColorBench-Robust. Every member must be correct for a
group to be correct. Freeze these groups in a JSONL manifest:

```json
{"benchmark":"MMVP","id":"pair_1","items":["question_1","question_2"]}
```

Store each graded model call as `{"benchmark":"MMVP","id":"question_1","correct":true}`.
Then aggregate with:

```bash
python -m training.evaluate --suite molmo2 --units evaluation-units.jsonl \
  --verdicts verdicts.jsonl --output outputs/downstream.json
```

Example average weights each official evaluation unit equally. Benchmark average
weights each benchmark equally. The aggregator requires complete units; it does not
silently score an easier subset. The downstream datasets and their inference/judging
harnesses are obtained from their respective benchmark releases.
