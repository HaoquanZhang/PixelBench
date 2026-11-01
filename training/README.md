# Pixel Reconstruction Tuning

The configuration files define the paper's data amounts, initialization, optimization,
and seeds. Qwen and Molmo2 use separate environments. Model training is launched
explicitly; data preparation does not start a training run.

## Data format

Instruction rows use one image and one question/answer pair:

```json
{"id":"example_1","image":"training_data/images/example_1.jpg","prompt":"Describe the image.","target":"A bird on a branch."}
```

Export FineVision or LLaVA-OneVision-1.5 single-image conversations to this flat format,
with one row per supervised answer. Resolve relative image paths from this repository.
Provide a text file containing SHA-256 hashes of evaluation images to exclude exact
matches during instruction and Pixel selection.

Pixel source manifests use `id` and `image`; PixMo-Cap rows additionally contain the
human-written `caption`. Qwen inputs are the accepted 256×256 training crops. All
three resolution variants use the same source-image manifest and sampling seed.

```json
{"id":"pixmo_example","image":"training_data/pixmo/example.jpg","caption":"A human-written dense caption."}
```

## Qwen3-VL

```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -e '.[qwen,baselines]'
```

Prepare 64,000 FineVision occurrences with answers capped at 2,048 tokens. This pool
is shared by the FV baseline and every additive Pixel arm.

```bash
python -m training.instructions qwen --input finevision.jsonl \
  --tokenizer Qwen/Qwen3-VL-4B-Instruct --eval-hashes eval-hashes.txt \
  --output training_data/fv64k.jsonl

OMP_NUM_THREADS=1 python -m training.pixels qwen --pool pixel-sources.jsonl \
  --eval-hashes eval-hashes.txt --resolution mix \
  --output training_data/qwen-pixel3k.jsonl

python -m training.prepare qwen --instructions training_data/fv64k.jsonl \
  --pixels training_data/qwen-pixel3k.jsonl --arm additive \
  --output training_data/qwen-additive.jsonl

torchrun --standalone --nproc_per_node 4 -m training.qwen --size 4b --arm additive \
  --data training_data/qwen-additive.jsonl --output checkpoints/qwen4b-pixel
```

`--size 2b|4b|8b` selects the released Instruct checkpoint. `--model` accepts a local
checkpoint. All parameters, including the vision encoder, are trained. The script
uses assistant-only loss, bf16, AdamW, LR 1e-5, cosine decay, 3% warmup, zero weight
decay, gradient clipping 1.0, and seed 42. Four GPUs × batch 4 × accumulation 4 gives
an effective batch of 64; DeepSpeed ZeRO-2 is enabled.

The additive arm has 67,000 examples and 1,047 optimizer steps. For the FV baseline,
run `training.prepare` and `training.qwen` with `--arm fv` and omit `--pixels` during
preparation: 64,000 examples, 1,000 steps. `--arm fixed` selects the appendix control
with 61,000 FV + 3,000 Pixel examples and 1,000 steps.

For resolution ablations, repeat `training.pixels` with `--resolution 16`, `32`, or
`64`. Source images are sampled without replacement and the same 3,000 image IDs are
used across runs. The mixed arm assigns exactly one resolution to each image, evenly
across 16/32/64. PIA is recomputed at native resolution; 64² targets are not capped
at the FineVision answer limit.

## Molmo2-4B

Use the official [Molmo2 repository](https://github.com/allenai/molmo2) and its native
**Molmo2-4B-Pretrain step32000** checkpoint. This is instruction tuning from the image
pretraining checkpoint; there is no alignment stage.

```bash
python -m venv .venv-molmo2
source .venv-molmo2/bin/activate
git clone https://github.com/allenai/molmo2 third_party/molmo2
pip install -e 'third_party/molmo2[train]'
pip install -e '.[eval,baselines]'
export PYTHONPATH="$PWD/third_party/molmo2${PYTHONPATH:+:$PYTHONPATH}"
```

Prepare 500,000 single-image OV instruction rows. Answers longer than 2,048 tokens
are removed and images are limited to a 1,024-pixel long side.

```bash
python -m training.instructions molmo2 --input onevision.jsonl \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 --eval-hashes eval-hashes.txt \
  --output training_data/ov500k.jsonl
```

Start with a 20k-image PixMo-Cap pool. Remove evaluation matches, text-heavy/composite
layouts, and aspect ratios above 1.33. Generate deterministic eight-color PIA grids,
then rank candidates by paired SigLIP2 source/grid cosine. Keep the top 40% at 16² and
top 80% at 32². The main set contains 1,000 unique images per resolution.

```bash
OMP_NUM_THREADS=1 python -m training.pixels molmo2 --pool pixmo20k.jsonl \
  --eval-hashes eval-hashes.txt --count 2000 --output training_data/molmo-selected2k.jsonl

export OPENAI_API_KEY='your-api-key'
# Set OPENAI_BASE_URL to a Gemini-compatible chat endpoint if needed.
python -m training.trajectories --input training_data/molmo-selected2k.jsonl \
  --output training_data/molmo-pixel2k.jsonl

python -m training.prepare molmo2 --instructions training_data/ov500k.jsonl \
  --pixels training_data/molmo-pixel2k.jsonl --output training_data/molmo-additive.jsonl
```

Gemini 3.8 Flash writes only the caption and palette explanation at temperature 0.4
and thinking budget 2,048. It receives the original image, rendered grid, per-letter
mask sheet, human caption, and measured palette/layout hints. The deterministic grid
is appended unchanged. Every palette letter must be discussed exactly once and in
order. Three user-prompt variants are assigned deterministically.

Targets use `<CAPTION>`, `<PALETTE>`, and `<GRID>` sections. Inside `<GRID>`, the first
line is the hex palette as a JSON list, followed by comma-separated rows of letters
`a`–`h`, grouped into blocks of eight letters with spaces. Palette order follows pixel
share, most-used first.

Launch the following on **each of four nodes with eight A100-80GB GPUs**. Supply a
shared checkpoint/data/output filesystem, a common `MASTER_ADDR`/`MASTER_PORT`, and a
different `NODE_RANK` (0–3) on each node.

```bash
torchrun --nnodes 4 --nproc_per_node 8 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  -m training.molmo2 --checkpoint /path/to/Molmo2-4B-Pretrain/step32000 \
  --data training_data/molmo-additive.jsonl --pixel-count 2000 --seed s0 \
  --output checkpoints/molmo2-pixel2k-s0
```

The adapter uses the official FSDP2 trainer and dynamic packer: 128 sequences × 4,096
tokens, one image per sample, up to eight high-resolution crops, and maximum answer
length 1,800. All modules are trainable. LLM LR is 1e-5; vision/connector LR is 5e-6;
warmup is 60 steps and the final LR multiplier is 0.1. Use `--pixel-count 0` with the
OV-only mixture for the baseline.

| Pixel rows | Instruction rows | Steps | Seed pairs |
| ---: | ---: | ---: | --- |
| 0 | 500,000 | 1,190 | s0, s1, s2, s3 |
| 1,000 | 500,000 | 1,194 | s1, s2 |
| 2,000 | 500,000 | 1,198 | s0, s1, s2, s3 |
| 4,000 | 500,000 | 1,206 | s0, s1 |
| 8,000 | 500,000 | 1,222 | s0, s1 |

| Seed | Training seed | Data seed |
| --- | ---: | ---: |
| s0 | 6198 | 50189 |
| s1 | 271828 | 314159 |
| s2 | 1729 | 424242 |
| s3 | 31337 | 777 |

For 4k/8k, expand the PixMo pool with unused images and pass `--existing` to
`training.pixels` to preserve the smaller selection. Pass the corresponding count
to the trainer. Obtain a balanced 1k set with `--count 1000`.

Convert a native checkpoint with the official exporter:

```bash
python -m olmo.hf_model.convert_molmo2_to_hf \
  checkpoints/molmo2-pixel2k-s0/step1198 checkpoints/molmo2-pixel2k-s0-hf \
  --use_bfloat16 --attn_implementation sdpa --override_max_model_len 4096
```

Downstream evaluation uses the frozen benchmark units and aggregation in
[EVALUATION.md](../EVALUATION.md). Large upstream corpora, model weights, and the
individual downstream benchmark harnesses are obtained separately.
