<h1 align="center">
  <img src="assets/pixelbench-logo.png" alt="PixelBench" width="340" height="58">
</h1>

<h3 align="center">Can LLMs Understand and Generate Images Pixel by Pixel?</h3>

<p align="center">
  <a href="https://haoquanzhang.github.io/">Haoquan Zhang</a> ·
  <a href="https://zaychikliu.github.io/">Feiyang Liu</a> ·
  <a href="https://tianshijing.github.io/">Juanxi Tian</a> ·
  <a href="https://li-wanhua.github.io/">Wanhua Li</a> ·
  <a href="https://kennymckormick.github.io/">Haodong Duan</a> ·
  <a href="https://wyliu.com/">Weiyang Liu</a>
</p>

<p align="center">
  <a href="https://haoquanzhang.github.io/pixelbench/">Project Blog</a> ·
  <a href="https://arxiv.org/abs/2611.02540">Paper</a> ·
  <a href="https://huggingface.co/collections/haoquan03/pixel-reconstruction-tuning">Models</a> ·
  <a href="https://x.com/Haoquan03/status/2117657081981685860">X Post</a>
</p>

![Four 16 × 16 model-generated objects—mushroom, owl, cherry, and turtle—transitioning from colored pixels to palette index digits](assets/teaser.png)

PixelBench evaluates image generation and understanding through ordinary text tokens.
A model produces a small RGB palette and a grid of palette indices; rendering is a
deterministic lookup. Pixel Reconstruction Tuning (PRT) uses this representation as
visual supervision for Molmo2 and Qwen3-VL.

![Compact pixel representation: a rendered mushroom, its small RGB palette, and the corresponding grid of palette indices](assets/representation.png)

| Task | Input | Evaluation set | Main metrics |
| --- | --- | --- | --- |
| Text-to-pixel | Object label | 100 concepts × 3 resolutions | 500-way SigLIP Top-1, VLM judge |
| Image-to-pixel | Reference image | 100 ImageNet crops × 3 resolutions | 100-way Top-1, SigLIP cosine, DINOv2 cosine |
| Pixel understanding | Palette + grid as text | 600 ten-choice questions | Accuracy |

The benchmark data and saved predictions are included in `data/`. No dataset download
or Git LFS is needed for PixelBench. Training corpora and model weights are separate.

<details>
<summary><strong>Explore benchmark results</strong> — generation, understanding, and resolution scaling</summary>

### Text-to-pixel

![Text-to-pixel examples and 500-way SigLIP Top-1 accuracy across models](assets/t2p.png)

### Image-to-pixel

![Image-to-pixel reconstructions and SigLIP similarity, with source images for comparison](assets/i2p.png)

The generation scores above average 16², 32², and 64² results; example images show
individual 32² outputs. See the [evaluation protocol](EVALUATION.md) for metric definitions.

### Pixel understanding

![Ten-choice image recognition from palette indices, measured across 600 questions](assets/understanding.png)

### Resolution scaling

![Model outputs at increasing pixel resolutions, showing detail improvements and generation failures](assets/scaling.png)

The resolution sweep uses a separate 50-image subset. Explore the full interactive
results on the [project page](https://haoquanzhang.github.io/pixelbench/).

</details>

## Setup

Use Python 3.11. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[eval,baselines,test]'
```

Install a CUDA-compatible PyTorch build for GPU scoring. The encoder defaults are
`google/siglip-so400m-patch14-384` and `facebook/dinov2-large`.
Use `--siglip`, `--dino`, and `--device cpu` to select local weights or CPU execution.

## Models

The 14 checkpoints used in the paper's main comparisons are available in the
[Pixel Reconstruction Tuning](https://huggingface.co/collections/haoquan03/pixel-reconstruction-tuning-6ae7775931cc30228f5ca2f8)
collection on Hugging Face, including Molmo2 and Qwen3-VL PRT models and their baselines.

## Evaluate saved outputs

Pixel understanding needs no model weights:

```bash
python -m pixelbench.evaluate understanding \
  --predictions data/predictions/understanding/gpt6astra.jsonl \
  --output outputs/understanding.json
```

Recompute generation metrics using the paper's encoders:

```bash
python -m pixelbench.evaluate t2p \
  --predictions data/predictions/t2p/gpt6astra.jsonl \
  --judge data/predictions/judge/gpt6astra.jsonl \
  --output outputs/t2p.json

python -m pixelbench.evaluate i2p \
  --predictions data/predictions/i2p/gpt6astra.jsonl \
  --output outputs/i2p.json
```

Each result contains per-item scores and per-resolution averages. Invalid and missing
generations receive zero for Top-1, cosine, and judge scores. **MRR is averaged over
valid outputs**, as in the appendix. A flat fill is a valid image rendered in one RGB
color. Main-table resolutions are 16, 32, and 64.

T2P aggregate scores average the three Top-1 percentages and three judge scores.
I2P aggregate scores first min–max normalize the three resolution-averaged metric
columns across the compared methods, then average them. After scoring all six models
and three baselines, run `python -m pixelbench.table outputs/i2p-*.json` with those
nine result files. A single method has no independent I2P aggregate score.
See [the evaluation protocol](EVALUATION.md) for preprocessing and scoring details.

## Run a model

The runner uses a standard chat-completions API, including compatible local servers.
Credentials are read from the environment.

```bash
export OPENAI_API_KEY='your-api-key'
# export OPENAI_BASE_URL='https://your-provider.example/v1'

python -m pixelbench.run t2p --model YOUR_MODEL_ID --preset gpt6astra \
  --output outputs/my-model-t2p.jsonl
python -m pixelbench.run i2p --model YOUR_MODEL_ID --preset gpt6astra \
  --output outputs/my-model-i2p.jsonl
python -m pixelbench.run understanding --model YOUR_MODEL_ID \
  --output outputs/my-model-understanding.jsonl
python -m pixelbench.run judge --model gemini-3.5-flash \
  --predictions outputs/my-model-t2p.jsonl --output outputs/my-model-judge.jsonl
```

Choose the matching model preset in [configs/inference.json](configs/inference.json).
Model IDs and reasoning parameter names depend on the API provider; `--settings`
accepts a JSON object of chat-completion parameters to override the preset.
The judge uses Gemini 3.5 Flash with reasoning disabled; set your provider's reasoning
switch in that settings file. Completed invalid generations are retained and scored
zero. Transient API failures and token-limit truncations are retried. Rerunning a
command resumes from completed `(id, size)` entries.

Generate the three main-table algorithmic baselines:

```bash
OMP_NUM_THREADS=1 python -m pixelbench.baselines --method pia --output outputs/pia.jsonl
```

`--method nearest` and `--method ssim` use median-cut quantization; every baseline has
an eight-color budget. PIA uses SLIC in Lab space and area-weighted eight-means.

<details>
<summary><strong>View the GPT-6 Astra gallery</strong> — 18 examples at 32 × 32</summary>

![Sixteen image-to-pixel reconstructions and two text-to-pixel generations by GPT-6 Astra, all at 32 × 32](assets/gallery-32x32.png)

</details>

## Train with Pixel supervision

![Pixel Reconstruction Tuning gains for Molmo2-4B on 16 visual understanding benchmarks](assets/prt.png)

The chart shows the change from adding 2,000 PRT samples to 500k instruction pairs
for Molmo2-4B, with a +0.6 average score change across the 16 benchmarks.

| Setting | Instruction data | Added Pixel data | Steps | Effective batch |
| --- | --- | --- | --- | --- |
| Molmo2 OV | OV 500k | 0 | 1,190 | 128 packed sequences |
| Molmo2 OV + Pixel | OV 500k | 2k unique trajectories, 16/32 | 1,198 | 128 packed sequences |
| Qwen FV | FineVision 64k | 0 | 1,000 | 64 |
| Qwen FV + Pixel | FineVision 64k | 3k unique grids, 16/32/64 | 1,047 | 64 |
| Qwen fixed-budget control | FineVision 61k | 3k unique grids | 1,000 | 64 |

Molmo2 starts from **Molmo2-4B-Pretrain, step 32,000**, with instruction tuning only.
Qwen starts from **Qwen3-VL-Instruct, 2B/4B/8B**. Both fine-tune all modules.
The implementations, data preparation commands, quantity ablations, seeds, and
environment setup are in [training/README.md](training/README.md).

## Repository

```text
pixelbench/    prompts, API runner, rendering, metrics, three baselines
data/          fixed benchmark inputs and saved predictions
configs/       paper inference, training, and evaluation settings
training/      data preparation and Molmo2 / Qwen trainers
tests/         format, scoring, grouping, and data-integrity checks
```

```bash
python -m pytest -q
```

## Citation

```bibtex
@article{zhang2026pixelbench,
    title = {{PixelBench}: Can {LLMs} Understand and Generate Images Pixel by Pixel?},
    author = {Zhang, Haoquan and Liu, Feiyang and Tian, Juanxi and Li, Wanhua and Duan, Haodong and Liu, Weiyang},
    journal = {arXiv preprint arXiv:2611.02540},
    year = {2026},
    url = {https://haoquanzhang.github.io/pixelbench/}
}
```

Code is MIT licensed. See [data/README.md](data/README.md) for data sources and terms.
