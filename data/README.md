# Included data

| File | Contents |
| --- | --- |
| `t2p.json` | 100 object concepts, fixed IDs and categories |
| `candidates.json` | Ordered 500-way retrieval vocabulary; first 100 are targets |
| `i2p.json` | 100 ImageNet-1K validation images, class IDs and original filenames |
| `images/` | Canonical 256×256 ADM/DiT center crops used as I2P inputs |
| `understanding.jsonl` | 600 frozen PIA / Astra-I2P grids, ten choices, and answers |
| `predictions/t2p/` | Seven models, 300 outputs each |
| `predictions/i2p/` | Six models and three algorithms, 300 outputs each |
| `predictions/understanding/` | Saved ten-choice answers |
| `predictions/judge/` | Saved Gemini 3.5 Flash rubric totals |

All paths are relative to the repository root. Predictions contain only the fields
needed to render and score the outputs. Credentials, API payload headers, machine
paths, and experiment logs are not part of the dataset.

ImageNet images originate from [ImageNet](https://www.image-net.org/); their underlying
image rights and ImageNet terms continue to apply. The code license does not relicense
third-party images. PIA grids are deterministic derivatives of those images. The
other grids are model-generated outputs.

Training instruction corpora are distributed separately by
[FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision),
[LLaVA-OneVision-1.5-Instruct](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data),
and [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap), under their own terms.
