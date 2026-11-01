"""Task prompts used by PixelBench."""

import json

LETTERS = "ABCDEFGHIJ"


def t2p_prompt(label, size):
    return f'''Create exactly 1 recognizable {size}x{size} pixel-art icons, one for each of these English labels in this exact order:
{label}

Return JSON only, with no Markdown fence or commentary, using:
{{"images": [{{"label": "{label}",
 "palette": ["#FFFFFF", "#D62828", "#2A9D3F"],
 "pixels": ["000...", "... exactly {size} strings"]}}]}}

Requirements:
- Every icon must have exactly {size} rows and {size} columns.
- Use 2 to 8 opaque RGB colors, including a background color.
- Encode each pixel as one digit from 0 to 7.
- Favor simple silhouettes, high contrast, and centered subjects.
- Make each object recognizable at {size}x{size} resolution; use no text.'''

def i2p_prompt(sample_id: str, size: int) -> str:
    example_row = "0" * size
    return f"""
Convert the supplied reference image into a faithful {size}x{size} pixel-art
image. Preserve the main subject's identity, silhouette, pose, composition, and
dominant colors. Simplify fine detail only as required by the pixel resolution.
Do not use any object name or text in the image.

Return JSON only, with no Markdown or commentary:
{{
  "sample_id": "{sample_id}",
  "palette": ["#FFFFFF", "#111111"],
  "pixels": [
    "{example_row}",
    "... exactly {size} strings, each exactly {size} digits long"
  ]
}}

Use 2 to 8 opaque #RRGGBB colors. Encode each pixel as a digit from 0 to 7
indexing the palette. Return compact row strings, not arrays.
""".strip()

def understanding_prompt(size: int, payload: dict, choices: list[str]) -> str:
    options = "\n".join(f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices))
    return (
        f'This is a {size}x{size} pixel-art icon encoded as JSON.\n'
        f'"palette" is 2-8 hex colors. "pixels" are {size} row strings; '
        "each character is a palette index 0-7.\n"
        f"Identify the depicted object. Reply with ONE letter only ({', '.join(LETTERS[:len(choices)])}).\n\n"
        f"{json.dumps(payload, separators=(',', ':'))}\n\n{options}"
    )

def judge_prompt(label: str, size: int) -> str:
    return f"""
You are a severe benchmark evaluator for text-to-pixel generation. The target
concept is {json.dumps(label)}. The image is a nearest-neighbor enlargement of a
native {size}x{size} pixel-art image; large square pixels are intentional.

Recognizability alone is not sufficient for a high score. A generic icon that
suggests the broad category but omits identity-bearing parts, correct structure,
or resolution-appropriate details must receive a moderate score. Do not reward
complexity, visual polish, or colorfulness unless they accurately serve the
target.

Before assigning scores:
- Write a literal observation using only what is visibly present.
- List 3-6 concrete visual attributes expected in a strong depiction.
- For each expected attribute, identify visible evidence or state that it is
  missing, malformed, merged, or ambiguous.
- Identify at least one visible limitation unless the depiction is genuinely
  exceptional.

Score these independent dimensions:
1. target_identity (0-20): Exact concept-level match, not merely broad-category
   resemblance. 18-20 unmistakable and specific; 12-17 recognizable but
   simplified; 5-11 ambiguous/generic; 0-4 wrong or absent.
2. defining_parts (0-25): Presence and correctness of identity-bearing parts
   and their relations. Missing a major defining part must materially reduce
   the score.
3. structure_proportions (0-20): Shape, topology, relative placement, symmetry
   where appropriate, pose, and proportions. Penalize fused, detached, or
   malformed parts.
4. detail_specificity (0-20): Useful target-specific details beyond the minimum
   silhouette. Penalize generic templates, omitted secondary parts, noisy
   pseudo-detail, and details that contradict the target. Judge relative to the
   native resolution; do not demand photorealism.
5. pixel_craft (0-10): Deliberate pixel clusters, clean edges, coherent fills,
   and economical use of the native grid.
6. palette_readability (0-5): Palette coherence, contrast, and separation of
   important parts.

Total-score calibration:
- 95-100: exceptional and nearly flawless; reserve this range for rare cases.
- 85-94: strong and specific, with only minor visible defects.
- 70-84: clearly recognizable but meaningfully simplified or structurally
  imperfect.
- 50-69: broad concept is plausible, but important parts/details are missing.
- 25-49: weak, generic, or substantially malformed resemblance.
- 0-24: wrong, unreadable, or absent subject.
Do not assign a maximum component score when a concrete defect exists in that
component. Do not let the provided target label override what is actually
visible.

Return JSON only:
{{
  "literal_observation": "what is visibly present without interpreting intent",
  "expected_attributes": ["attribute 1", "attribute 2", "attribute 3"],
  "attribute_evidence": [
    {{"attribute": "...", "verdict": "present|partial|missing|incorrect", "evidence": "..."}}
  ],
  "visible_limitations": ["specific limitation"],
  "target_identity": {{"score": 0, "reason": "brief evidence"}},
  "defining_parts": {{"score": 0, "reason": "brief evidence"}},
  "structure_proportions": {{"score": 0, "reason": "brief evidence"}},
  "detail_specificity": {{"score": 0, "reason": "brief evidence"}},
  "pixel_craft": {{"score": 0, "reason": "brief evidence"}},
  "palette_readability": {{"score": 0, "reason": "brief evidence"}},
  "confidence": "low|medium|high"
}}
""".strip()
