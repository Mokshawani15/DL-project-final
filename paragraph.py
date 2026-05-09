"""
paragraph.py — Caption Expansion Module
Uses GPT-2 (local, no API key needed) to expand a short image caption
into a detailed descriptive paragraph of 300–500 words.
"""

import torch
from transformers import pipeline, set_seed

# Device setup
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU

# Load pipeline
_generator = pipeline(
    "text-generation",
    model="gpt2-medium",
    device=device,
)

# Fix pad_token_id safely (VERY IMPORTANT for GPT-2)
_generator.tokenizer.pad_token = _generator.tokenizer.eos_token

set_seed(42)

_PROMPT_TEMPLATE = """\
You are an expert image description writer. Based on the following short image caption, \
write a rich, detailed descriptive paragraph of 300 to 500 words that vividly describes \
the scene, objects, people, colours, lighting, mood, and any activities visible in the image.

Short caption: {caption}

Detailed description:"""


def expand_caption(caption: str) -> str:
    """
    Expand a short AI caption into a 300–500 word descriptive paragraph.
    """
    if not caption or not caption.strip():
        return "No caption provided to expand."

    prompt = _PROMPT_TEMPLATE.format(caption=caption.strip())

    try:
        # ✅ Stable sampling configuration
        outputs = _generator(
            prompt,
            max_new_tokens=200,              # increased for better paragraph length
            do_sample=True,
            temperature=0.7,                 # safe range
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            num_beams=1,                     # 🚫 disable beam search
            pad_token_id=_generator.tokenizer.eos_token_id,
            eos_token_id=_generator.tokenizer.eos_token_id,
            return_full_text=True
        )

    except RuntimeError as e:
        print("⚠️ Sampling failed, switching to safe fallback:", e)

        # ✅ Deterministic fallback (NO sampling)
        outputs = _generator(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
            pad_token_id=_generator.tokenizer.eos_token_id,
            eos_token_id=_generator.tokenizer.eos_token_id,
            return_full_text=True
        )

    generated_text: str = outputs[0]["generated_text"]

    # Extract only generated part
    if "Detailed description:" in generated_text:
        paragraph = generated_text.split("Detailed description:", 1)[1].strip()
    else:
        paragraph = generated_text[len(prompt):].strip()

    # Clean weird artifacts
    paragraph = paragraph.replace("\n", " ").strip()

    # Trim to ~500 words max
    words = paragraph.split()
    if len(words) > 500:
        paragraph = " ".join(words[:500])
        for sep in [".", "!", "?"]:
            last = paragraph.rfind(sep)
            if last != -1:
                paragraph = paragraph[: last + 1]
                break

    return paragraph