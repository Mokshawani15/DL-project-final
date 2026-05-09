"""
caption.py — Image Captioning Module
Uses a fine-tuned BLIP model (or falls back to the base model) to
generate a short caption for an input image.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# ── Model path: uses fine-tuned model if available, else falls back to base ──
FINETUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fine_tuned_blip")
BASE_MODEL_NAME      = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    """Load BLIP processor and model (fine-tuned > base fallback)."""
    if os.path.isdir(FINETUNED_MODEL_PATH):
        print(f"[caption] Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
        processor = BlipProcessor.from_pretrained(FINETUNED_MODEL_PATH)
        model     = BlipForConditionalGeneration.from_pretrained(FINETUNED_MODEL_PATH).to(device)
    else:
        print(f"[caption] Fine-tuned model not found. Loading base model: {BASE_MODEL_NAME}")
        processor = BlipProcessor.from_pretrained(BASE_MODEL_NAME)
        model     = BlipForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(device)
    model.eval()
    return processor, model

# Load once at module import time
_processor, _model = _load_model()


def generate_caption(image: Image.Image) -> str:
    """
    Generate a short caption for the given PIL Image.

    Args:
        image: A PIL.Image.Image object (RGB).

    Returns:
        A string caption describing the image.
    """
    image = image.convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=5,
            early_stopping=True
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption
