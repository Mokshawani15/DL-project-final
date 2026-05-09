"""
similarity.py — Semantic Similarity Module
Converts two text strings into sentence embeddings using
sentence-transformers and computes their cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# Load the model once at import time
# all-MiniLM-L6-v2 is fast (~80 MB) and highly accurate for semantic similarity
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    """
    Convert a text string into a sentence embedding vector.

    Args:
        text: Input text string.

    Returns:
        A 1-D numpy array (384-dim for all-MiniLM-L6-v2).
    """
    embedding = _embed_model.encode(text, convert_to_numpy=True)
    return embedding


def compute_similarity(reference_text: str, user_text: str) -> float:
    """
    Compute the cosine similarity between the AI-generated reference paragraph
    and the user's descriptive answer.

    Args:
        reference_text: The AI-generated detailed paragraph.
        user_text:      The user's descriptive answer.

    Returns:
        A float in [0.0, 1.0] representing semantic similarity.
    """
    ref_vec  = get_embedding(reference_text).reshape(1, -1)
    user_vec = get_embedding(user_text).reshape(1, -1)

    similarity = float(sk_cosine(ref_vec, user_vec)[0][0])
    # Clamp to [0, 1] in case of floating-point edge cases
    return max(0.0, min(1.0, similarity))


def similarity_to_score(similarity: float) -> float:
    """
    Convert a cosine similarity value to a percentage score (0–100).

    Args:
        similarity: Cosine similarity in [0.0, 1.0].

    Returns:
        Score as a float in [0.0, 100.0].
    """
    return round(similarity * 100, 2)
