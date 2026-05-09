"""
feedback.py — Feedback & Scoring Module
Converts a similarity score (0–100) into a grade label,
emoji, colour, and detailed constructive feedback message.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FeedbackResult:
    score: float          # 0–100
    grade: str            # "Excellent" / "Good" / "Fair" / "Poor"
    emoji: str
    color: str            # hex colour for UI
    short_message: str    # one-liner
    detailed_feedback: str  # multi-sentence constructive feedback


# ── Thresholds (inclusive lower bound) ──────────────────────────────────────
_TIERS = [
    {
        "min": 85,
        "grade": "Excellent",
        "emoji": "🏆",
        "color": "#22c55e",   # green-500
        "short": "Outstanding description!",
        "detail": (
            "Your answer captures the scene with impressive accuracy and detail. "
            "The vocabulary, structure, and key visual elements closely match the AI-generated "
            "reference. You identified the main subjects, setting, and mood of the image. "
            "Keep up this level of descriptive precision — you are performing at an expert level."
        ),
    },
    {
        "min": 70,
        "grade": "Good",
        "emoji": "✅",
        "color": "#84cc16",   # lime-500
        "short": "Good answer, minor gaps remain.",
        "detail": (
            "Your description is solid and covers most of the important elements in the image. "
            "You correctly identified the main subjects and general setting. "
            "To improve further, try to include more specific details such as colours, textures, "
            "spatial relationships between objects, and the overall mood or atmosphere of the scene. "
            "Adding descriptive adjectives and action verbs will also strengthen your answer."
        ),
    },
    {
        "min": 50,
        "grade": "Fair",
        "emoji": "⚠️",
        "color": "#f59e0b",   # amber-500
        "short": "Needs more detail and depth.",
        "detail": (
            "Your answer shows a basic understanding of the image but misses several key aspects. "
            "Try to describe not just what is present, but also how things look — colours, sizes, "
            "positions, expressions, lighting conditions, and background details. "
            "A strong image description tells a story: mention who or what is in the scene, "
            "what they are doing, where they are, and what the overall feeling of the image conveys. "
            "Aim to write at least 3–5 detailed sentences."
        ),
    },
    {
        "min": 0,
        "grade": "Poor",
        "emoji": "❌",
        "color": "#ef4444",   # red-500
        "short": "Significant improvement needed.",
        "detail": (
            "Your description does not closely match the image content. "
            "It may be too short, too vague, or focused on the wrong elements. "
            "Start by identifying the main subject of the image, then describe its surroundings. "
            "Include colours, shapes, actions, and any people or animals present. "
            "Practice by observing everyday scenes and writing detailed descriptions — "
            "aim for at least 5–8 sentences that cover what you see from foreground to background."
        ),
    },
]


def generate_feedback(score: float) -> FeedbackResult:
    """
    Generate a structured feedback result based on a 0–100 score.

    Args:
        score: Percentage similarity score (0–100).

    Returns:
        A FeedbackResult dataclass instance.
    """
    for tier in _TIERS:
        if score >= tier["min"]:
            return FeedbackResult(
                score=score,
                grade=tier["grade"],
                emoji=tier["emoji"],
                color=tier["color"],
                short_message=tier["short"],
                detailed_feedback=tier["detail"],
            )
    # Fallback (should never reach here)
    return FeedbackResult(
        score=score,
        grade="Unknown",
        emoji="❓",
        color="#6b7280",
        short_message="Score could not be evaluated.",
        detailed_feedback="Please try again.",
    )


def score_breakdown(score: float) -> dict:
    """
    Return a dictionary with score details for display in the UI.
    """
    result = generate_feedback(score)
    return {
        "score":    result.score,
        "grade":    result.grade,
        "emoji":    result.emoji,
        "color":    result.color,
        "short":    result.short_message,
        "detail":   result.detailed_feedback,
    }
