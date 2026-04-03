"""
Utility helpers for RepImprov plugin.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_pegasus_response(raw_text: str) -> dict:
    """
    Strip markdown code fences from a Pegasus text response and parse as JSON.
    Returns a dict, or an empty dict on any parse failure.
    """
    if not raw_text:
        return {}

    # Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Attempt to extract a JSON object if there is surrounding text
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse Pegasus JSON response: %s\nRaw text: %s", exc, raw_text[:500])
        return {}


def compute_posture_score(issues: list) -> float:
    """
    Calculate a posture score (0-100) based on the severity of detected issues.

    Deductions:
      critical  → -15 points each
      moderate  → -8 points each
      minor     → -3 points each

    The score is clamped to [0, 100].
    """
    DEDUCTIONS = {
        "critical": 15,
        "moderate": 8,
        "minor": 3,
    }

    score = 100.0
    for issue in issues:
        severity = str(issue.get("severity", "minor")).lower()
        score -= DEDUCTIONS.get(severity, 3)

    return float(max(0.0, min(100.0, score)))


def frames_from_timestamp(seconds: float, frame_rate: float) -> list:
    """
    Convert a timestamp in seconds to a [frame_start, frame_end] pair
    suitable for fo.TemporalDetection support.

    Uses a ±0.5 second window around the timestamp so the detection
    spans roughly one second.
    """
    if frame_rate <= 0:
        frame_rate = 30.0

    half_window = 0.5  # seconds on each side of the timestamp
    frame_start = max(0, int((seconds - half_window) * frame_rate))
    frame_end = max(frame_start + 1, int((seconds + half_window) * frame_rate))
    return [frame_start, frame_end]
