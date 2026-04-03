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
    Returns a dict on success, empty dict on any failure.
    """
    if not raw_text:
        logger.warning("parse_pegasus_response: received empty/None response")
        return {}

    try:
        # Remove ```json ... ``` or ``` ... ``` fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        # Extract a JSON object if there is surrounding prose
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        else:
            logger.warning(
                "parse_pegasus_response: no JSON object found in response. "
                "First 200 chars: %s", raw_text[:200]
            )
            return {}

        result = json.loads(cleaned)
        logger.debug("parse_pegasus_response: parsed %d keys", len(result))
        return result

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "parse_pegasus_response: JSON decode failed — %s. "
            "Raw text (first 500 chars): %s", exc, raw_text[:500]
        )
        return {}
    except Exception as exc:
        logger.exception("parse_pegasus_response: unexpected error — %s", exc)
        return {}


def compute_posture_score(issues: list) -> float:
    """
    Calculate a posture score (0–100) from issue severities.
    Deductions: critical −15, moderate −8, minor −3. Clamped to [0, 100].
    """
    DEDUCTIONS = {"critical": 15, "moderate": 8, "minor": 3}

    if not isinstance(issues, list):
        logger.warning("compute_posture_score: expected list, got %s", type(issues))
        return 100.0

    score = 100.0
    try:
        for issue in issues:
            if not isinstance(issue, dict):
                logger.debug("compute_posture_score: skipping non-dict issue: %s", issue)
                continue
            severity = str(issue.get("severity", "minor")).lower().strip()
            deduction = DEDUCTIONS.get(severity, 3)
            if severity not in DEDUCTIONS:
                logger.debug(
                    "compute_posture_score: unknown severity %r, defaulting to −3", severity
                )
            score -= deduction

        result = float(max(0.0, min(100.0, score)))
        logger.debug(
            "compute_posture_score: %d issues → score %.1f", len(issues), result
        )
        return result

    except Exception as exc:
        logger.exception("compute_posture_score: unexpected error — %s", exc)
        return 100.0


def frames_from_timestamp(seconds: float, frame_rate: float) -> list:
    """
    Convert a timestamp (seconds) to [frame_start, frame_end] for TemporalDetection.
    Uses a ±0.5 s window. Falls back to 30 fps if frame_rate is invalid.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        logger.warning("frames_from_timestamp: invalid seconds %r — %s, defaulting to 0", seconds, exc)
        seconds = 0.0

    try:
        frame_rate = float(frame_rate)
    except (TypeError, ValueError) as exc:
        logger.warning("frames_from_timestamp: invalid frame_rate %r — %s, defaulting to 30", frame_rate, exc)
        frame_rate = 30.0

    if frame_rate <= 0:
        logger.warning("frames_from_timestamp: non-positive frame_rate %.2f, defaulting to 30", frame_rate)
        frame_rate = 30.0

    half = 0.5
    frame_start = max(0, int((seconds - half) * frame_rate))
    frame_end   = max(frame_start + 1, int((seconds + half) * frame_rate))

    logger.debug(
        "frames_from_timestamp: %.2fs @ %.1f fps → [%d, %d]",
        seconds, frame_rate, frame_start, frame_end,
    )
    return [frame_start, frame_end]
