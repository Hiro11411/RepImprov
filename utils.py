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


FORM_ASSESSMENT_REQUIRED = {"form_score", "form_grade", "exercise_detected", "verdict"}
POSTURE_ANALYSIS_REQUIRED = {"issues"}
STRENGTHS_REQUIRED = {"strengths", "top_priority_fix", "coaching_summary"}

KNOWN_EXERCISES = {"squat", "deadlift", "bench_press", "pushup", "shoulder_press"}
VALID_GRADES = {"A", "B", "C", "D", "F"}
VALID_SEVERITIES = {"critical", "moderate", "minor"}


def validate_form_assessment(data: dict) -> dict:
    """
    Check that a form assessment response has all required fields with valid types.
    Logs a warning for each problem found. Returns the data unchanged.
    """
    missing = FORM_ASSESSMENT_REQUIRED - data.keys()
    if missing:
        logger.warning("Form assessment missing fields: %s", missing)

    score = data.get("form_score")
    if score is not None and not (0 <= float(score) <= 100):
        logger.warning("form_score out of range: %s", score)

    grade = data.get("form_grade")
    if grade is not None and str(grade).upper() not in VALID_GRADES:
        logger.warning("Unexpected form_grade value: %s", grade)

    exercise = data.get("exercise_detected")
    if exercise is not None and str(exercise).lower() not in KNOWN_EXERCISES:
        logger.warning("Unrecognized exercise_detected: %s", exercise)

    confidence = data.get("confidence")
    if confidence is not None and not (0 <= float(confidence) <= 100):
        logger.warning("confidence out of range: %s", confidence)

    return data


def validate_posture_analysis(data: dict, duration: float = None) -> dict:
    """
    Check that a posture analysis response has a valid issues array.
    Logs warnings for malformed issue entries. Returns the data unchanged.

    Args:
        data: parsed response dict
        duration: video duration in seconds — when provided, timestamps that
                  exceed it are flagged as out-of-bounds hallucinations.
    """
    if "issues" not in data:
        logger.warning("Posture analysis missing 'issues' key")
        return data

    if not isinstance(data["issues"], list):
        logger.warning("'issues' is not a list: %s", type(data["issues"]))
        return data

    timestamps = []
    for i, issue in enumerate(data["issues"]):
        for field in ("timestamp_seconds", "problem", "severity", "fix"):
            if field not in issue:
                logger.warning("Issue[%d] missing field '%s': %s", i, field, issue)
        severity = str(issue.get("severity", "")).lower()
        if severity and severity not in VALID_SEVERITIES:
            logger.warning("Issue[%d] has unexpected severity '%s'", i, severity)
        ts = issue.get("timestamp_seconds")
        if ts is not None:
            ts_f = float(ts)
            if duration is not None and duration > 0 and ts_f > duration:
                logger.warning(
                    "Issue[%d] timestamp %.1fs exceeds video duration %.1fs — possible hallucination",
                    i, ts_f, duration,
                )
            timestamps.append(round(ts_f, 1))

    # Clustering check: if >50% of issues share the same rounded timestamp, flag it
    if len(timestamps) >= 2:
        most_common_count = max(timestamps.count(t) for t in set(timestamps))
        if most_common_count / len(timestamps) > 0.5:
            logger.warning(
                "Possible timestamp hallucination: %d/%d issues share the same timestamp",
                most_common_count, len(timestamps),
            )

    return data


def validate_strengths(data: dict) -> dict:
    """
    Check that a strengths response has all required fields with valid types.
    Logs a warning for each problem found. Returns the data unchanged.
    """
    missing = STRENGTHS_REQUIRED - data.keys()
    if missing:
        logger.warning("Strengths response missing fields: %s", missing)

    strengths = data.get("strengths")
    if strengths is not None and not isinstance(strengths, list):
        logger.warning("'strengths' is not a list: %s", type(strengths))

    return data


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
