"""
Exercise identification via a dedicated Pegasus pre-prompt.

Asks a single focused question — "what exercise is this?" — before
the main analysis prompts. A narrow single-task prompt is more accurate
than embedding "auto" inside a larger scoring prompt.

Falls back to "auto" if the response is unrecognized or the call fails.
"""

import logging

logger = logging.getLogger(__name__)

IDENTIFY_PROMPT = """Watch this video and identify the exercise being performed.

Look at:
- The athlete's body orientation (standing, horizontal, lying on back, seated)
- Equipment present (barbell, bodyweight, bench, pull-up bar, dumbbells, cables)
- The primary movement pattern (knee bend, hip hinge, horizontal push, vertical pull, etc.)

Known exercises:
- squat: standing, descends by bending knees, hips below knees
- deadlift: standing, picks a barbell off the floor by hinging at hips
- bench_press: lying on back on a bench, pressing barbell upward
- pushup: face-down horizontal, pushing bodyweight up from the floor
- shoulder_press: standing or seated, pressing weight overhead
- pull_up: hanging from a bar, pulling body upward
- crunch: lying on back, upper body curls toward knees, core contraction

Respond with ONLY the exercise name from the list above — one word or two words, nothing else.
Examples of valid responses: squat | deadlift | bench_press | pushup | shoulder_press | pull_up | crunch
"""

KNOWN_EXERCISES = {"squat", "deadlift", "bench_press", "pushup", "shoulder_press", "pull_up", "crunch"}


def classify_exercise(client, video_id: str) -> str:
    """
    Use a dedicated Pegasus prompt to identify the exercise in an already-indexed video.

    Args:
        client: TwelveLabs client instance
        video_id: the indexed video ID

    Returns:
        exercise label string (e.g. "pushup") or "auto" if unrecognized/failed
    """
    try:
        raw = client.analyze(video_id=video_id, prompt=IDENTIFY_PROMPT).data
        if not raw:
            logger.warning("Empty response from exercise identification prompt")
            return "auto"

        label = raw.strip().lower().replace(" ", "_").replace("-", "_")

        # Strip any punctuation or extra words the model may have added
        for known in KNOWN_EXERCISES:
            if known in label:
                logger.info("Exercise identified: %s", known)
                return known

        logger.warning("Unrecognized exercise label from identification prompt: '%s'", raw.strip())
        return "auto"

    except Exception as exc:
        logger.warning("Exercise identification failed: %s — falling back to 'auto'", exc)
        return "auto"
