"""
Core FiftyOne operator: analyze_workout_form

Uploads exercise videos to TwelveLabs, runs Pegasus 1.2 analysis across
three structured prompts, and writes coaching labels back to the dataset.
"""

import os
import logging

import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone as fo
from dotenv import load_dotenv

from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task

from .prompts import (
    FORM_ASSESSMENT_PROMPT,
    POSTURE_ANALYSIS_PROMPT,
    STRENGTHS_PROMPT,
    REP_COUNT_PROMPT,
    _REP_COUNT_INSTRUCTIONS,
    confirmed_exercise_context,
    _AUTO_EXERCISE_CONTEXT,
)
from .utils import parse_pegasus_response, compute_posture_score, frames_from_timestamp, validate_form_assessment, validate_posture_analysis, validate_strengths
from .classifier import classify_exercise

load_dotenv()
logger = logging.getLogger(__name__)

EXERCISE_CHOICES = ["auto", "squat", "deadlift", "bench_press", "pushup"]
SENSITIVITY_CHOICES = ["moderate", "strict", "lenient"]
PEGASUS_MODEL = "pegasus1.2"
CONFIDENCE_THRESHOLD = 60  # skip posture + strengths analysis below this


class AnalyzeWorkoutForm(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_workout_form",
            label="RepImprov: Analyze Workout Form",
            description="Upload exercise videos to TwelveLabs Pegasus and write structured coaching labels back to the dataset.",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "api_key",
            label="TwelveLabs API Key",
            description="Your TwelveLabs API key (pre-filled from TWELVELABS_API_KEY env var)",
            default=os.getenv("TWELVELABS_API_KEY", ""),
            required=True,
        )

        exercise_dropdown = types.Dropdown(label="Exercise Type")
        for choice in EXERCISE_CHOICES:
            exercise_dropdown.add_choice(choice, label=choice.replace("_", " ").title())
        inputs.enum(
            "exercise_type",
            values=EXERCISE_CHOICES,
            label="Exercise Type",
            description="Select exercise or 'auto' to let Pegasus detect it",
            default="auto",
            view=exercise_dropdown,
        )

        sensitivity_dropdown = types.Dropdown(label="Analysis Sensitivity")
        for choice in SENSITIVITY_CHOICES:
            sensitivity_dropdown.add_choice(choice, label=choice.title())
        inputs.enum(
            "sensitivity",
            values=SENSITIVITY_CHOICES,
            label="Analysis Sensitivity",
            description="strict = flag every deviation | moderate = meaningful issues | lenient = safety risks only",
            default="moderate",
            view=sensitivity_dropdown,
        )

        return types.Property(inputs)

    def execute(self, ctx):

        api_key = ctx.params.get("api_key") or os.getenv("TWELVELABS_API_KEY", "")
        exercise_type = ctx.params.get("exercise_type", "auto")
        sensitivity = ctx.params.get("sensitivity", "moderate")

        if not api_key:
            return {"error": "No TwelveLabs API key provided. Set TWELVELABS_API_KEY in your .env file."}

        client = TwelveLabs(api_key=api_key)

        # Build a TwelveLabs index for this run (or reuse by name)
        index_name = "repimprov_analysis"
        index = _get_or_create_index(client, index_name)

        processed = 0
        errors = 0

        for sample in ctx.dataset.iter_samples(autosave=True, progress=True):
            filepath = sample.filepath
            if not filepath:
                logger.warning("Sample %s has no filepath, skipping.", sample.id)
                continue

            try:
                # ── Step 1: Upload video ─────────────────────────────────────
                logger.info("Uploading %s to TwelveLabs...", filepath)
                import time
                task = client.tasks.create(
                    index_id=index.id,
                    video_file=open(filepath, "rb"),
                )
                task_id = task.id
                while True:
                    task_status = client.tasks.retrieve(task_id)
                    if task_status.status in ("ready", "failed"):
                        break
                    time.sleep(5)

                if task_status.status != "ready":
                    logger.error("Task %s not ready (status=%s), skipping.", task_id, task_status.status)
                    sample["form_grade"] = "ERROR"
                    errors += 1
                    continue

                video_id = task_status.video_id
                frame_rate = getattr(sample.metadata, "frame_rate", None) or 30.0
                duration = getattr(sample.metadata, "duration", None)

                # ── Step 1b: Pre-identify exercise ───────────────────────────
                effective_exercise = exercise_type
                if exercise_type == "auto":
                    identified = classify_exercise(client, video_id)
                    if identified != "auto":
                        logger.info("Exercise identified as '%s' for %s", identified, filepath)
                        effective_exercise = identified
                        sample["classifier_exercise"] = identified

                # Build exercise context: confirmed fact or auto-detect guidance
                if effective_exercise == "auto":
                    exercise_context = _AUTO_EXERCISE_CONTEXT
                else:
                    exercise_context = confirmed_exercise_context(effective_exercise)

                # ── Step 2: Prompt A – overall form assessment ───────────────
                prompt_a = FORM_ASSESSMENT_PROMPT.format(
                    exercise_context=exercise_context,
                    sensitivity=sensitivity,
                )
                raw_a = client.analyze(video_id=video_id, prompt=prompt_a).data

                data_a = validate_form_assessment(parse_pegasus_response(raw_a))
                form_score = float(data_a.get("form_score", 50))
                form_grade = str(data_a.get("form_grade", "C"))
                # Use confirmed exercise if known; fall back to what Prompt A reported
                exercise_detected = (
                    effective_exercise
                    if effective_exercise != "auto"
                    else str(data_a.get("exercise_detected", "unknown"))
                )
                verdict = str(data_a.get("verdict", "Analysis complete"))
                confidence = float(data_a.get("confidence", 100))

                # ── Step 3: Prompt D – dedicated rep count ───────────────────
                rep_count = 0
                if exercise_detected in _REP_COUNT_INSTRUCTIONS:
                    prompt_d = REP_COUNT_PROMPT.format(
                        exercise=exercise_detected,
                        instructions=_REP_COUNT_INSTRUCTIONS[exercise_detected],
                    )
                    raw_d = client.analyze(video_id=video_id, prompt=prompt_d).data
                    data_d = parse_pegasus_response(raw_d)
                    rep_count = int(data_d.get("rep_count", 0))
                    logger.info("Rep count for %s: %d (confidence=%s)", exercise_detected, rep_count, data_d.get("confidence"))

                # ── Gate B + C on confidence ─────────────────────────────────
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.warning(
                        "Skipping posture/strength analysis for %s: confidence=%.0f < %d",
                        filepath, confidence, CONFIDENCE_THRESHOLD,
                    )
                    issues = []
                    posture_score = 100.0
                    temporal_detections = []
                    strengths = []
                    top_priority_fix = ""
                    coaching_summary = (
                        f"Exercise identification confidence too low ({confidence:.0f}%) "
                        "for detailed analysis. Please re-run with the exercise type specified."
                    )
                else:
                    # ── Step 4: Prompt B – posture & issue detection ─────────
                    prompt_b = POSTURE_ANALYSIS_PROMPT.format(
                        exercise_type=exercise_detected,
                        sensitivity=sensitivity,
                    )
                    raw_b = client.analyze(video_id=video_id, prompt=prompt_b).data

                    data_b = validate_posture_analysis(parse_pegasus_response(raw_b), duration=duration)
                    issues = data_b.get("issues", [])
                    posture_score = compute_posture_score(issues)

                    # Build TemporalDetections from issues
                    temporal_detections = []
                    for issue in issues:
                        ts = float(issue.get("timestamp_seconds", 0.0))
                        support = frames_from_timestamp(ts, frame_rate)
                        det = fo.TemporalDetection(
                            label=str(issue.get("problem", "unknown_issue")),
                            support=support,
                        )
                        det["severity"] = str(issue.get("severity", "minor"))
                        det["fix"] = str(issue.get("fix", ""))
                        temporal_detections.append(det)

                    # ── Step 5: Prompt C – strengths analysis ─────────────────
                    prompt_c = STRENGTHS_PROMPT.format(
                        exercise_type=exercise_detected,
                        sensitivity=sensitivity,
                    )
                    raw_c = client.analyze(video_id=video_id, prompt=prompt_c).data

                    data_c = validate_strengths(parse_pegasus_response(raw_c))
                    strengths = data_c.get("strengths", [])
                    top_priority_fix = str(data_c.get("top_priority_fix", ""))
                    coaching_summary = str(data_c.get("coaching_summary", ""))

                # ── Step 6: Write labels back to sample ──────────────────────
                sample["exercise_detected"] = exercise_detected
                sample["rep_count"] = rep_count
                sample["form_score"] = form_score
                sample["form_grade"] = form_grade
                sample["posture_score"] = posture_score
                sample["top_priority_fix"] = top_priority_fix
                sample["strengths"] = strengths if isinstance(strengths, list) else list(strengths)
                sample["coaching_summary"] = coaching_summary
                sample["form_classification"] = fo.Classification(
                    label=verdict,
                    confidence=form_score / 100.0,
                )
                if temporal_detections:
                    sample["form_issues"] = fo.TemporalDetections(
                        detections=temporal_detections
                    )
                else:
                    sample["form_issues"] = fo.TemporalDetections(detections=[])

                processed += 1
                logger.info(
                    "Sample %s: %s | score=%.1f | grade=%s | reps=%d | confidence=%.0f | issues=%d",
                    sample.id, exercise_detected, form_score, form_grade, rep_count, confidence, len(issues),
                )

            except Exception as exc:
                logger.exception("Error processing sample %s: %s", sample.id, exc)
                sample["form_grade"] = "ERROR"
                errors += 1

        return {
            "message": (
                f"RepImprov complete! Analyzed {processed} video(s). "
                f"{errors} error(s) encountered."
            ),
            "processed": processed,
            "errors": errors,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("message", label="Result")
        outputs.int("processed", label="Videos Processed")
        outputs.int("errors", label="Errors")
        return types.Property(outputs)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_create_index(client, name: str):
    """Return an existing TwelveLabs index by name, or create a new one."""
    try:
        for index in client.indexes.list():
            if index.index_name == name:
                logger.info("Reusing existing TwelveLabs index '%s' (id=%s)", name, index.id)
                return index
    except Exception:
        pass

    logger.info("Creating new TwelveLabs index '%s'...", name)
    return client.indexes.create(
        index_name=name,
        models=[
            {
                "model_name": PEGASUS_MODEL,
                "model_options": ["visual", "audio"],
            }
        ],
    )


def register(plugin):
    plugin.register(AnalyzeWorkoutForm)
