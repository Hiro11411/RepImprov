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

from .prompts import (
    FORM_ASSESSMENT_PROMPT, POSTURE_ANALYSIS_PROMPT, STRENGTHS_PROMPT,
    _AUTO_EXERCISE_CONTEXT, confirmed_exercise_context,
)
from .utils import parse_pegasus_response, compute_posture_score, frames_from_timestamp

# Load .env from the plugin directory — works regardless of where run.py is called from
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PLUGIN_DIR, ".env"))

logger = logging.getLogger(__name__)

EXERCISE_CHOICES  = ["auto", "squat", "deadlift", "bench_press", "pushup"]
SENSITIVITY_CHOICES = ["moderate", "strict", "lenient"]
PEGASUS_MODEL     = "pegasus1.2"


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
            description="Pre-filled from TWELVELABS_API_KEY env var",
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
        api_key       = ctx.params.get("api_key") or os.getenv("TWELVELABS_API_KEY", "")
        exercise_type = ctx.params.get("exercise_type", "auto")
        sensitivity   = ctx.params.get("sensitivity", "moderate")

        logger.info(
            "analyze_workout_form started — exercise=%s sensitivity=%s dataset=%s",
            exercise_type, sensitivity, ctx.dataset.name,
        )
        logger.info("API key present: %s", bool(api_key))

        if not api_key:
            logger.error("No TwelveLabs API key provided")
            return {"message": "ERROR: No TwelveLabs API key. Set TWELVELABS_API_KEY in repimprov/.env", "processed": 0, "errors": 0}

        try:
            client = TwelveLabs(api_key=api_key)
            logger.info("TwelveLabs client initialised")
        except Exception as exc:
            logger.exception("Failed to initialise TwelveLabs client: %s", exc)
            return {"message": f"ERROR: TwelveLabs client failed — {exc}", "processed": 0, "errors": 0}

        try:
            index = _get_or_create_index(client, "repimprov_analysis")
            logger.info("Using TwelveLabs index id=%s", index.id)
        except Exception as exc:
            logger.exception("Failed to get/create TwelveLabs index: %s", exc)
            return {"message": f"ERROR: Index setup failed — {exc}", "processed": 0, "errors": 0}

        processed = 0
        errors    = 0
        schema    = ctx.dataset.get_field_schema()
        has_score_field = "form_score" in schema

        for sample in ctx.dataset.iter_samples(autosave=True, progress=True):
            filepath = sample.filepath
            if not filepath:
                logger.warning("Sample %s has no filepath — skipping", sample.id)
                continue

            # Skip already-analyzed samples to avoid wasting API quota.
            # BUT re-analyze if score=50 + grade=C — that is the parse-failure default,
            # meaning Prompt A silently failed and never wrote a real result.
            if has_score_field:
                existing_score = sample.get_field("form_score")
                existing_grade = sample.get_field("form_grade") if "form_grade" in schema else None
                is_default_fail = (existing_score == 50.0 and existing_grade == "C")
                if existing_score is not None and not is_default_fail:
                    logger.debug("Sample %s already analyzed (score=%.1f) — skipping", sample.id, existing_score)
                    processed += 1
                    continue
                if is_default_fail:
                    logger.info("Sample %s has default score=50/C — re-analyzing", sample.id)

            logger.info("Processing sample %s: %s", sample.id, filepath)

            try:
                # ── Upload ────────────────────────────────────────────────────
                logger.debug("Uploading %s …", filepath)
                with open(filepath, "rb") as _vf:
                    task = client.tasks.create(index_id=index.id, video_file=_vf)
                logger.info("Task created: id=%s for sample %s", task.id, sample.id)

                done = client.tasks.wait_for_done(task_id=task.id, sleep_interval=5)
                logger.info("Task %s finished with status=%s", task.id, done.status)

                if done.status != "ready":
                    logger.error(
                        "Task %s not ready (status=%s) — marking sample %s as ERROR",
                        task.id, done.status, sample.id,
                    )
                    sample["form_grade"] = "ERROR"
                    errors += 1
                    continue

                video_id   = done.video_id
                frame_rate = getattr(sample.metadata, "frame_rate", None) or 30.0
                logger.debug("video_id=%s frame_rate=%.1f", video_id, frame_rate)

                # ── Prompt A: overall form assessment ─────────────────────────
                logger.debug("Running Prompt A (form assessment) for video %s", video_id)
                exercise_context = (
                    _AUTO_EXERCISE_CONTEXT if exercise_type == "auto"
                    else confirmed_exercise_context(exercise_type)
                )
                try:
                    raw_a = client.analyze(
                        video_id=video_id,
                        prompt=FORM_ASSESSMENT_PROMPT.format(
                            exercise_context=exercise_context, sensitivity=sensitivity
                        ),
                    ).data
                    logger.debug("Prompt A raw response (first 200): %s", str(raw_a)[:200])
                except Exception as exc:
                    logger.exception("Prompt A failed for video %s: %s", video_id, exc)
                    raw_a = ""

                data_a            = parse_pegasus_response(raw_a)
                form_score        = float(data_a.get("form_score", 50))
                form_grade        = str(data_a.get("form_grade", "C"))
                rep_count         = int(data_a.get("rep_count", 0))
                exercise_detected = str(data_a.get("exercise_detected", exercise_type))
                verdict           = str(data_a.get("verdict", "Analysis complete"))
                confidence        = float(data_a.get("confidence", 100))
                logger.info(
                    "Prompt A result — score=%.1f grade=%s reps=%d exercise=%s confidence=%.0f",
                    form_score, form_grade, rep_count, exercise_detected, confidence,
                )

                # ── Prompt B: posture & issue detection ───────────────────────
                logger.debug("Running Prompt B (posture analysis) for video %s", video_id)
                try:
                    raw_b = client.analyze(
                        video_id=video_id,
                        prompt=POSTURE_ANALYSIS_PROMPT.format(
                            exercise_type=exercise_detected, sensitivity=sensitivity
                        ),
                    ).data
                    logger.debug("Prompt B raw response (first 200): %s", str(raw_b)[:200])
                except Exception as exc:
                    logger.exception("Prompt B failed for video %s: %s", video_id, exc)
                    raw_b = ""

                data_b        = parse_pegasus_response(raw_b)
                issues        = data_b.get("issues", [])
                posture_score = compute_posture_score(issues)
                logger.info("Prompt B result — %d issues, posture_score=%.1f", len(issues), posture_score)

                temporal_detections = []
                for issue in issues:
                    try:
                        ts      = float(issue.get("timestamp_seconds", 0.0))
                        support = frames_from_timestamp(ts, frame_rate)
                        det     = fo.TemporalDetection(
                            label=str(issue.get("problem", "unknown_issue")),
                            support=support,
                        )
                        det["severity"] = str(issue.get("severity", "minor"))
                        det["fix"]      = str(issue.get("fix", ""))
                        temporal_detections.append(det)
                    except Exception as exc:
                        logger.warning("Skipping malformed issue %s: %s", issue, exc)

                # ── Prompt C: strengths ───────────────────────────────────────
                logger.debug("Running Prompt C (strengths) for video %s", video_id)
                try:
                    raw_c = client.analyze(
                        video_id=video_id,
                        prompt=STRENGTHS_PROMPT.format(
                            exercise_type=exercise_detected, sensitivity=sensitivity
                        ),
                    ).data
                    logger.debug("Prompt C raw response (first 200): %s", str(raw_c)[:200])
                except Exception as exc:
                    logger.exception("Prompt C failed for video %s: %s", video_id, exc)
                    raw_c = ""

                data_c           = parse_pegasus_response(raw_c)
                raw_strengths    = data_c.get("strengths", [])
                top_priority_fix = str(data_c.get("top_priority_fix", ""))
                coaching_summary = str(data_c.get("coaching_summary", ""))

                # Parse strengths — new format is list of dicts with timestamps,
                # old format was list of strings; handle both gracefully
                strength_texts       = []
                highlight_detections = []
                for s in (raw_strengths if isinstance(raw_strengths, list) else []):
                    try:
                        if isinstance(s, dict):
                            ts      = float(s.get("timestamp_seconds", 0.0))
                            label   = str(s.get("label", "good_form")).replace(" ", "_")
                            desc    = str(s.get("description", label))
                            support = frames_from_timestamp(ts, frame_rate)
                            det     = fo.TemporalDetection(label=label, support=support)
                            det["description"] = desc
                            det["type"]        = "strength"
                            highlight_detections.append(det)
                            strength_texts.append(desc)
                        else:
                            # plain string fallback
                            strength_texts.append(str(s))
                    except Exception as exc:
                        logger.warning("Skipping malformed strength %s: %s", s, exc)

                logger.info(
                    "Prompt C result — %d strengths (%d with timestamps), fix=%r",
                    len(strength_texts), len(highlight_detections), top_priority_fix[:60],
                )

                # ── Derive extra labels ───────────────────────────────────────
                severity_rank = {"critical": 3, "moderate": 2, "minor": 1}
                worst = max(
                    (issue.get("severity", "minor") for issue in issues),
                    key=lambda s: severity_rank.get(str(s).lower(), 0),
                    default="clean",
                ) if issues else "clean"

                # ── Write labels back ─────────────────────────────────────────
                sample["exercise_detected"] = exercise_detected
                sample["rep_count"]         = rep_count
                sample["form_score"]        = form_score
                sample["form_grade"]        = form_grade
                sample["posture_score"]     = posture_score
                sample["top_priority_fix"]  = top_priority_fix
                sample["coaching_summary"]  = coaching_summary
                sample["strengths"]         = strength_texts

                # ── FiftyOne Classification labels (colored tags in grid) ──────
                sample["form_classification"] = fo.Classification(
                    label=verdict,
                    confidence=form_score / 100.0,
                )
                sample["grade_label"] = fo.Classification(
                    label=form_grade,
                    confidence=form_score / 100.0,
                )
                sample["exercise_label"] = fo.Classification(
                    label=exercise_detected,
                    confidence=confidence / 100.0,
                )
                sample["worst_issue_severity"] = fo.Classification(
                    label=worst,
                )

                # ── TemporalDetections — bad form (red) ───────────────────────
                sample["form_issues"] = fo.TemporalDetections(
                    detections=temporal_detections
                )

                # ── TemporalDetections — good form highlights (green) ─────────
                sample["form_highlights"] = fo.TemporalDetections(
                    detections=highlight_detections
                )

                processed += 1
                logger.info(
                    "Sample %s done — %s score=%.1f grade=%s issues=%d highlights=%d",
                    sample.id, exercise_detected, form_score, form_grade,
                    len(temporal_detections), len(highlight_detections),
                )

            except Exception as exc:
                logger.exception("Unhandled error for sample %s (%s): %s", sample.id, filepath, exc)
                sample["form_grade"] = "ERROR"
                errors += 1

        logger.info(
            "analyze_workout_form complete — processed=%d errors=%d", processed, errors
        )
        return {
            "message": (
                f"RepImprov complete! Analyzed {processed} video(s). "
                f"{errors} error(s) encountered."
            ),
            "processed": processed,
            "errors":    errors,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("message", label="Result")
        outputs.int("processed", label="Videos Processed")
        outputs.int("errors",    label="Errors")
        return types.Property(outputs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_index(client, name: str):
    """Return an existing TwelveLabs index by name, or create a new one."""
    try:
        for index in client.indexes.list():
            if index.index_name == name:
                logger.info("Reusing existing index '%s' (id=%s)", name, index.id)
                return index
    except Exception as exc:
        logger.warning("Could not list TwelveLabs indexes: %s — will attempt to create", exc)

    logger.info("Creating new TwelveLabs index '%s' …", name)
    index = client.indexes.create(
        index_name=name,
        models=[{"model_name": PEGASUS_MODEL, "model_options": ["visual", "audio"]}],
    )
    logger.info("Index created: id=%s", index.id)
    return index


def register(plugin):
    plugin.register(AnalyzeWorkoutForm)
