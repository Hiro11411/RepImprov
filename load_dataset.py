"""
Phase 1 — Dataset for RepImprov

Loads workout videos into FiftyOne, computes metadata,
initializes all required fields, and launches the app.
"""

import fiftyone as fo
from operator import eq

VIDEO_DIR = "workoutfitness-video"   # or absolute path

DATASET_NAME = "repimprov_dataset"


def main():
    """
    I cooked.
    """
    # ── Create or load dataset ──────────────────────────────────────────────
    if DATASET_NAME in fo.list_datasets():
        print(f"Loading existing dataset: {DATASET_NAME}")
        dataset = fo.load_dataset(DATASET_NAME)
    else:
        print(f"Creating dataset from: {VIDEO_DIR}")
        dataset = fo.Dataset.from_videos_dir(
            VIDEO_DIR,
            name=DATASET_NAME,
        )

    # ── Compute metadata (REQUIRED) ─────────────────────────────────────────
    print("Computing metadata...")
    dataset.compute_metadata(overwrite=False)

    # ── Add required fields (safe if rerun) ─────────────────────────────────
    def ensure_field(name, field_type, kwargs):
        if name not in dataset.get_field_schema():
            dataset.add_sample_field(name, field_type, kwargs)

    # Core outputs (matches your operator.py exactly)
    ensure_field("exercise_detected", fo.StringField)
    ensure_field("rep_count", fo.IntField)
    ensure_field("form_score", fo.FloatField)
    ensure_field("form_grade", fo.StringField)
    ensure_field("posture_score", fo.FloatField)
    ensure_field("top_priority_fix", fo.StringField)
    ensure_field("coaching_summary", fo.StringField)

    # Lists
    ensure_field("strengths", fo.ListField, subfield=fo.StringField)
    ensure_field(
        "form_classification",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Classification,
    )

    ensure_field(
        "form_issues",
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.TemporalDetections,
    )

    # ── Persist dataset ─────────────────────────────────────────────────────
    dataset.persistent = True
    dataset.save()

    # ── Sanity checks ───────────────────────────────────────────────────────
    print("\nDataset ready ✅")
    print(f"Samples: {len(dataset)}")

    sample = dataset.first()
    if sample:
        print("\nExample sample:")
        print("filepath:", sample.filepath)
        print("metadata:", sample.metadata)

    # ── Launch FiftyOne App ─────────────────────────────────────────────────
    print("\nLaunching FiftyOne App...")
    session = fo.launch_app(dataset)
    session.wait()


if name == "main":
    main()


