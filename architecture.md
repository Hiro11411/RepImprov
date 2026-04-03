RepImprov 💪
AI-Powered Workout Coaching Intelligence
We are building an AI-Powered Workout Coaching Intelligence by Posture/Form Analysis Plugin

Team Members: Hiroaki Okumura, Hutch Turner, Laxmi Balcha, Ethan Lee

Video Understanding AI Hackathon at Northeastern University

Built on FiftyOne × TwelveLabs Pegasus
Dataset: https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video


Overview :
RepImprov is a FiftyOne plugin that transforms raw workout footage into structured, actionable coaching intelligence. By combining FiftyOne's dataset management and visualization capabilities with TwelveLabs' Pegasus video language model, RepImprov automatically analyzes exercise form, flags posture issues, scores performance, and surfaces comparative accuracy metrics across sessions — all browsable inside the FiftyOne App.

Ethan (Dataset loading + TwelveLabs integration)
Load workout videos into FiftyOne
Set up TwelveLabs index
Upload videos and poll for completion
Own operator.py upload section

 Hutch (Pegasus prompts + JSON parsing)
Write all 3 prompts in prompts.py
Write utils.py (parse_pegasus_response, compute_posture_score)
Test that Pegasus returns clean JSON
Own prompts.py + utils.py

Laxmi (FiftyOne label writing + filtering)
Take parsed JSON and write all fields back to samples
fo.Classification, fo.TemporalDetections setup
Test that labels show up correctly in the app
Own the bottom half of operator.py
—----
Here’s a clean, working implementation for the bottom half of operator.py that covers:
JSON → FiftyOne fields
Label creation (Classification + TemporalDetections)
Writing to samples
Filtering utilities
Debug-friendly structure
You can adapt field names to your schema, but this is production-ready scaffolding.
Assumptions 
{
  "id": "sample_001",
  "filepath": "/path/video.mp4",
  "classification": {
    "label": "event_a",
    "confidence": 0.92
  },
  "temporal_detections": [
    {
      "label": "action_1",
      "start": 1.2,
      "end": 3.4,
      "confidence": 0.88
    }
  ],
  "metadata": {
    "source": "model_v1"
  }
}

import fiftyone as fo
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# JSON → FiftyOne Label Builders
# -----------------------------

def build_classification(classification_json):
    if not classification_json:
        return None

    try:
        return fo.Classification(
            label=classification_json.get("label"),
            confidence=classification_json.get("confidence", None),
        )
    except Exception as e:
        logger.error(f"Failed to build classification: {e}")
        return None


def build_temporal_detections(detections_json):
    if not detections_json:
        return None

    detections = []

    for det in detections_json:
        try:
            detection = fo.TemporalDetection(
                label=det.get("label"),
                support=[det.get("start"), det.get("end")],
                confidence=det.get("confidence", None),
            )
            detections.append(detection)
        except Exception as e:
            logger.warning(f"Skipping bad detection: {det} | error: {e}")

    return fo.TemporalDetections(detections=detections)


# -----------------------------
# Attach Data to Sample
# -----------------------------

def write_labels_to_sample(sample, json_data):
    """
    Writes all fields + labels back to a FiftyOne sample
    """

    # --- Write raw fields ---
    for key, value in json_data.items():
        if key not in ["classification", "temporal_detections"]:
            try:
                sample[key] = value
            except Exception as e:
                logger.warning(f"Failed to write field {key}: {e}")

    # --- Classification ---
    classification = build_classification(json_data.get("classification"))
    if classification:
        sample["classification"] = classification

    # --- Temporal detections ---
    temporal = build_temporal_detections(
        json_data.get("temporal_detections")
    )
    if temporal:
        sample["temporal_detections"] = temporal

    sample.save()


# -----------------------------
# Bulk Ingestion
# -----------------------------

def ingest_json_list(dataset, json_list):
    """
    Takes list of parsed JSON objects and writes into dataset
    """

    for item in json_list:
        try:
            sample = fo.Sample(filepath=item["filepath"])
            write_labels_to_sample(sample, item)
            dataset.add_sample(sample)
        except Exception as e:
            logger.error(f"Failed to ingest sample: {item.get('id')} | {e}")


# -----------------------------
# Filtering Utilities
# -----------------------------

def filter_by_classification(dataset, label):
    return dataset.match(
        fo.ViewField("classification.label") == label
    )


def filter_by_confidence(dataset, threshold=0.5):
    return dataset.match(
        fo.ViewField("classification.confidence") >= threshold
    )


def filter_temporal_by_label(dataset, label):
    return dataset.filter_labels(
        "temporal_detections",
        fo.ViewField("label") == label
    )


def filter_temporal_by_time(dataset, start, end):
    return dataset.filter_labels(
        "temporal_detections",
        (fo.ViewField("support")[0] >= start) &
        (fo.ViewField("support")[1] <= end)
    )


# -----------------------------
# Debug / Validation Helpers
# -----------------------------

def validate_dataset(dataset):
    print(f"Total samples: {len(dataset)}")

    sample = dataset.first()
    if not sample:
        print("Dataset empty")
        return

    print("\nSample fields:")
    print(sample.field_names)

    print("\nClassification:")
    print(sample.get("classification", None))

    print("\nTemporal detections:")
    print(sample.get("temporal_detections", None))


# -----------------------------
# Launch App for QA
# -----------------------------

def launch_app(dataset):
    session = fo.launch_app(dataset)
    logger.info("FiftyOne app launched")
    return session


Person 4 Hiro(Panel + demo prep)
Build repimprov_dashboard panel
Load the demo dataset (download Kaggle videos)
Make sure the app launches cleanly
Prepare the demo script for judging
Own panel.py + README
System Architecture
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                           │
│   FiftyOne App (localhost:5151)                             │
│   ┌─────────────────┐    ┌──────────────────────────────┐  │
│   │  Sample Grid    │    │     RepSight Panel           │  │
│   │  - Video clips  │    │  - Form score chart          │  │
│   │  - Form labels  │    │  - Accuracy % across videos  │  │
│   │  - Issue tags   │    │  - Progress over time        │  │
│   │  - Grade badge  │    │  - Top priority fixes        │  │
│   └─────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │  FiftyOne Plugin API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      PLUGIN LAYER                           │
│                                                             │
│   Operator: analyze_workout_form                            │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  1. Resolve Input (exercise type, sensitivity)       │  │
│   │  2. Iterate over dataset samples                     │  │
│   │  3. Upload video → TwelveLabs                        │  │
│   │  4. Poll for indexing completion                     │  │
│   │  5. Send coaching prompt → Pegasus                   │  │
│   │  6. Parse JSON response                              │  │
│   │  7. Write labels back to FiftyOne sample             │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   Panel: repimprov_dashboard                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  - Renders form score distribution chart             │  │
│   │  - Accuracy % comparison across all analyzed videos  │  │
│   │  - Per-exercise breakdown                            │  │
│   │  - Most common issues across dataset                 │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │  TwelveLabs Python SDK
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TWELVELABS LAYER                         │
│                                                             │
│   Model: Pegasus 1.2                                        │
│   Modalities: Visual + Audio                                │
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │   Upload     │    │    Index     │    │   Analyze   │  │
│   │   /tasks     │───▶│   /indexes  │───▶│  /generate  │  │
│   │              │    │              │    │    .text    │  │
│   └──────────────┘    └──────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│                                                             │
│   Input: Video files (mp4, mov, avi)                        │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  workout_videos/                                     │  │
│   │  ├── squat_session_1.mp4                             │  │
│   │  ├── deadlift_session_1.mp4                          │  │
│   │  ├── bench_session_1.mp4                             │  │
│   │  └── ...                                             │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   Output: Labeled FiftyOne Dataset                          │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  sample fields:                                      │  │
│   │  - exercise_detected    (str)                        │  │
│   │  - rep_count            (int)                        │  │
│   │  - form_score           (float 0-100)                │  │
│   │  - form_grade           (str A-F)                    │  │
│   │  - posture_score        (float 0-100)                │  │
│   │  - form_classification  (fo.Classification)          │  │
│   │  - form_issues          (fo.TemporalDetections)      │  │
│   │  - strengths            (list[str])                  │  │
│   │  - top_priority_fix     (str)                        │  │
│   │  - coaching_summary     (str)                        │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘


Workflow Documentation
Phase 1 — Dataset Preparation
Goal: Load workout videos into FiftyOne and compute metadata.
User provides video folder
        │
        ▼
fo.Dataset.from_videos_dir("/path/to/workouts")
        │
        ▼
dataset.compute_metadata()   ← required for frame_rate, duration
        │
        ▼
fo.launch_app(dataset)       ← verify videos load correctly

Expected input:
Short workout clips (15 seconds – 5 minutes)
Single exercise per clip recommended for cleaner analysis
Side or front angle preferred for posture accuracy
MP4 format recommended

Phase 2 — Form Analysis (Core Operator)
Goal: Upload each video to TwelveLabs, run Pegasus analysis, write structured labels back.
For each sample in dataset:
        │
        ├── 1. Upload video → TwelveLabs /tasks endpoint
        │
        ├── 2. Poll task status until "ready"
        │
        ├── 3. Send Prompt A → FORM ASSESSMENT
        │         "Score overall form, count reps, grade performance"
        │         Returns: form_score, grade, rep_count, verdict
        │
        ├── 4. Send Prompt B → POSTURE & ISSUE DETECTION  
        │         "Identify specific posture breakdowns with timestamps"
        │         Returns: issues[], timestamp_seconds, severity, fix
        │
        ├── 5. Send Prompt C → STRENGTHS ANALYSIS
        │         "What is the athlete doing well?"
        │         Returns: strengths[], coaching_summary
        │
        └── 6. Write all fields back to sample → autosave

Prompt Strategy:
Prompt
Purpose
Output Fields
Form Assessment
Overall scoring
form_score, form_grade, rep_count, verdict
Posture Analysis
Breakdown detection
form_issues (TemporalDetections), severity
Strengths
Positive coaching
strengths[], top_priority_fix, coaching_summary


Phase 3 — Posture Analysis Detail
Goal: Surface specific biomechanical issues as temporal labels viewable on the video timeline.
Each detected issue becomes a fo.TemporalDetection:
Issue detected at timestamp T seconds
        │
        ▼
frame_start = T × frame_rate
frame_end   = (T + 2) × frame_rate   ← 2 second window
        │
        ▼
fo.TemporalDetection(
    label    = "knee_cave",
    support  = [frame_start, frame_end],
    severity = "critical",
    fix      = "Push knees outward in line with toes"
)

Exercise-specific posture checks:
Exercise
Critical Checks
Squat
Knee cave, forward lean, depth, heel rise, back rounding
Deadlift
Lower back rounding, bar drift, hip hinge pattern, lockout
Bench Press
Elbow flare >45°, wrist alignment, bar path deviation
Push Up
Hip sag, elbow position, head forward, insufficient depth
Shoulder Press
Lower back arch, elbow position, bar path


Phase 4 — Accuracy & Comparison Panel
Goal: Render a custom FiftyOne panel showing comparative form accuracy across all analyzed videos.
Metrics computed across dataset:
# Overall dataset accuracy
avg_form_score = dataset.mean("form_score")

# Per-exercise breakdown
squat_avg    = dataset.match(F("exercise_detected") == "squat").mean("form_score")
deadlift_avg = dataset.match(F("exercise_detected") == "deadlift").mean("form_score")

# Most common issues
all_issues = dataset.values("form_issues.detections.label", unwind=True)
issue_freq = Counter(all_issues)

# Progress over time (if sessions are tagged with date)
# Compare form_score across chronological samples

Panel displays:
📊 Form score distribution histogram across all videos
🏆 Accuracy % per exercise type (squat vs deadlift vs bench)
📈 Progress trend if multiple sessions exist
⚠️ Top 5 most frequent issues across entire dataset
✅ Most common strengths

Phase 5 — Filtering & Review in FiftyOne
Goal: Use FiftyOne's built-in filtering to create targeted review views.
from fiftyone import ViewField as F

# Review worst performers first
needs_work = dataset.match(F("form_score") < 60).sort_by("form_score")

# Find all critical issues across dataset
critical = dataset.match(
    F("form_issues.detections").filter(
        F("severity") == "critical"
    ).length() > 0
)

# Compare squats only
squats = dataset.match(F("exercise_detected") == "squat")

# Find best sessions to use as reference
best = dataset.match(F("form_grade").is_in(["A", "B"]))

# Find sessions with knee cave specifically
knee_cave = dataset.filter_labels(
    "form_issues",
    F("label") == "knee_cave"
)


File Structure
workout-form-analyzer/
├── fiftyone.yml          ← plugin manifest
├── __init__.py           ← registers operators + panel
├── operator.py           ← AnalyzeWorkoutForm operator
├── panel.py              ← RepSight dashboard panel
├── prompts.py            ← all Pegasus prompt templates
├── utils.py              ← JSON parsing, score calculation
├── .env                  ← TWELVELABS_API_KEY (never commit)
├── .env.example          ← template for teammates
├── .gitignore
├── requirements.txt
└── README.md


Data Flow Summary
Raw Video Files
      │
      ▼
FiftyOne Dataset (unlabeled)
      │
      ▼  [RepSight Operator]
      │
      ├── TwelveLabs Upload
      ├── Pegasus Analysis (Visual + Audio)
      └── JSON Response Parsing
      │
      ▼
FiftyOne Dataset (labeled)
      │
      ├── form_score, form_grade, rep_count
      ├── form_classification (fo.Classification)
      ├── form_issues (fo.TemporalDetections)
      └── coaching_summary, strengths, top_priority_fix
      │
      ▼
FiftyOne App
      │
      ├── Sample Grid → browse all videos with labels
      ├── Timeline View → click timestamps for issues
      ├── Filters → sort by score, exercise, severity
      └── RepSight Panel → accuracy % comparison dashboard


Requirements
fiftyone>=0.21
twelvelabs>=0.3
python-dotenv


Environment Variables
# .env
TWELVELABS_API_KEY=your_key_here


Key Design Decisions
Decision
Choice
Reason
TwelveLabs model
Pegasus 1.2
Generative reasoning, not search
Modalities
Visual + Audio
Audio captures breathing, verbal cues
Video mode
Recorded only
Reliability for demo, no stream latency
Output format
JSON from Pegasus
Structured, parseable, consistent
Label type for issues
TemporalDetections
Clickable on video timeline in FiftyOne
Label type for verdict
Classification
Enables dataset-level filtering
Prompt strategy
3 focused prompts
Cleaner JSON, less hallucination


