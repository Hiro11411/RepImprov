# RepImprov

AI-powered workout form analyzer — a FiftyOne plugin powered by TwelveLabs Pegasus 1.2.

Upload exercise videos, get structured coaching labels written directly into your FiftyOne dataset: form scores, posture breakdowns with timestamps, rep counts, grade letters, and actionable coaching cues.

---

## Features

| Label field            | Type                  | Description                                     |
|------------------------|-----------------------|-------------------------------------------------|
| `exercise_detected`    | `str`                 | Exercise identified by Pegasus                  |
| `rep_count`            | `int`                 | Number of complete reps in the video            |
| `form_score`           | `float`               | Overall form quality (0–100)                    |
| `form_grade`           | `str`                 | Letter grade A–F                                |
| `posture_score`        | `float`               | Posture quality computed from issue severity    |
| `top_priority_fix`     | `str`                 | Single most important coaching cue              |
| `strengths`            | `list[str]`           | What the athlete does well                      |
| `coaching_summary`     | `str`                 | 2–3 sentence coaching overview                  |
| `form_classification`  | `fo.Classification`   | Verdict label with confidence = form_score/100  |
| `form_issues`          | `fo.TemporalDetections` | Per-issue temporal detections with severity + fix |

---

## Quickstart

### 1. Install dependencies

```bash
pip install fiftyone twelvelabs python-dotenv
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your TwelveLabs API key
```

### 3. Install the plugin

```bash
# From the plugin directory parent:
fiftyone plugins download /path/to/repimprov

# Or symlink for local development:
fiftyone plugins symlink /path/to/repimprov
```

### 4. Launch FiftyOne with a test dataset

```python
import fiftyone as fo
import glob

# Point this at a folder of .mp4 workout videos
video_paths = glob.glob("/path/to/workout/videos/*.mp4")

dataset = fo.Dataset(name="workout_test", overwrite=True)
samples = [fo.Sample(filepath=p) for p in video_paths]
dataset.add_samples(samples)

# Compute metadata so frame_rate is available
dataset.compute_metadata()

session = fo.launch_app(dataset)
```

Then open the **Operator Browser** (`` ` `` key or the lightning bolt icon) and search for **RepImprov: Analyze Workout Form**.

### 5. View the dashboard

Open the **Panel Browser** and select **RepImprov Dashboard** to see aggregate metrics across all analyzed samples.

---

## Plugin Structure

```
repimprov/
├── fiftyone.yml      # Plugin manifest
├── __init__.py       # Registration entry point
├── operator.py       # analyze_workout_form operator
├── panel.py          # repimprov_dashboard panel
├── prompts.py        # Pegasus prompt templates
├── utils.py          # parse_pegasus_response, compute_posture_score, frames_from_timestamp
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## How It Works

1. **Upload** — each video is uploaded to a TwelveLabs index via `client.task.create()`.
2. **Index** — `task.wait_for_done()` polls until Pegasus finishes indexing.
3. **Generate** — three `client.generate.text()` calls run against Pegasus 1.2 with visual + audio modalities:
   - **Prompt A** — overall form score, grade, rep count, verdict
   - **Prompt B** — timestamped posture issues with severity and fix cues
   - **Prompt C** — strengths, top priority fix, coaching summary
4. **Label** — structured results are written back as FiftyOne fields and temporal detections.

---

## Sensitivity Levels

| Level    | Behavior                                              |
|----------|-------------------------------------------------------|
| strict   | Flags even minor deviations from textbook form        |
| moderate | Flags meaningful deviations affecting safety/performance |
| lenient  | Flags significant safety risks only                   |

---

## Supported Exercises

- `auto` — Pegasus detects the exercise automatically
- `squat`
- `deadlift`
- `bench_press`
- `pushup`

---

## License

MIT
