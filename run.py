"""
RepImprov demo launcher.

Extracts workoutfitness-video.zip → workoutfitness-video/, builds the
FiftyOne dataset with exercise_folder tags, and opens the App.

Run from anywhere:
    python repimprov/run.py
    python run.py          (if you're already inside repimprov/)
"""

import os
import zipfile
import fiftyone as fo

# ── Paths ─────────────────────────────────────────────────────────────────────
# repimprov/ sits one level below Hackathon/ where the zip lives
_PLUGIN_DIR  = os.path.dirname(os.path.abspath(__file__))
_HACKATHON   = os.path.dirname(_PLUGIN_DIR)          # Hackathon/

ZIP_PATH     = os.path.join(_HACKATHON, "workoutfitness-video.zip")
EXTRACT_DIR  = os.path.join(_HACKATHON, "workoutfitness-video")
DATASET_NAME = "repimprov_demo"

# ── Step 1: Extract zip (skipped if already done) ─────────────────────────────
if not os.path.isdir(EXTRACT_DIR):
    if not os.path.isfile(ZIP_PATH):
        raise FileNotFoundError(
            f"Could not find {ZIP_PATH}\n"
            "Place workoutfitness-video.zip in the Hackathon/ folder next to repimprov/."
        )
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to: {EXTRACT_DIR}")
else:
    print(f"Videos already extracted at: {EXTRACT_DIR}")

# ── Step 2: Collect .mp4 paths ───────────────────────────────────────────────
video_paths = []
for root, _dirs, files in os.walk(EXTRACT_DIR):
    for fname in sorted(files):
        if fname.lower().endswith(".mp4"):
            video_paths.append(os.path.join(root, fname))

exercise_dirs = [
    d for d in os.listdir(EXTRACT_DIR)
    if os.path.isdir(os.path.join(EXTRACT_DIR, d))
]
print(f"Found {len(video_paths)} videos across {len(exercise_dirs)} exercise categories.")

# ── Step 3: Build or reload FiftyOne dataset ──────────────────────────────────
if fo.dataset_exists(DATASET_NAME):
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"Loaded existing dataset '{DATASET_NAME}' ({len(dataset)} samples).")
else:
    print(f"Building dataset '{DATASET_NAME}' ...")
    samples = []
    for path in video_paths:
        # Parent folder = exercise category (e.g. "squat", "bench press")
        exercise_folder = os.path.basename(os.path.dirname(path))
        sample = fo.Sample(filepath=path)
        sample["exercise_folder"] = exercise_folder
        samples.append(sample)

    dataset = fo.Dataset(name=DATASET_NAME)
    dataset.add_samples(samples)
    dataset.compute_metadata()
    dataset.persistent = True
    print(f"Dataset ready: {len(dataset)} samples.")

# ── Step 4: Launch App ────────────────────────────────────────────────────────
print("\nLaunching FiftyOne App ...")
print("  Press ` to open the Operator Browser → run 'RepImprov: Analyze Workout Form'")
print("  Open the Panel Browser → 'RepImprov Dashboard' to see results\n")

session = fo.launch_app(dataset)
session.wait()
