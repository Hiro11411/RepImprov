"""
RepImprov demo launcher.

Extracts workoutfitness-video.zip (if not already done), loads all 590
exercise videos into a FiftyOne dataset, and opens the App.

Run:
    python run.py
"""

import os
import zipfile
import fiftyone as fo

# ── Config ────────────────────────────────────────────────────────────────────
ZIP_PATH     = os.path.join(os.path.dirname(__file__), "workoutfitness-video.zip")
EXTRACT_DIR  = os.path.join(os.path.dirname(__file__), "workoutfitness-video")
DATASET_NAME = "repimprov_demo"

# ── Step 1: Extract zip (skip if already done) ────────────────────────────────
if not os.path.isdir(EXTRACT_DIR):
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to: {EXTRACT_DIR}")
else:
    print(f"Videos already extracted at: {EXTRACT_DIR}")

# ── Step 2: Collect all .mp4 paths ───────────────────────────────────────────
video_paths = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for fname in sorted(files):
        if fname.lower().endswith(".mp4"):
            video_paths.append(os.path.join(root, fname))

print(f"Found {len(video_paths)} videos across {len(os.listdir(EXTRACT_DIR))} exercise folders.")

# ── Step 3: Build FiftyOne dataset ───────────────────────────────────────────
if fo.dataset_exists(DATASET_NAME):
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"Loaded existing dataset '{DATASET_NAME}' ({len(dataset)} samples).")
else:
    print(f"Building dataset '{DATASET_NAME}' ...")
    samples = []
    for path in video_paths:
        # Derive exercise label from the parent folder name
        exercise_folder = os.path.basename(os.path.dirname(path))
        sample = fo.Sample(filepath=path)
        sample["exercise_folder"] = exercise_folder   # visible in the sidebar before analysis
        samples.append(sample)

    dataset = fo.Dataset(name=DATASET_NAME)
    dataset.add_samples(samples)
    dataset.compute_metadata()
    dataset.persistent = True
    print(f"Dataset ready: {len(dataset)} samples.")

# ── Step 4: Launch App ────────────────────────────────────────────────────────
print("\nLaunching FiftyOne App ...")
print("  Open the Operator Browser (` key) and run 'RepImprov: Analyze Workout Form'")
print("  Open the Panel Browser to view the RepImprov Dashboard\n")

session = fo.launch_app(dataset)
session.wait()
