import fiftyone as fo
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

dataset = fo.Dataset(name="repimprov_workouts", overwrite=True)
dataset.add_videos_dir(
    r"C:\Users\zuick\OneDrive\Desktop\Hackathon\repimprov\workoutfitness-video"
)
dataset.compute_metadata()

print(f"Loaded {len(dataset)} videos")

session = fo.launch_app(dataset)
session.wait()