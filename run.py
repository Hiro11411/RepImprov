import fiftyone as fo

dataset = fo.Dataset(name="repimprov_workouts", overwrite=True)
dataset.add_videos_dir(
    r"C:\Users\zuick\OneDrive\Desktop\Hackathon\repimprov\workoutfitness-video"
)
dataset.compute_metadata()

print(f"Loaded {len(dataset)} videos")

session = fo.launch_app(dataset)
session.wait()