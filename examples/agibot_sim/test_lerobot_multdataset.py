# test multi dataset support in lerobot
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

ds = MultiLeRobotDataset(
    repo_ids=[
        "bot-pi/agibot-sim-restock-supermarket-items", 
        "bot-pi/agibot-sim-stamp-the-seal"
    ],
    root='dataset'
)


print(f"Number of episodes: {len(ds)}")
print(ds[0])

print(ds[2000])