# test multi dataset support in lerobot
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

ds = MultiLeRobotDataset(
    repo_ids=[
        "bot-pi/agibot-sim-clear-table-in-the-restaurant",
        "bot-pi/agibot-sim-clear-the-countertop-waste",
        "bot-pi/agibot-sim-heat-the-food-in-the-microwave",
        "bot-pi/agibot-sim-make-a-sandwich",
        "bot-pi/agibot-sim-open-drawer-and-store-items",
        "bot-pi/agibot-sim-pack-in-the-supermarket",
        "bot-pi/agibot-sim-pack-moving-objects-from-conveyor",
        "bot-pi/agibot-sim-pickup-items-from-the-freezer",
        "bot-pi/agibot-sim-restock-supermarket-items", 
        "bot-pi/agibot-sim-stamp-the-seal"
    ],
    root='test'
)


print(f"Number of episodes: {len(ds)}")
