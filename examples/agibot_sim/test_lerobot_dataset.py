from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset


# test load data from local dir
ds = LeRobotDataset(
    repo_id='libero',
    root='/home/yyf/github/openpi/dataset/bot-pi/agibot-simdata-sample/stamp_the_seal',
    #episodes=[0, 1, 2]
    )

for i in range(10):
    sample = ds[i]
    print(sample)
    print(f"Sample {i}:")
    print(f"  Action: {sample['action'].shape}")
    print(f"  State: {sample['observation.state'].shape}")
    print(f"  Timestamp: {sample['timestamp']}")
    print()