# directly decode from torchcodec
from torchcodec.decoders import VideoDecoder

frame_idx = [246]

video_path = "dataset/bot-pi/agibot-simdata-sample/stamp_the_seal/videos/chunk-000/observation.images.hand_left_color/episode_000001.mp4"
decoder = VideoDecoder(video_path, device='cpu', seek_mode="approximate")

frames_batch = decoder.get_frames_at(indices=frame_idx)

print(f"frame index: {frame_idx}")
print(f"query timestamp: {[idx / 30 for idx in frame_idx]}")
print(f"actual timestamp: {frames_batch.pts_seconds.tolist()}")


# decode from lerobot decode function
from lerobot.datasets.video_utils import (decode_video_frames_torchcodec, 
                                          decode_video_frames_torchvision,
                                        decode_video_frames
    )
print('decoding using lerobot decode function: ------------------------\n')

decode_video_frames(
    video_path=video_path,
    timestamps=[idx / 30 for idx in frame_idx],
    tolerance_s=0.2,
    backend='pyav',)


print('decoding using torchvision: --------------------------\n')
frames_batch = decode_video_frames_torchvision(
    video_path=video_path,
    timestamps=[idx / 30 for idx in frame_idx],
    tolerance_s=0.2,
    backend='pyav',
    log_loaded_timestamps=True)

print("decoding using torchcodec: ------------------------\n")
frames_batch = decode_video_frames_torchcodec(
    video_path=video_path,
    timestamps=[idx / 30 for idx in frame_idx],
    tolerance_s=0.2,
    log_loaded_timestamps=True)


# decode from lerobot 
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('\n decoding using lerobot dataset: ------------------------\n')
dataset = LeRobotDataset(
    repo_id = 'agibot',
    root = 'dataset/bot-pi/agibot-simdata-sample/stamp_the_seal',
    episodes=[1],
    tolerance_s=0.2,
    video_backend = 'pyav')


for idx in frame_idx:
    print(f"Frame {idx}:")
    frame = dataset[idx]
    print(f"Timestamp: {frame['timestamp']}")
    