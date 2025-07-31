"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
import packaging

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def main(config_name: str='pi0_agibot_sim_lora', max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader = _data_loader.create_torch_data_loader(
        data_config=data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    tot_frames = len(data_loader._data_loader._data_loader.dataset)
    print(f"Total frames in dataset: {tot_frames}")
    tot_steps = tot_frames // config.batch_size + 1
    
    # Create the iterator once outside the loop
    data_iter = iter(data_loader)
    
    for i in tqdm.tqdm(range(tot_steps), desc="Loading dataset", total=tot_steps):
        batch = next(data_iter)


if __name__ == "__main__":
    tyro.cli(main)
