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

    dataset = _data_loader.create_torch_dataset(
        data_config=data_config, 
        action_horizon=config.model.action_horizon, 
        model_config=config.model
    )

    # make sure dataset version is >= 2.1
    #assert dataset.meta._version >= packaging.version.parse("v2.1"), \
    #    f"Dataset version {dataset.meta._version} is not supported. Please use a dataset with version >= 2.1."
        
    norm_stats = dataset.stats 
    
    # key mapping from agibot dataset to openpi interface
    keys={
        "observation.state": "state",
        "action": "actions"
    }

    # convert keys
    for key, new_key in keys.items():
        norm_stats.update({new_key: norm_stats[key]})
        norm_stats.pop(key, None)
        
        # pad dimensions to model action_dim
        # if action_dim is greater than the dimension of state or actions, pad with zeros
        # else, truncate to action_dim
        for k, v in norm_stats[new_key].items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                norm_stats[new_key][k] = transforms.pad_to_dim(v, config.model.action_dim)[:config.model.action_dim]
    
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
