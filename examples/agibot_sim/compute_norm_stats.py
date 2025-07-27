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





class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


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
