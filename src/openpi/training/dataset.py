import random 
import torch
import logging
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset as Dataset

class MultiLeRobotDataset(Dataset):

    def __getitem__(self, idx: int):
        """Get item from the dataset.
        Args:
            index (int): Index of the item to get.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        
        # Try to get the item, handle errors gracefully
        try:
            item = self._datasets[dataset_idx][idx - start_idx]
        except Exception as e:
            logging.warning(f"Error accessing item at index {idx} (dataset {self.repo_ids[dataset_idx]}, local index {idx - start_idx}): {e}")
            item = None
        
        # If item is None, randomly sample a new index and retry
        if item is None:
            random_idx = random.randint(0, len(self) - 1)
            logging.info(f"Retrying with random index {random_idx}")
            return self.__getitem__(random_idx)
        
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item