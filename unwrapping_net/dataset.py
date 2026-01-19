from torch.utils.data import Dataset
import torch
import numpy as np
import os


class OrientationDataset(Dataset):

    def __init__(
        self,
        wrapped_data_dir,
        unwrapped_data_dir,
        wrapped_data_list,
        unwrapped_data_list,
    ):
        self.unwrapped_data_files = unwrapped_data_list
        self.wrapped_data_files = wrapped_data_list

        self.wrapped_data_dir = wrapped_data_dir
        self.unwrapped_data_dir = unwrapped_data_dir

        assert len(self.unwrapped_data_files) == len(
            self.wrapped_data_files
        ), "Mismatch in number of files."

    def __len__(self) -> int:
        return len(self.wrapped_data_files)

    def __getitem__(self, idx: int) -> dict:
        wrapped = np.load(
            os.path.join(self.wrapped_data_dir, self.wrapped_data_files[idx])
        )
        unwrapped = np.load(
            os.path.join(self.unwrapped_data_dir, self.unwrapped_data_files[idx])
        )

        sample = {
            "wrapped": torch.tensor(wrapped, dtype=torch.float32).unsqueeze(
                0
            ),  # (1, H, W)
            "unwrapped": torch.tensor(unwrapped, dtype=torch.float32).unsqueeze(
                0
            ),  # (1, H, W)
        }

        return sample
