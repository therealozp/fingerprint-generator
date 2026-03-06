import os
from random import random
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class FingerprintOrientationDataset(Dataset):
    def __init__(
        self,
        orientation_paths,
        minutiae_paths,
        frequency_paths,
        cos_cont_paths,
        cos_full_paths,
        sin_cont_paths,
        sin_full_paths,
        img_size=(256, 256),
    ):
        self.orientation_paths = orientation_paths
        self.minutiae_paths = minutiae_paths
        self.frequency_paths = frequency_paths

        self.cos_cont_paths = cos_cont_paths
        self.cos_full_paths = cos_full_paths
        self.sin_cont_paths = sin_cont_paths
        self.sin_full_paths = sin_full_paths

        self.img_size = img_size

    def __len__(self):
        return len(self.orientation_paths)

    def _generate_heatmap(self, minutiae, h, w, sigma=3):
        """
        Converts list of (x, y) points into a Gaussian heatmap.
        """
        heatmap = torch.zeros((h, w))

        # Create a coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        for x, y, _ in minutiae:
            if 0 <= x < w and 0 <= y < h:
                dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
                heatmap += torch.exp(-dist_sq / (2 * sigma**2))

        # Normalize heatmap to [0, 1] for stability
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.unsqueeze(0)  # Add channel dim -> (1, H, W)

    def _get_spiral_phase(self, minutiae_list, h, w):
        spiral_phase = np.zeros((h, w), dtype=np.float32)
        Y, X = np.mgrid[:h, :w]
        for xx, yy, polarity in minutiae_list:
            spiral_phase += polarity * np.arctan2(Y - yy, X - xx)

        return torch.from_numpy(spiral_phase).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        orientation = np.load(self.orientation_paths[idx]).astype(np.float32)

        ori_tensor = torch.from_numpy(orientation)

        min_path = self.minutiae_paths[idx]
        minutiae_points = []
        if os.path.exists(min_path):
            # Parse "x,y,type"
            with open(min_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        # Assuming format is x, y, type
                        minutiae_points.append(
                            (float(parts[0]), float(parts[1]), int(parts[2]))
                        )

        # Generate Gaussian Heatmap
        h, w = ori_tensor.shape
        freq = np.load(self.frequency_paths[idx]).astype(np.float32)

        minutiae_map = self._generate_heatmap(minutiae_points, h, w)
        cos2theta = torch.cos(2 * ori_tensor).unsqueeze(0)  # (1, H, W)
        sin2theta = torch.sin(2 * ori_tensor).unsqueeze(0)  # (1, H, W)
        freq_tensor = torch.from_numpy(freq).unsqueeze(0)  # (1, H, W)

        cos_cont = np.load(self.cos_cont_paths[idx]).astype(np.float32)
        cos_full = np.load(self.cos_full_paths[idx]).astype(np.float32)

        sin_cont = np.load(self.sin_cont_paths[idx]).astype(np.float32)
        sin_full = np.load(self.sin_full_paths[idx]).astype(np.float32)

        spiral_phase = self._get_spiral_phase(minutiae_points, h, w)

        spiral_phasor_cos = torch.cos(spiral_phase)
        spiral_phasor_sin = torch.sin(spiral_phase)
        spiral_phasor = torch.cat([spiral_phasor_sin, spiral_phasor_cos], dim=0)
        inputs = torch.cat([sin2theta, cos2theta, minutiae_map, freq_tensor], dim=0)

        return {
            "inputs": inputs,  # Shape: (3, H, W)
            "cos_cont": torch.from_numpy(cos_cont).unsqueeze(0),
            "cos_full": torch.from_numpy(cos_full).unsqueeze(0),
            "sin_cont": torch.from_numpy(sin_cont).unsqueeze(0),
            "sin_full": torch.from_numpy(sin_full).unsqueeze(0),
            "minutiae_map": minutiae_map,  # (1, H, W)
            "spiral_phasor": spiral_phasor,  # (2, H, W)
        }
