import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class FingerprintOrientationDataset(Dataset):
    def __init__(
        self,
        orientation_paths,
        minutiae_paths,
        continuous_paths,
        full_paths,
        img_size=(256, 256),
    ):
        self.orientation_paths = orientation_paths
        self.minutiae_paths = minutiae_paths
        self.continuous_paths = continuous_paths
        self.full_paths = full_paths
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
            # Simple Gaussian stamp
            # Check bounds
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

        cos2theta = torch.cos(2 * ori_tensor).unsqueeze(0)  # (1, H, W)
        sin2theta = torch.sin(2 * ori_tensor).unsqueeze(0)  # (1, H, W)

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
        minutiae_map = self._generate_heatmap(minutiae_points, h, w)

        # Helper to load and normalize image to [-1, 1]
        def load_image(path):
            img = Image.open(path).convert("L")  # Grayscale
            img = img.resize((w, h))  # Ensure matches orientation dimensions
            img_tensor = torch.from_numpy(np.array(img)).float()
            img_tensor = (img_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]
            return img_tensor.unsqueeze(0)

        cont_path = self.continuous_paths[idx]
        full_path = self.full_paths[idx]

        continuous_img = load_image(cont_path)
        full_img = load_image(full_path)
        spiral_phase = self._get_spiral_phase(minutiae_points, h, w)

        spiral_phasor_cos = torch.cos(spiral_phase)
        spiral_phasor_sin = torch.sin(spiral_phase)
        spiral_phasor = torch.cat([spiral_phasor_sin, spiral_phasor_cos], dim=0)
        inputs = torch.cat([sin2theta, cos2theta, minutiae_map], dim=0)

        return {
            "inputs": inputs,  # Shape: (3, H, W)
            "target_continuous": continuous_img,  # Shape: (1, H, W)
            "target_full": full_img,  # Shape: (1, H, W)
            "spiral_phasor": spiral_phasor,  # (2, H, W)
        }
