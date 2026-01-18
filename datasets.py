import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class FingerprintOrientationDataset(Dataset):
    def __init__(
        self,
        orientation_dir,
        minutiae_dir,
        continuous_dir,
        full_dir,
        img_size=(256, 256),
    ):
        """
        Args:
            orientation_dir (str): Path to .npy orientation files (H, W).
            minutiae_dir (str): Path to text files with "x,y,type".
            continuous_dir (str): Path to target continuous fingerprints (.png).
            full_dir (str): Path to target full fingerprints (.png).
            img_size (tuple): Target resize dimension (H, W).
        """
        self.orientation_dir = orientation_dir
        self.minutiae_dir = minutiae_dir
        self.continuous_dir = continuous_dir
        self.full_dir = full_dir
        self.img_size = img_size

        # Get list of filenames (assuming basenames match across directories)
        # e.g., "finger_01.npy" matches "finger_01.txt", etc.
        self.filenames = [
            f.replace(".npy", "")
            for f in os.listdir(orientation_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.filenames)

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
        base_name = self.filenames[idx]

        # ---------------------------
        # 1. Load Orientation (.npy)
        # ---------------------------
        ori_path = os.path.join(self.orientation_dir, base_name + ".npy")
        orientation = np.load(ori_path).astype(np.float32)

        ori_tensor = torch.from_numpy(orientation)

        cos2theta = torch.cos(2 * ori_tensor).unsqueeze(0)  # (1, H, W)
        sin2theta = torch.sin(2 * ori_tensor).unsqueeze(0)  # (1, H, W)

        min_path = os.path.join(self.minutiae_dir, base_name + ".txt")
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

        cont_path = os.path.join(
            self.continuous_dir, base_name + ".png"
        )  # Adjust extension if needed
        full_path = os.path.join(self.full_dir, base_name + ".png")

        continuous_img = load_image(cont_path)
        full_img = load_image(full_path)
        spiral_phase = self._get_spiral_phase(minutiae_points, h, w)

        spiral_phasor_cos = torch.cos(spiral_phase)
        spiral_phaser_sin = torch.sin(spiral_phase)
        spiral_phasor = torch.cat([spiral_phasor_cos, spiral_phaser_sin], dim=0)
        inputs = torch.cat([cos2theta, sin2theta, minutiae_map], dim=0)

        return {
            "inputs": inputs,  # Shape: (3, H, W)
            "target_continuous": continuous_img,  # Shape: (1, H, W)
            "target_full": full_img,  # Shape: (1, H, W)
            "spiral_phasor": spiral_phasor,  # (2, H, W)
            "filename": base_name,  # For debugging
        }
