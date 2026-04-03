import os
from random import random
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from abc import ABC, abstractmethod


class BaseFingerprintDataset(Dataset, ABC):
    """Base class for fingerprint datasets with common functionality."""
    
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

    def _get_spiral_phase(self, minutiae_list, h, w):
        """Generate spiral phase from minutiae list."""
        spiral_phase = np.zeros((h, w), dtype=np.float32)
        Y, X = np.mgrid[:h, :w]
        for xx, yy, polarity in minutiae_list:
            spiral_phase += polarity * np.arctan2(Y - yy, X - xx)

        return torch.from_numpy(spiral_phase).unsqueeze(0)  # (1, H, W)

    def _load_minutiae_points(self, min_path):
        """Load minutiae points from file."""
        minutiae_points = []
        if os.path.exists(min_path):
            with open(min_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        minutiae_points.append(
                            (float(parts[0]), float(parts[1]), int(parts[2]))
                        )
        return minutiae_points

    def _load_data(self, idx):
        """Load common data for a sample."""
        orientation = np.load(self.orientation_paths[idx]).astype(np.float32)
        ori_tensor = torch.from_numpy(orientation)
        h, w = ori_tensor.shape
        
        minutiae_points = self._load_minutiae_points(self.minutiae_paths[idx])
        freq = np.load(self.frequency_paths[idx]).astype(np.float32)

        cos_cont = np.load(self.cos_cont_paths[idx]).astype(np.float32)
        cos_full = np.load(self.cos_full_paths[idx]).astype(np.float32)

        sin_cont = np.load(self.sin_cont_paths[idx]).astype(np.float32)
        sin_full = np.load(self.sin_full_paths[idx]).astype(np.float32)

        return {
            "ori_tensor": ori_tensor,
            "orientation": orientation,
            "minutiae_points": minutiae_points,
            "freq": freq,
            "h": h,
            "w": w,
            "cos_cont": cos_cont,
            "cos_full": cos_full,
            "sin_cont": sin_cont,
            "sin_full": sin_full,
        }

    @abstractmethod
    def _generate_minutiae_map(self, orientation, freq, minutiae_points, h, w):
        """Generate minutiae map. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _build_inputs(self, sin2theta, cos2theta, freq_tensor, minutiae_map):
        """Build input tensor. Must be implemented by subclasses."""
        pass

    def __getitem__(self, idx):
        data = self._load_data(idx)
        ori_tensor = data["ori_tensor"]
        orientation = data["orientation"]
        minutiae_points = data["minutiae_points"]
        freq = data["freq"]
        h, w = data["h"], data["w"]

        cos2theta = torch.cos(2 * ori_tensor).unsqueeze(0)
        sin2theta = torch.sin(2 * ori_tensor).unsqueeze(0)
        freq_tensor = torch.from_numpy(freq).unsqueeze(0)

        minutiae_map = self._generate_minutiae_map(orientation, freq, minutiae_points, h, w)

        spiral_phase = self._get_spiral_phase(minutiae_points, h, w)
        spiral_phasor_cos = torch.cos(spiral_phase)
        spiral_phasor_sin = torch.sin(spiral_phase)
        spiral_phasor = torch.cat([spiral_phasor_sin, spiral_phasor_cos], dim=0)

        inputs = self._build_inputs(sin2theta, cos2theta, freq_tensor, minutiae_map)

        return {
            "inputs": inputs,
            "cos_cont": torch.from_numpy(data["cos_cont"]).unsqueeze(0),
            "cos_full": torch.from_numpy(data["cos_full"]).unsqueeze(0),
            "sin_cont": torch.from_numpy(data["sin_cont"]).unsqueeze(0),
            "sin_full": torch.from_numpy(data["sin_full"]).unsqueeze(0),
            "minutiae_map": minutiae_map,
            "spiral_phasor": spiral_phasor,
        }


class FingerprintOrientationDataset(BaseFingerprintDataset):
    """V1: Simple Gaussian heatmap for minutiae."""

    def _generate_heatmap(self, minutiae, h, w, sigma=3):
        """Convert list of (x, y) points into a Gaussian heatmap."""
        heatmap = torch.zeros((h, w))
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        for x, y, _ in minutiae:
            if 0 <= x < w and 0 <= y < h:
                dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
                heatmap += torch.exp(-dist_sq / (2 * sigma**2))

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.unsqueeze(0)

    def _generate_minutiae_map(self, orientation, freq, minutiae_points, h, w):
        return self._generate_heatmap(minutiae_points, h, w)

    def _build_inputs(self, sin2theta, cos2theta, freq_tensor, minutiae_map):
        return torch.cat([sin2theta, cos2theta, minutiae_map, freq_tensor], dim=0)


def map_frequency_to_radius(freq_array, in_min=0.087, in_max=0.183, out_min=7, out_max=3):
    normalized = (freq_array - in_min) / (in_max - in_min)
    mapped = out_min + normalized * (out_max - out_min)
    clipped = np.clip(mapped, min(out_min, out_max), max(out_min, out_max))
    return np.round(clipped).astype(int)

def get_spaced_minutiae_map(fingerprint_image, orientation_map, freq_map, minutiae_list, h, w, sigma_s):
    H_ending = np.zeros((h, w, 6))
    H_bifurcation = np.zeros((h, w, 6))
    max_freq = freq_map.max()
    min_freq = freq_map.min()
    Y, X = np.indices((h, w))
    
    # 1. Loop through minutiae first (outer loop)
    for x, y, p in minutiae_list:
        x, y = int(x), int(y)
        # -- SPATIAL MATH (Done once per minutia) --
        dist_sq = (X - x)**2 + (Y - y)**2
        Cs = np.exp(-dist_sq / (2 * sigma_s**2))
    
        theta = orientation_map[y, x]
        radius = map_frequency_to_radius(freq_map[y, x], in_min=min_freq, in_max=max_freq, out_min=7, out_max=3)

        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1) # slicing exclusive at the end
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        neighborhood = fingerprint_image[y_min:y_max, x_min:x_max]
        total_pixels = neighborhood.size
        negative_count = np.sum(neighborhood < 0)
        
        is_ending = negative_count > (total_pixels / 2)

        theta = theta + np.pi / 2 if is_ending else theta - np.pi / 2
        theta = theta % (2 * np.pi)
        
        # -- CHANNEL MATH (Done 6 times per minutia) --
        for k in range(6):
            theta_k = k * np.pi / 3 
            
            abs_diff = np.abs(theta - theta_k)
            if abs_diff <= np.pi:
                d_phi = abs_diff
            else:
                d_phi = 2 * np.pi - abs_diff
            
            Co = np.exp(-d_phi / (2 * sigma_s**2))
            contribution = Cs * Co
            
            if is_ending:
                H_ending[:, :, k] += contribution
            else:
                H_bifurcation[:, :, k] += contribution
                
    H_combined = np.concatenate([H_ending, H_bifurcation], axis=-1)
    tensor = torch.from_numpy(H_combined).float()
    final_tensor = tensor.permute(2, 0, 1)
    
    return final_tensor

def get_combined_minutiae_map(fingerprint_image, orientation_map, freq_map, minutiae_list, h, w, sigma_s):
    H = np.zeros((h, w, 6))
    max_freq = freq_map.max()
    min_freq = freq_map.min()
    Y, X = np.indices((h, w))
    
    # 1. Loop through minutiae first (outer loop)
    for x, y, p in minutiae_list:
        x, y = int(x), int(y)
        # -- SPATIAL MATH (Done once per minutia) --
        dist_sq = (X - x)**2 + (Y - y)**2
        Cs = np.exp(-dist_sq / (2 * sigma_s**2))
    
        theta = orientation_map[y, x]
        radius = map_frequency_to_radius(freq_map[y, x], in_min=min_freq, in_max=max_freq, out_min=7, out_max=3)

        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1) # slicing exclusive at the end
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        neighborhood = fingerprint_image[y_min:y_max, x_min:x_max]
        total_pixels = neighborhood.size
        negative_count = np.sum(neighborhood < 0)
        
        is_ending = negative_count > (total_pixels / 2)

        theta = theta + np.pi / 2 if is_ending else theta - np.pi / 2
        theta = theta % (2 * np.pi)
        
        # -- CHANNEL MATH (Done 6 times per minutia) --
        for k in range(6):
            theta_k = k * np.pi / 3 
            
            abs_diff = np.abs(theta - theta_k)
            if abs_diff <= np.pi:
                d_phi = abs_diff
            else:
                d_phi = 2 * np.pi - abs_diff
            
            Co = np.exp(-d_phi / (2 * sigma_s**2))
            contribution = Cs * Co
            
            if is_ending:
                H[:, :, k] += contribution
                
    tensor = torch.from_numpy(H).float()
    final_tensor = tensor.permute(2, 0, 1)
    
    return final_tensor

# V3 has multi-channel minutiae maps with spaced representation (differentiates endings/bifurcations)
class FingerprintOrientationDatasetV3(BaseFingerprintDataset):
    """V3: Multi-channel minutiae maps with spaced representation."""

    def _generate_minutiae_map(self, orientation, freq, minutiae_points, h, w):
        return get_spaced_minutiae_map(
            fingerprint_image=orientation,
            orientation_map=orientation,
            freq_map=freq,
            minutiae_list=minutiae_points,
            h=h,
            w=w,
            sigma_s=2,
        )

    def _build_inputs(self, sin2theta, cos2theta, freq_tensor, minutiae_map):
        return torch.cat([sin2theta, cos2theta, freq_tensor, minutiae_map], dim=0)

# V2 has multi-channel minutiae maps (combined, doesn't differentiate endings/bifurcations)
class FingerprintOrientationDatasetV2(BaseFingerprintDataset):
    """V2: Multi-channel minutiae maps with combined representation."""

    def _generate_minutiae_map(self, orientation, freq, minutiae_points, h, w):
        return get_combined_minutiae_map(
            fingerprint_image=orientation,
            orientation_map=orientation,
            freq_map=freq,
            minutiae_list=minutiae_points,
            h=h,
            w=w,
            sigma_s=2,
        )

    def _build_inputs(self, sin2theta, cos2theta, freq_tensor, minutiae_map):
        return torch.cat([sin2theta, cos2theta, freq_tensor, minutiae_map], dim=0)

if __name__ == "__main__":
    base_dir = "/data/hot/khangphuanhle/data_v3"

    orientation_dir = "orientation_maps"
    minutiae_dir = "minutiae_locations"
    frequency_dir = "freq_maps"
    cos_cont_dir = "cos_cont"
    cos_full_dir = "cos_full"
    sin_cont_dir = "sin_cont"
    sin_full_dir = "sin_full"

    orientation_paths = []
    minutiae_paths = []
    frequency_paths = []
    cos_cont_paths = []
    cos_full_paths = []
    sin_cont_paths = []
    sin_full_paths = []

    for item in os.listdir(os.path.join(base_dir, orientation_dir)):
        if item.endswith(".npy"):
            cos_cont_paths.append(os.path.join(base_dir, cos_cont_dir, item))
            cos_full_paths.append(os.path.join(base_dir, cos_full_dir, item))
            sin_cont_paths.append(os.path.join(base_dir, sin_cont_dir, item))
            sin_full_paths.append(os.path.join(base_dir, sin_full_dir, item))
            minutiae_paths.append(
                os.path.join(base_dir, minutiae_dir, item.replace(".npy", ".txt"))
            )
            orientation_paths.append(os.path.join(base_dir, orientation_dir, item))
            frequency_paths.append(os.path.join(base_dir, frequency_dir, item))

    assert (
        len(orientation_paths)
        == len(minutiae_paths)
        == len(frequency_paths)
        == len(cos_cont_paths)
        == len(cos_full_paths)
        == len(sin_cont_paths)
        == len(sin_full_paths)
    ), "Mismatch in dataset lengths."

    dataset = FingerprintOrientationDatasetV2(
        orientation_paths=orientation_paths,
        minutiae_paths=minutiae_paths,
        frequency_paths=frequency_paths,
        cos_cont_paths=cos_cont_paths,
        cos_full_paths=cos_full_paths,
        sin_cont_paths=sin_cont_paths,
        sin_full_paths=sin_full_paths,
    )

    sample = dataset[0]
    print(sample["inputs"].shape)  # Should be (3, H, W)
