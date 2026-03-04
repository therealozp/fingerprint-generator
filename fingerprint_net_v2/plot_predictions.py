import torch
import torch.nn as nn

from model import FingerprintUNet
from loss_functions import FingerprintLoss
from torch.utils.data import DataLoader
from dataset import FingerprintOrientationDataset

from dataclasses import dataclass
from tqdm import tqdm

import os


if __name__ == "__main__":
    original_images_dir = "./data/full_images"
    target_images_dir = "./data/continuous_images"
    minutiae_dir = "./data/minutiae_locations"
    orientation_maps_dir = "./data/orientation_maps"

    orig_paths = []
    cont_paths = []
    minutiae_paths = []
    orientation_map_paths = []

    for item in os.listdir(original_images_dir):
        if item.endswith(".png"):
            orig_paths.append(os.path.join(original_images_dir, item))
            cont_paths.append(os.path.join(target_images_dir, item))
            minutiae_paths.append(
                os.path.join(minutiae_dir, item.replace(".png", ".txt"))
            )
            orientation_map_paths.append(
                os.path.join(orientation_maps_dir, item.replace(".png", ".npy"))
            )

    orientation_map_paths = orientation_map_paths[:-5]
    minutiae_paths = minutiae_paths[:-5]
    cont_paths = cont_paths[:-5]
    orig_paths = orig_paths[:-5]

    dataset = FingerprintOrientationDataset(
        orientation_paths=orientation_map_paths,
        minutiae_paths=minutiae_paths,
        continuous_paths=cont_paths,
        full_paths=orig_paths,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = FingerprintUNet(in_channels=3, out_channels=2)
    model.load_state_dict(torch.load("checkpoints/fingerprint_unet_best.pth"))
    model.eval()

    # plot predictions
    import matplotlib.pyplot as plt

    for i, batch in enumerate(dataloader):
        inputs = batch["inputs"]  # (B, 3, H, W)
        spiral_phasor = batch["spiral_phasor"]  # (B, 2, H, W)

        target_f = batch["target_full"]  # (B, 1, H, W)
        target_c = batch["target_continuous"]  # (B, 1, H, W)

        with torch.no_grad():
            pred = model(inputs)
            sin_c = pred[:, 0, :, :]
            cos_c = pred[:, 1, :, :]

            sin_s = spiral_phasor[:, 0, :, :]
            cos_s = spiral_phasor[:, 1, :, :]

            pred_cont = cos_c
            pred_full = cos_c * cos_s - sin_c * sin_s

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.title("Input")
        plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.title("Target Continuous Fingerprint")
        plt.imshow(target_c[0, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(torch.atan2(sin_c[0], cos_c[0]).cpu().numpy(), cmap="gray")
        plt.title("Predicted phase")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.title("Predicted Continuous Fingerprint")
        plt.imshow(pred_cont[0].cpu().numpy(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.title("Target Full Fingerprint")
        plt.imshow(target_f[0, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.title("Predicted Full Fingerprint")
        plt.imshow(pred_full[0].cpu().numpy(), cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        if i >= 5:
            break
