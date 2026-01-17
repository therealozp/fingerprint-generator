import math
from dataclasses import dataclass
from typing import Optional, Tuple

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import PIL.Image as Image
import numpy as np
import os

from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------------
# 1) U-Net building blocks
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # in_ch here should already reflect concatenation sizing
            self.up = nn.ConvTranspose2d(
                in_ch // 2, in_ch // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_ch, out_ch)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder skip
        x1 = self.up(x1)

        # Pad if odd sizes occur
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetPhase(nn.Module):
    """
    U-Net that predicts a phase field phi (one channel).
    Then you render image = cos(phi) outside the model.
    """

    def __init__(self, in_ch: int = 1, base: int = 64, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base * 8, (base * 16) // factor)

        self.up1 = Up(base * 16, base * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base * 8, base * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base * 4, base * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base * 2, base, bilinear=bilinear)

        # Predict phase (phi). 1 output channel.
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

        # Optional: learn a scale so early training doesn't go unstable.
        # You can start with scale=pi so cos(phi) spans reasonable variation.
        self.phase_scale = nn.Parameter(torch.tensor(math.pi, dtype=torch.float32))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        phi_raw = self.outc(x)
        # Scale helps keep phase in a "useful" range initially.
        phi = phi_raw * self.phase_scale
        return phi


# ----------------------------
# 2) Dataset stub
# ----------------------------
class FingerprintPhaseDataset(Dataset):
    def __init__(
        self,
        orig_paths,
        cont_paths,
        minutiae_paths=None,
        mask_probability=0.5,
        mask_size=32,
    ):
        assert len(orig_paths) == len(cont_paths)
        self.orig_paths = orig_paths
        self.cont_paths = cont_paths
        self.minutiae_paths = minutiae_paths

        self.mask_probability = mask_probability
        self.mask_size = mask_size

        self.sigma = 5.0
        self.k_size = 15
        self.k_half = self.k_size // 2
        self.kernel = self._generate_gaussian_kernel(self.k_size, self.sigma)

    def _generate_gaussian_kernel(self, size, sigma):
        # Create a grid of coordinates
        coords = torch.arange(size).float() - (size // 2)
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")

        # Calculate Gaussian
        gaussian = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()  # Normalize peak to 1.0
        return gaussian

    def _load_png(self, path):
        img = Image.open(path).convert("L")  # grayscale
        img = np.array(img, dtype=np.float32)
        img = img / 127.5 - 1.0  # [-1,1]
        return torch.from_numpy(img).unsqueeze(0)  # [1,H,W]

    def _load_minutiae_positions(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

        for item in lines:
            r, c, _ = item.strip().split(",")
            yield (int(r), int(c))

    def __len__(self):
        return len(self.orig_paths)

    def __getitem__(self, idx):
        x = self._load_png(self.orig_paths[idx])
        y = self._load_png(self.cont_paths[idx])
        weights = torch.ones_like(y)

        # The strength of the minutiae importance (e.g., 50x more important)
        minutiae_strength = 50.0

        img_h, img_w = weights.shape[-2:]

        for item in self._load_minutiae_positions(self.minutiae_paths[idx]):
            r, c = int(item[0]), int(item[1])  # Ensure integers

            # Define limits on the main image
            r_min = max(r - self.k_half, 0)
            r_max = min(r + self.k_half + 1, img_h)
            c_min = max(c - self.k_half, 0)
            c_max = min(c + self.k_half + 1, img_w)

            # Define limits on the Gaussian kernel (handling boundary crops)
            k_r_min = self.k_half - (r - r_min)
            k_r_max = k_r_min + (r_max - r_min)
            k_c_min = self.k_half - (c - c_min)
            k_c_max = k_c_min + (c_max - c_min)

            weights[..., r_min:r_max, c_min:c_max] += (
                self.kernel[k_r_min:k_r_max, k_c_min:k_c_max] * minutiae_strength
            )

            noise_r_min = max(r - self.mask_size // 2, 0)
            noise_r_max = min(r + self.mask_size // 2 + 1, img_h)
            noise_c_min = max(c - self.mask_size // 2, 0)
            noise_c_max = min(c + self.mask_size // 2 + 1, img_w)

            if random.random() < self.mask_probability:
                noise = torch.randn_like(
                    x[..., noise_r_min:noise_r_max, noise_c_min:noise_c_max]
                )
                x[..., noise_r_min:noise_r_max, noise_c_min:noise_c_max] = noise

        return x, y, weights


# ----------------------------
# 3) Training
# ----------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 2e-4
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True  # mixed precision if cuda


def train_unet_phase(model: nn.Module, loader: DataLoader, cfg: TrainConfig):
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(
        enabled=(cfg.amp and cfg.device.startswith("cuda")),
    )

    model.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for step, (x, y_cont, weight) in enumerate(tqdm(loader)):
            x = x.to(cfg.device, non_blocking=True)
            y_cont = y_cont.to(cfg.device, non_blocking=True)
            weight = weight.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=cfg.device,
                enabled=scaler.is_enabled(),
            ):
                phi = model(x)  # [B,1,H,W]
                y_hat = torch.cos(phi)  # [B,1,H,W]

                pixel_loss = F.l1_loss(y_hat, y_cont, reduction="none")
                weighted_loss = pixel_loss * weight

                loss = weighted_loss.mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        avg = running / max(1, len(loader))
        print(
            f"epoch {epoch+1:03d}/{cfg.epochs} | L1(cos(phi), cont) = {avg:.6f} | phase_scale={model.phase_scale.item():.4f}"
        )


# if __name__ == "__main__":
#     original_images_dir = "spiral_images/"
#     target_images_dir = "continuous_images/"
#     minutiae_dir = "minutiae_locations/"

#     orig_paths = []
#     cont_paths = []
#     minutiae_paths = []

#     for item in os.listdir(original_images_dir):
#         if item.endswith(".png"):
#             orig_paths.append(os.path.join(original_images_dir, item))
#             cont_paths.append(os.path.join(target_images_dir, item))
#             minutiae_paths.append(
#                 os.path.join(minutiae_dir, item.replace(".png", ".txt"))
#             )

#     train_ds = FingerprintPhaseDataset(
#         orig_paths,
#         cont_paths,
#         minutiae_paths,
#         mask_probability=0.6,
#         mask_size=32,
#     )

#     loader = DataLoader(
#         train_ds,
#         batch_size=TrainConfig().batch_size,
#         shuffle=True,
#         num_workers=TrainConfig().num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     for x, y_cont, weight in loader:
#         plt.figure(figsize=(14, 6))
#         plt.subplot(1, 3, 1)
#         plt.title("x")
#         plt.imshow(x[0, 0].cpu().numpy(), cmap="gray")
#         plt.axis("off")

#         plt.subplot(1, 3, 2)
#         plt.title("y_cont")
#         plt.imshow(y_cont[0, 0].cpu().numpy(), cmap="gray")
#         plt.axis("off")

#         plt.subplot(1, 3, 3)
#         plt.title("weight")
#         plt.imshow(weight[0, 0].cpu().numpy(), cmap="gray")
#         plt.axis("off")

#         plt.show()

#         break


if __name__ == "__main__":
    original_images_dir = "spiral_images/"
    target_images_dir = "continuous_images/"
    minutiae_dir = "minutiae_locations/"

    orig_paths = []
    cont_paths = []
    minutiae_paths = []

    for item in os.listdir(original_images_dir):
        if item.endswith(".png"):
            orig_paths.append(os.path.join(original_images_dir, item))
            cont_paths.append(os.path.join(target_images_dir, item))
            minutiae_paths.append(
                os.path.join(minutiae_dir, item.replace(".png", ".txt"))
            )

    train_paths = orig_paths[:-6]
    train_targets = cont_paths[:-6]
    train_minutiae = minutiae_paths[:-6]

    test_paths = orig_paths[-6:]
    test_targets = cont_paths[-6:]
    test_minutiae = minutiae_paths[-6:]

    train_ds = FingerprintPhaseDataset(
        train_paths,
        train_targets,
        train_minutiae,
        mask_probability=0.6,
        mask_size=32,
    )
    loader = DataLoader(
        train_ds,
        batch_size=TrainConfig().batch_size,
        shuffle=True,
        num_workers=TrainConfig().num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = UNetPhase(in_ch=1, base=128, bilinear=True)
    cfg = TrainConfig(epochs=50)
    train_unet_phase(model, loader, cfg)

    test_ds = FingerprintPhaseDataset(
        test_paths, test_targets, test_minutiae, mask_probability=0.0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    # make predictions, compare to target continuous images
    model.eval()
    with torch.no_grad():
        for x, y_cont, weight in test_loader:
            x = x.to(cfg.device, non_blocking=True)
            y_cont = y_cont.to(cfg.device, non_blocking=True)

            phi = model(x)
            y_hat = torch.cos(phi)

            plt.figure(figsize=(14, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input image")
            plt.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Predicted continuous image")
            plt.imshow(y_hat[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Phase (phi)")
            plt.imshow(phi[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.show()
