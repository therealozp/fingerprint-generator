import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataclasses import dataclass
import os

from datasets import FingerprintOrientationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class PhaseReconstructionUNet(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Output: 2 Channels for Cos(phi_c) and Sin(phi_c)
        self.final_conv = nn.Conv2d(features[0], 2, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, spiral_phasor):
        """
        x: [Batch, 3, H, W] -> (Cos2Theta, Sin2Theta, MinutiaeMap)
        spiral_phasor: [Batch, 2, H, W] -> (Cos(phi_s), Sin(phi_s))
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        # 1. Predict Continuous Phase Phasor
        # Output shape: [Batch, 2, H, W] -> Channel 0: Cos_c, Channel 1: Sin_c
        pred_phasor = self.tanh(self.final_conv(x))

        cos_c = pred_phasor[:, 0:1, :, :]
        sin_c = pred_phasor[:, 1:2, :, :]

        # 2. Physics-Informed Combination (AM-FM Model)
        # Formula: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        # a = Continuous Phase, b = Spiral Phase
        cos_s = spiral_phasor[:, 0:1, :, :]
        sin_s = spiral_phasor[:, 1:2, :, :]

        # Reconstructed Full Fingerprint (Image)
        full_reconstruction = (cos_c * cos_s) - (sin_c * sin_s)

        # Reconstructed Continuous Fingerprint (Image)
        cont_reconstruction = cos_c

        return cont_reconstruction, full_reconstruction, pred_phasor


class FingerprintLoss(nn.Module):
    def __init__(self, lambda_img=1.0, lambda_norm=0.1, lambda_grad=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_img = lambda_img
        self.lambda_norm = lambda_norm
        self.lambda_grad = lambda_grad

    def gradient_loss(self, pred_sin, pred_cos, orientation_field):
        # orientation_field is in radians (0 to pi)
        # We want the gradient of the phase (atan2(sin, cos)) to align with orientation
        # This is complex to compute directly due to wrapping.
        # Simplified approach: Penalize divergence in local flow.
        # Alternatively, we rely on the reconstruction loss implicitly handling this.
        # For this snippet, we will stick to reconstruction + norm for stability.
        return 0.0

    def forward(self, pred_cont, pred_full, pred_phasor, target_cont, target_full):
        loss_cont = self.mse(pred_cont, target_cont)
        loss_full = self.mse(pred_full, target_full)

        cos_c = pred_phasor[:, 0:1, :, :]
        sin_c = pred_phasor[:, 1:2, :, :]
        norm_map = cos_c**2 + sin_c**2
        loss_norm = self.mse(norm_map, torch.ones_like(norm_map))

        total_loss = (self.lambda_img * (loss_cont + loss_full)) + (
            self.lambda_norm * loss_norm
        )

        return total_loss, loss_cont, loss_full, loss_norm


device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhaseReconstructionUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 5e-4
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True


def train(model: nn.Module, loader: DataLoader, cfg: TrainConfig):
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(
        enabled=(cfg.amp and cfg.device.startswith("cuda")),
    )
    criterion = FingerprintLoss()

    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        running = 0.0
        loss_from_continuous_phase = 0.0
        loss_from_full_phase = 0.0
        sin_cos_normalization_loss = 0.0

        for step, input in enumerate(tqdm(loader)):
            inputs = input["inputs"].to(device)  # (B, 3, H, W)
            target_c = input["target_continuous"].to(device)  # (B, 1, H, W)
            target_f = input["target_full"].to(device)  # (B, 1, H, W)
            spiral_phasor = input["spiral_phasor"].to(device)  # (B, 2, H, W)

            optimizer.zero_grad()

            # Forward Pass
            with torch.amp.autocast(
                device_type=cfg.device,
                enabled=scaler.is_enabled(),
            ):
                pred_c, pred_f, pred_phasor = model(inputs, spiral_phasor)

                # Calculate Loss
                loss, l_cont, l_full, l_norm = criterion(
                    pred_c, pred_f, pred_phasor, target_c, target_f
                )

                loss.backward()
                optimizer.step()

            running += loss.item()
            loss_from_continuous_phase += l_cont.item()
            loss_from_full_phase += l_full.item()
            sin_cos_normalization_loss += l_norm.item()

        avg = running / max(1, len(loader))
        print(
            f"epoch {epoch+1:03d}/{cfg.epochs} | total loss = {avg:.6f} | cont phase loss = {loss_from_continuous_phase / max(1, len(loader)):.6f} | full phase loss = {loss_from_full_phase / max(1, len(loader)):.6f} | norm loss = {sin_cos_normalization_loss / max(1, len(loader)):.6f}"
        )


if __name__ == "__main__":
    original_images_dir = "data/spiral_images/"
    target_images_dir = "data/continuous_images/"
    minutiae_dir = "data/minutiae_locations/"
    orientation_maps_dir = "data/orientation_maps/"

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

    train_paths = orig_paths[:-8]
    train_targets = cont_paths[:-8]
    train_minutiae = minutiae_paths[:-8]
    train_orientations = orientation_map_paths[:-8]

    test_paths = orig_paths[-8:]
    test_targets = cont_paths[-8:]
    test_minutiae = minutiae_paths[-8:]
    test_orientations = orientation_map_paths[-8:]

    train_ds = FingerprintOrientationDataset(
        orientation_paths=train_orientations,
        continuous_paths=train_paths,
        full_paths=train_targets,
        minutiae_paths=train_minutiae,
    )

    cfg = TrainConfig(epochs=20, lr=1e-1)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = PhaseReconstructionUNet(in_channels=3, features=[64, 128, 256, 512])

    train(model, loader, cfg)

    test_ds = FingerprintOrientationDataset(
        orientation_paths=test_orientations,
        continuous_paths=test_paths,
        full_paths=test_targets,
        minutiae_paths=test_minutiae,
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
        for input in test_loader:
            x = input["inputs"].to(device)  # (1, 3, H, W)
            spiral_phasors = input["spiral_phasor"].to(device)  # (1, 2, H, W)
            pred_c, pred_f, _ = model(x, spiral_phasors)  # (1, 2, H, W)

            plt.figure(figsize=(14, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input image")
            plt.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Predicted continuous image")
            plt.imshow(pred_c[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Phase (phi)")
            plt.imshow(pred_f[0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            plt.show()
