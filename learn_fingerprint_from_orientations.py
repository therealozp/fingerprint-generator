import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 1. Reconstruction Losses (Pixel-wise MSE)
        loss_cont = self.mse(pred_cont, target_cont)
        loss_full = self.mse(pred_full, target_full)

        # 2. Phasor Normalization Loss
        # Enforce cos^2 + sin^2 = 1
        cos_c = pred_phasor[:, 0:1, :, :]
        sin_c = pred_phasor[:, 1:2, :, :]
        norm_map = cos_c**2 + sin_c**2
        loss_norm = self.mse(norm_map, torch.ones_like(norm_map))

        total_loss = (self.lambda_img * (loss_cont + loss_full)) + (
            self.lambda_norm * loss_norm
        )

        return total_loss, loss_cont, loss_full, loss_norm


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhaseReconstructionUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = FingerprintLoss()


def train_step(batch):
    inputs = batch["inputs"].to(device)  # (B, 3, H, W)
    target_c = batch["target_continuous"].to(device)  # (B, 1, H, W)
    target_f = batch["target_full"].to(device)  # (B, 1, H, W)
    spiral_phasor = batch["spiral_phasor"].to(device)  # (B, 2, H, W)

    optimizer.zero_grad()

    # Forward Pass
    pred_c, pred_f, pred_phasor = model(inputs, spiral_phasor)

    # Calculate Loss
    loss, l_cont, l_full, l_norm = criterion(
        pred_c, pred_f, pred_phasor, target_c, target_f
    )

    loss.backward()
    optimizer.step()

    return loss.item()
