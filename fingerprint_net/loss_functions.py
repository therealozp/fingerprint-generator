import torch
import torch.nn as nn


class FingerprintLoss(nn.Module):
    def __init__(self, lambda_img=1.0, lambda_norm=0.1, lambda_grad=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_img = lambda_img
        self.lambda_grad = lambda_grad

    def mse_loss(self, pred, target):
        return self.mse(pred, target)

    def gradient_loss(self, pred_sin, pred_cos, orientation_field):
        # orientation_field is in radians (0 to pi)
        # We want the gradient of the phase (atan2(sin, cos)) to align with orientation
        # This is complex to compute directly due to wrapping.
        # Simplified approach: Penalize divergence in local flow.
        # Alternatively, we rely on the reconstruction loss implicitly handling this.
        # For this snippet, we will stick to reconstruction + norm for stability.
        return 0.0

    def forward(self, pred, spiral_phasor, target_cont, target_full):
        sin_c = pred[:, 0:1, :, :]
        cos_c = pred[:, 1:2, :, :]

        sin_s = spiral_phasor[:, 0:1, :, :]
        cos_s = spiral_phasor[:, 1:2, :, :]

        pred_cont = cos_c
        pred_full = cos_c * cos_s - sin_c * sin_s

        loss_cont = self.mse_loss(pred_cont, target_cont)
        loss_full = self.mse_loss(pred_full, target_full)

        total_loss = self.lambda_img * (loss_cont + loss_full)

        return total_loss, loss_cont, loss_full
