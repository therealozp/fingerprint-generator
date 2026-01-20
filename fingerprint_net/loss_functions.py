import torch
import torch.nn as nn


class FingerprintLoss(nn.Module):
    def __init__(self, lambda_img=1.0, lambda_norm=0.1, lambda_grad=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_img = lambda_img
        self.lambda_norm = lambda_norm
        self.lambda_grad = lambda_grad

    def multiscale_mse_loss(self, pred, target, scales=[1, 2, 4, 8, 16]):
        loss = 0
        for scale in scales:
            if scale == 1:
                loss += self.mse(pred, target)
            else:
                pool_layer = nn.AvgPool2d(kernel_size=scale, stride=scale)
                p_down = pool_layer(pred)
                t_down = pool_layer(target)
                loss += self.mse(p_down, t_down)
        return loss

    def gradient_loss(self, pred_sin, pred_cos, orientation_field):
        # orientation_field is in radians (0 to pi)
        # We want the gradient of the phase (atan2(sin, cos)) to align with orientation
        # This is complex to compute directly due to wrapping.
        # Simplified approach: Penalize divergence in local flow.
        # Alternatively, we rely on the reconstruction loss implicitly handling this.
        # For this snippet, we will stick to reconstruction + norm for stability.
        return 0.0

    def forward(self, pred, spiral_phasor, target_cont, target_full):
        sin_c = pred[:, 0, :, :]
        cos_c = pred[:, 1, :, :]

        sin_s = spiral_phasor[:, 0, :, :]
        cos_s = spiral_phasor[:, 1, :, :]

        pred_cont = cos_c
        pred_full = cos_c * cos_s - sin_c * sin_s

        loss_cont = self.multiscale_mse_loss(pred_cont, target_cont)
        loss_full = self.multiscale_mse_loss(pred_full, target_full)

        norm_map = cos_c**2 + sin_c**2
        loss_norm = self.mse(norm_map, torch.ones_like(norm_map))

        total_loss = (self.lambda_img * (loss_cont + loss_full)) + (
            self.lambda_norm * loss_norm
        )

        return total_loss, loss_cont, loss_full, loss_norm
