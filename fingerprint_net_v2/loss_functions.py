import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sobel_kernels(size):
    half = size // 2
    i = torch.arange(-half, half + 1).view(-1, 1).float()
    j = torch.arange(-half, half + 1).view(1, -1).float()

    denom = i**2 + j**2 + 1e-6  # avoid division by zero
    gx = j / denom
    gy = i / denom

    return gx.unsqueeze(0).unsqueeze(0), gy.unsqueeze(0).unsqueeze(0)


class MultiScaleEdgeLoss(nn.Module):
    def __init__(self, scales=[3, 5, 7]):
        super().__init__()
        for i, s in enumerate(scales):
            gx, gy = make_sobel_kernels(s)
            self.register_buffer(f"gx_{i}", gx)
            self.register_buffer(f"gy_{i}", gy)
        self.num_scales = len(scales)

    def edge_maps(self, img, gx_kernel, gy_kernel):
        gx = F.conv2d(img, gx_kernel, padding=gx_kernel.shape[-1] // 2)
        gy = F.conv2d(img, gy_kernel, padding=gy_kernel.shape[-1] // 2)

        return gx, gy

    def forward(self, pred, target):
        loss = 0.0
        for i in range(self.num_scales):
            gx_k = getattr(self, f"gx_{i}")
            gy_k = getattr(self, f"gy_{i}")
            pred_gx, pred_gy = self.edge_maps(pred, gx_k, gy_k)
            tgt_gx, tgt_gy = self.edge_maps(target, gx_k, gy_k)
            loss += F.mse_loss(pred_gx, tgt_gx) + F.mse_loss(pred_gy, tgt_gy)
        return loss / self.num_scales


class MSSSIMLoss(nn.Module):
    def __init__(self, scales=5, window_size=11, sigma=1.5, weights=None):
        super().__init__()
        self.scales = scales
        self.window_size = window_size
        # default weights from Wang et al. 2003
        if weights is None:
            self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.weights = torch.tensor(weights)
        assert len(self.weights) == scales
        self.register_buffer("window", self._gaussian_window(window_size, sigma))

    def _gaussian_window(self, size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window = g.outer(g)
        return window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def _ssim_components(self, x, y):
        C1 = 0.01**2
        C2 = 0.03**2
        pad = self.window_size // 2
        window = self.window.to(x.device)

        mu_x = F.conv2d(x, window, padding=pad)
        mu_y = F.conv2d(y, window, padding=pad)
        mu_x2 = mu_x**2
        mu_y2 = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

        luminance = (2 * mu_xy + C1) / (mu_x2 + mu_y2 + C1)
        contrast_structure = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)

        return luminance, contrast_structure

    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        mcs_list = []

        x, y = pred, target
        for i in range(self.scales):
            luminance, cs = self._ssim_components(x, y)
            if i == self.scales - 1:
                # at the finest scale, use full SSIM (luminance * cs)
                ssim_map = luminance * cs
                mcs_list.append(ssim_map.mean())
            else:
                mcs_list.append(cs.mean())
                # downsample for next scale
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        # product of (cs at each scale) * (ssim at finest scale), weighted by exponents
        score = torch.stack(
            [F.relu(mcs_list[i]) ** self.weights[i] for i in range(self.scales)]
        ).prod()

        return 1.0 - score


class FingerprintLossv2(nn.Module):
    def __init__(self, lambda_img=1.0, lambda_mse=1.0, lambda_grad=1.0):
        super().__init__()

        self.mse = nn.MSELoss()
        self.edge_loss = MultiScaleEdgeLoss()
        self.mssim = MSSSIMLoss()

        self.lambda_mse = lambda_mse
        self.lambda_img = lambda_img
        self.lambda_grad = lambda_grad

    def mse_loss(self, pred, target):
        return self.mse(pred, target)

    def continuous_phasor_loss(self, pred_sin, pred_cos, cont_sin, cont_cos):
        cos_delta = pred_cos * cont_cos + pred_sin * cont_sin
        sin_delta = pred_cos * cont_sin - pred_sin * cont_cos
        angle_error = torch.atan2(sin_delta, cos_delta)
        return (angle_error**2).mean()

    def forward(self, pred, spiral_phasor, cos_cont, cos_full, sin_cont, sin_full):
        sin_c = pred[:, 0:1, :, :]
        cos_c = pred[:, 1:2, :, :]

        sin_s = spiral_phasor[:, 0:1, :, :]
        cos_s = spiral_phasor[:, 1:2, :, :]

        pred_cont = cos_c
        pred_full = cos_c * cos_s - sin_c * sin_s

        loss_cont = self.mse_loss(pred_cont, cos_cont)
        loss_full = self.mse_loss(pred_full, cos_full)

        loss_sobel_cont = self.edge_loss(pred_cont, cos_cont)
        loss_sobel_full = self.edge_loss(pred_full, cos_full)

        ssim_loss_cont = self.mssim(pred_cont, cos_cont)
        ssim_loss_full = self.mssim(pred_full, cos_full)

        phasor_loss = self.continuous_phasor_loss(sin_c, cos_c, sin_cont, cos_cont)
        total_loss = (
            self.lambda_mse * (loss_cont + loss_full)
            + self.lambda_img * (ssim_loss_cont + ssim_loss_full)
            + self.lambda_grad * (loss_sobel_cont + loss_sobel_full)
            + phasor_loss
        )

        return total_loss, {
            "mse_cont": loss_cont,
            "mse_full": loss_full,
            "ssim_cont": ssim_loss_cont,
            "ssim_full": ssim_loss_full,
            "sobel_cont": loss_sobel_cont,
            "sobel_full": loss_sobel_full,
            "phasor": phasor_loss,
        }
