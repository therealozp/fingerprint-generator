import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np


class FilterLayer(torch.nn.Module):
    def __init__(
        self, K, soft_binarize=True, binarization_threshold=55.0, temperature=20.0
    ):
        """
        soft_binarize: use a smooth step (sigmoid) so the pipeline is differentiable.
        binarization_threshold: threshold in [0,100].
        temperature: larger => sharper transition near the threshold.
        """
        super().__init__()
        assert K % 2 == 1, "Use an odd K for symmetric padding."
        self.K = int(K)
        self.pad = K // 2

        self.soft_binarize = soft_binarize
        self.binarization_threshold = float(binarization_threshold)
        self.temperature = float(temperature)

    @staticmethod
    def _to_tensor(x, ref, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=dtype)
        return torch.as_tensor(x, device=ref.device, dtype=dtype)

    @staticmethod
    def _to_long(x, ref):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=torch.long)
        return torch.as_tensor(x, device=ref.device, dtype=torch.long)

    @staticmethod
    def _normalize_0_100(x, eps=1e-8):
        x_min = torch.amin(x)
        x_max = torch.amax(x)
        return 100.0 * (x - x_min) / (torch.clamp(x_max - x_min, min=eps))

    def _soft_binarize(self, x_0_100):
        # Map to [0,1], apply sigmoid at threshold, return to [0,100]
        thr01 = self.binarization_threshold / 100.0
        x01 = x_0_100 / 100.0
        y01 = torch.sigmoid(self.temperature * (x01 - thr01))
        return 100.0 * y01

    def forward(
        self,
        f_print1,  # [H_total, W_total]
        freq_ind,  # [H_total, W_total], 1-based
        orient_ind,  # [H_total, W_total], 1-based
        filterbank,  # [F, O, K, K]
        H,
        W,
        margin,
    ):
        # Anchor device/dtype
        if not isinstance(f_print1, torch.Tensor):
            f_print1 = torch.as_tensor(f_print1, dtype=torch.float32)
        device = f_print1.device
        f_print1 = f_print1.to(torch.float32)

        freq_ind = torch.as_tensor(freq_ind, device=device, dtype=torch.long)
        orient_ind = torch.as_tensor(orient_ind, device=device, dtype=torch.long)
        filterbank = torch.as_tensor(filterbank, device=device, dtype=torch.float32)

        print(freq_ind.shape)
        print(orient_ind.shape)

        H_total, W_total = f_print1.shape
        assert H_total == H + margin and W_total == W + margin

        Fdim, Odim, K1, K2 = filterbank.shape
        assert K1 == self.K and K2 == self.K, "filterbank must match fixed K"

        # Extract all K×K patches with symmetric zero padding
        img4d = f_print1.unsqueeze(0).unsqueeze(0)  # [1,1,H_total,W_total]
        patches = F.unfold(
            img4d, kernel_size=self.K, padding=self.pad, stride=1
        )  # [1,K*K,H_total*W_total]
        patches = patches[0].transpose(0, 1)  # [N, K*K], N = H_total*W_total

        # Per-pixel kernel selection
        f0 = (freq_ind - 1).clamp(0, Fdim - 1)
        o0 = (orient_ind - 1).clamp(0, Odim - 1)
        idx_flat = (f0 * Odim + o0).reshape(-1)  # [N]

        bank_flat = filterbank.reshape(Fdim * Odim, -1)  # [F*O, K*K]
        kernels = bank_flat.index_select(0, idx_flat)  # [N, K*K]

        # Dot product per pixel
        y_flat = (patches * kernels).sum(dim=1)  # [N]
        y_img = y_flat.view(H_total, W_total)  # [H_total, W_total]
        out = y_img

        # Post-processing: normalize -> (soft) binarize -> normalize
        out = self._normalize_0_100(out)
        out = self._soft_binarize(out)
        out = self._normalize_0_100(out)

        return out


class ContinuousFilterLayer(nn.Module):
    def __init__(
        self,
        K,
        soft_binarize=True,
        binarization_threshold=55.0,
        temperature=20.0,
        sigma=6.0,
        gamma=1,
        phase=0.0,
    ):
        super().__init__()
        assert K % 2 == 1
        self.K = K
        self.pad = K // 2
        # Precompute coordinate grid as buffers (no grad, moves with device)
        half = K // 2
        y = torch.arange(-half, half + 1, dtype=torch.float32)
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        YY, XX = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("XX", XX)  # [K,K]
        self.register_buffer("YY", YY)  # [K,K]

        # Default params (can be overridden per forward)
        self.sigma_default = float(sigma)
        self.gamma_default = float(gamma)
        self.phase_default = float(phase)

        self.soft_binarize = soft_binarize
        self.binarization_threshold = float(binarization_threshold)
        self.temperature = float(temperature)

    @staticmethod
    def _to_tensor(x, ref, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=dtype)
        return torch.as_tensor(x, device=ref.device, dtype=dtype)

    @staticmethod
    def _to_long(x, ref):
        if isinstance(x, torch.Tensor):
            return x.to(device=ref.device, dtype=torch.long)
        return torch.as_tensor(x, device=ref.device, dtype=torch.long)

    @staticmethod
    def _normalize_0_100(x, eps=1e-8):
        x_min = torch.amin(x)
        x_max = torch.amax(x)
        return 100.0 * (x - x_min) / (torch.clamp(x_max - x_min, min=eps))

    def _soft_binarize(self, x_0_100):
        # Map to [0,1], apply sigmoid at threshold, return to [0,100]
        thr01 = self.binarization_threshold / 100.0
        x01 = x_0_100 / 100.0
        y01 = torch.sigmoid(self.temperature * (x01 - thr01))
        return 100.0 * y01

    def forward(
        self,
        img,
        orient_ind,
        freq_ind,
        sigma=None,
        gamma=None,
        phase=None,
        show_kernels=False,
        num_kernels=16,
        cmap="gray",
        **kwargs,
    ):
        """
        img:        [H, W] float tensor
        theta_map:  [H, W] radians
        freq_map:   [H, W] cycles per pixel
        sigma,gamma,phase: scalars or [H,W] tensors (optional)
        returns:    [H, W] filtered image
        """
        H, W = img.shape
        N = H * W
        img4d = img.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        patches = F.unfold(
            img4d, kernel_size=self.K, padding=self.pad, stride=1
        )  # [1,K*K,N]
        patches = patches[0].transpose(0, 1).view(N, self.K, self.K)  # [N,K,K]

        # Move grids to image device/dtype
        XX = self.XX.to(img.device, img.dtype)
        YY = self.YY.to(img.device, img.dtype)

        def as_param(p, default):
            if p is None:
                p = default
            if torch.is_tensor(p):
                return p.to(device=img.device, dtype=img.dtype).view(N, 1, 1)
            else:
                return torch.tensor(p, device=img.device, dtype=img.dtype).view(1, 1, 1)

        # if using orient_ind_map
        # theta_map = (torch.pi / 2.0) - torch.deg2rad(orient_ind)

        # if using orient_map
        theta_map = (torch.pi / 2.0) - orient_ind

        f_min = 0.075
        f_max = 0.33
        f_max_over_min = f_max / f_min

        freq_map = f_min * torch.pow(
            (f_max_over_min), (freq_ind - 1) / 161
        )  # cycles per pixel

        theta = theta_map.to(img.device, img.dtype).view(N, 1, 1)  # [N,1,1]
        freq = freq_map.to(img.device, img.dtype).view(N, 1, 1)  # [N,1,1]

        sigma = as_param(sigma, self.sigma_default)  # [N,1,1] or [1,1,1]
        gamma = as_param(gamma, self.gamma_default)
        phase = as_param(phase, self.phase_default)

        # Rotate coordinates per pixel (broadcast [N,1,1] with [K,K] -> [N,K,K])
        ct, st = torch.cos(theta), torch.sin(theta)
        x_theta = XX * ct + YY * st
        y_theta = -XX * st + YY * ct

        # Gabor pieces
        gauss = torch.exp(
            -0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / (sigma**2)
        )  # [N,K,K]
        sinus = torch.cos(2.0 * torch.pi * freq * x_theta + phase)  # [N,K,K]

        kernel = gauss * sinus  # [N,K,K]

        # Fused multiply-accumulate with patches
        y = (patches * kernel).sum(dim=(1, 2))
        print(y.shape)
        y = y.view(H, W)

        y = self._normalize_0_100(y)
        y = self._soft_binarize(y)
        y = self._normalize_0_100(y)

        # test hard-binarize
        y = torch.where(y >= 55.0, 100.0, 0.0)

        return y
