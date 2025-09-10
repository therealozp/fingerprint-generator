import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np


def show_random_kernels(
    kernel,
    num_kernels=16,
    cmap="gray",
    figsize=(6, 6),
    seed=None,
    angles=None,
    freq_inds=None,
    freqs=None,
    filterbank_4d=None,
):
    """Display a random sample of kernels.

    kernel: Tensor of shape [N, K, K]
    num_kernels: how many kernels to display (will clamp to N)
    angles/freq_inds/freqs: optional per-kernel metadata arrays (length N)
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    if not torch.is_tensor(kernel):
        kernel = torch.as_tensor(kernel)

    N = kernel.shape[0]
    if N == 0:
        print("no kernels to show")
        return

    num = min(int(num_kernels), N)
    idx = torch.randperm(N)[:num]
    sample = kernel[idx].cpu().detach().numpy()

    # gather metadata if provided
    angle_vals = None
    freq_ind_vals = None
    freq_vals = None
    if angles is not None:
        a = (
            angles.cpu().detach().numpy().reshape(-1)
            if torch.is_tensor(angles)
            else (np.asarray(angles).reshape(-1))
        )
        angle_vals = a[idx.cpu().numpy()]
    if freq_inds is not None:
        fi = (
            freq_inds.cpu().detach().numpy().reshape(-1)
            if torch.is_tensor(freq_inds)
            else (np.asarray(freq_inds).reshape(-1))
        )
        freq_ind_vals = fi[idx.cpu().numpy()]
    if freqs is not None:
        f = (
            freqs.cpu().detach().numpy().reshape(-1)
            if torch.is_tensor(freqs)
            else (np.asarray(freqs).reshape(-1))
        )
        freq_vals = f[idx.cpu().numpy()]

    # If a 4D discrete filterbank is provided (or available module-level),
    # we'll display the generated kernel alongside the discrete kernel
    # selected by (freq_idx, orient_idx). The right-hand kernel will be
    # plotted when freq_inds and/or freq_vals are provided.
    fb = (
        filterbank_4d
        if filterbank_4d is not None
        else globals().get("filterbank_4Dmat", None)
    )
    has_fb = fb is not None

    # Prepare figure layout: single cell per sample. Each cell will contain
    # a horizontally concatenated image [generated | discrete] when discrete
    # filters are available, otherwise just the generated kernel.
    cols = int(math.ceil(math.sqrt(num)))
    rows = int(math.ceil(num / cols))
    plt.figure(figsize=(figsize[0] * cols, figsize[1] * rows))

    def _normalize_img(a):
        a = a.astype(np.float32)
        mn = a.min()
        a = a - mn
        mx = a.max()
        if mx <= 0:
            return a
        return a / (mx + 1e-8)

    def _fit_to(target_shape, arr):
        # center-crop or pad arr to target_shape (H,W)
        h_t, w_t = target_shape
        h, w = arr.shape
        # crop
        if h > h_t:
            start_h = (h - h_t) // 2
            arr = arr[start_h : start_h + h_t, :]
            h = h_t
        if w > w_t:
            start_w = (w - w_t) // 2
            arr = arr[:, start_w : start_w + w_t]
            w = w_t
        # pad
        pad_h = max(0, h_t - h)
        pad_w = max(0, w_t - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            arr = np.pad(
                arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant"
            )
        return arr

    for i in range(num):
        gen = sample[i]
        title_parts = []
        if angle_vals is not None:
            title_parts.append(f"ang={angle_vals[i]:.1f}\u00b0")
        if freq_ind_vals is not None:
            title_parts.append(f"f_idx={int(freq_ind_vals[i])}")
        if freq_vals is not None:
            title_parts.append(f"freq={freq_vals[i]:.3f}")

        if has_fb:
            try:
                # pick discrete filter as before
                if freq_ind_vals is not None and angle_vals is not None:
                    f_idx = int(freq_ind_vals[i])
                    o_idx = int(angle_vals[i])
                elif freq_ind_vals is not None:
                    f_idx = int(freq_ind_vals[i])
                    o_idx = 0
                else:
                    f_idx = 0
                    o_idx = 0

                fb_arr = np.asarray(fb)
                f_idx = max(0, min(f_idx, fb_arr.shape[0] - 1))
                o_idx = max(0, min(o_idx, fb_arr.shape[1] - 1))
                discrete = np.asarray(fb_arr[f_idx][o_idx]).astype(np.float32)
                # fit discrete to generated kernel shape
                discrete = _fit_to(gen.shape, discrete)
                # normalize both and concat horizontally
                gen_n = _normalize_img(gen)
                disc_n = _normalize_img(discrete)
                combined = np.concatenate([gen_n, disc_n], axis=1)
                plt.subplot(rows, cols, i + 1)
                plt.imshow(combined, cmap=cmap)
                if title_parts:
                    plt.title(", ".join(title_parts), fontsize=8)
                plt.axis("off")
            except Exception as e:
                plt.subplot(rows, cols, i + 1)
                plt.text(0.5, 0.5, f"err:{e}", ha="center")
                plt.axis("off")
        else:
            gen_n = _normalize_img(gen)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(gen_n, cmap=cmap)
            if title_parts:
                plt.title(", ".join(title_parts), fontsize=8)
            plt.axis("off")

    plt.suptitle(f"Sampled {num} kernels (continuous | discrete)")
    plt.tight_layout()
    plt.show()


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

        # Optional visualization: show a random sample of the per-pixel kernels
        # before the multiply-accumulate is performed. This is useful to
        # quickly inspect whether the generated Gabor kernels look correct.
        # Call with show_kernels=True when invoking forward, or call the
        # method `show_random_kernels` directly. Note: plt.show() may block.
        if show_kernels:
            # show a small sample (this moves data to CPU and detaches)
            try:
                show_random_kernels(
                    kernel,
                    num_kernels=num_kernels,
                    cmap=cmap,
                    angles=orient_ind,
                    freq_inds=freq_ind,
                    freqs=freq_map,
                    filterbank_4d=kwargs.get("filterbank_4d", None),
                )
            except Exception as e:
                print(f"failed to show kernels: {e}")

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
