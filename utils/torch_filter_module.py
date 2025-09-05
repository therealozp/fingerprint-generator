import torch
import torch.nn.functional as F


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
