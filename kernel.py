import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


from gabor_torch import generate_custom_gabor_torch


def binarize2Df_print_torch(f_print, threshold):
    """
    Binarizes a 2D or 4D torch tensor using a threshold.
    Values > threshold become 100, others become 0.
    """
    return torch.where(
        f_print > threshold,
        torch.tensor(100.0, device=f_print.device),
        torch.tensor(0.0, device=f_print.device),
    )


def normalize2Df_print_torch(f_print):
    """
    Normalizes a 2D or 4D torch tensor to the range [0, 100].
    Handles [H, W] or [B, C, H, W].
    """
    minval = torch.min(f_print)
    maxval = torch.max(f_print)

    # Shift values if negative
    if minval < 0.0:
        f_print = f_print + torch.abs(minval)
    else:
        f_print = f_print - minval

    # Recompute max after shifting
    maxval = torch.max(f_print)
    if maxval != 0:
        f_print = (f_print / maxval) * 100.0

    return f_print


class DifferentiableGaborKernel(nn.Module):
    def __init__(self, filt_map, sigma=6.0, gamma_0=1.0, gamma_delta=0.6):
        super().__init__()

        self.filt_map = filt_map.copy()

        self.sigma = sigma
        self.gamma_0 = gamma_0
        self.gamma_delta = gamma_delta
        self.kernel_size = 31  # default value right now
        self.binarization_threshold = 55  # default value right now

    def set_filter_area(self, H, W, margin, max_filter_size):

        pad = max_filter_size // 2

        # — Horizontal pass —
        i = 0
        while i < H + margin:
            flag_in = False
            j = 0
            while j < W + margin:
                if self.filt_map[i][j] == 1 and not flag_in:
                    for k in range(j - pad, j):
                        if 0 <= k < W + margin:
                            self.filt_map[i][k] = 1
                    flag_in = True

                elif self.filt_map[i][j] == 0 and flag_in:
                    for k in range(j, j + pad + 1):
                        if 0 <= k < W + margin:
                            self.filt_map[i][k] = 1
                    j += pad + 1
                    flag_in = False
                    continue

                j += 1
            i += 1

        # — Vertical pass (same pattern) —
        j = 0
        while j < W + margin:
            flag_in = False
            i = 0
            while i < H + margin:
                if self.filt_map[i][j] == 1 and not flag_in:
                    for k in range(i - pad, i):
                        if 0 <= k < H + margin:
                            self.filt_map[k][j] = 1
                    flag_in = True

                elif self.filt_map[i][j] == 0 and flag_in:
                    for k in range(i, i + pad + 1):
                        if 0 <= k < H + margin:
                            self.filt_map[k][j] = 1
                    i += pad + 1
                    flag_in = False
                    continue

                i += 1
            j += 1

        plt.imshow(self.filt_map, cmap="gray")
        plt.title("Expanded Filter Map")
        plt.colorbar()
        plt.show()

    def forward(self, fprint, freq_map, theta_map, phase_map=None):
        padding = self.kernel_size // 2

        H, W = fprint.shape
        device = fprint.device
        output = fprint.clone()

        start = time.time()
        # self.set_filter_area(H, W, 30, self.kernel_size)
        # print("Time taken to set filter area: ", time.time() - start)

        i_start = padding
        j_start = padding
        i_end = int(H - np.ceil(self.kernel_size / 2.0))
        j_end = int(W - np.ceil(self.kernel_size / 2.0))

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                freq_idx = int(freq_map[i, j])
                theta_idx = int(theta_map[i, j])
                theta = (theta_idx / 180) * torch.pi
                freq = 0.025 + 0.0015 * freq_idx

                kernel = generate_custom_gabor_torch(
                    size=self.kernel_size,
                    theta=theta,
                    freq_0=freq,
                    freq_delta=freq / 3,
                    sigma=self.sigma,
                    gamma_0=self.gamma_0,
                    gamma_delta=self.gamma_delta,
                )

                pad = self.kernel_size // 2
                i_min = max(i - pad, 0)
                i_max = min(i + pad + 1, H)
                j_min = max(j - pad, 0)
                j_max = min(j + pad + 1, W)

                patch = fprint[i_min:i_max, j_min:j_max]

                f_i_min = max(pad - (i - i_min), 0)
                f_i_max = f_i_min + (i_max - i_min)
                f_j_min = max(pad - (j - j_min), 0)
                f_j_max = f_j_min + (j_max - j_min)

                kernel_crop = kernel[f_i_min:f_i_max, f_j_min:f_j_max]
                output[i, j] = torch.sum(patch * kernel_crop)

        output = normalize2Df_print_torch(output)
        output = binarize2Df_print_torch(output, self.binarization_threshold)

        return output
