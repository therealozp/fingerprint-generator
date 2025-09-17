import torch
import matplotlib.pyplot as plt
import numpy as np
import math


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
