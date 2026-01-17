from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *
from utils.torch_filter_module import ContinuousFilterLayer

import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def initialize_maps(fprint1):
    """Vectorized initialization (removed margin usage)."""
    fprint1.fill(0)


def to_display(img):
    """Convert fingerprint matrix values (0..100) to 8-bit grayscale image for plotting."""
    return np.clip((1 - img / 100.0) * 255, 0, 255).astype(np.uint8)


def plot_difference(current, compare):
    """Plot `current`, `compare`, and their absolute difference as 3 subplots.

    If a global `overlay_points` (array of (row, col) points) exists, plot
    them in red on the `current` subplot.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].imshow(to_display(current), cmap="gray")
    axs[0].set_title("Current")
    axs[0].axis("off")

    # Overlay seed points if provided
    pts = globals().get("overlay_points", None)
    if pts is not None and len(pts) > 0:
        ys, xs = pts[:, 0], pts[:, 1]
        axs[0].scatter(xs, ys, c="red", s=10, marker="o")

    axs[1].imshow(to_display(compare), cmap="gray")
    axs[1].set_title("Compare")
    axs[1].axis("off")

    diff = np.abs(current - compare)
    im = axs[2].imshow(diff, cmap="hot")
    axs[2].set_title("Abs difference")
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def seed_centers_from_mask(seed_img):
    """Return array of (row, col) seed top-left centers from seeded image.

    Seeds are 4x4 blocks set to 100; we detect the top-left pixel of each block
    by selecting mask pixels whose above and left neighbors are False.
    """
    mask = seed_img == 100
    if not np.any(mask):
        return np.zeros((0, 2), dtype=int)
    up = np.roll(mask, 1, axis=0)
    left = np.roll(mask, 1, axis=1)
    up[0, :] = False
    left[:, 0] = False
    top_left = mask & (~up) & (~left)
    pts = np.argwhere(top_left)
    return pts


if __name__ == "__main__":
    # Generate orientation map
    singularity_type = 3
    width = 256
    height = 256
    margin = 0  # kept only for compatibility with existing filter APIs
    padding = 0
    f_print1 = np.zeros((height, width), dtype=np.float32)
    f_print_compare = np.zeros((height, width), dtype=np.float32)

    history = []

    core_positions, delta_positions, arch_fact1, arch_fact2, k_arch = (
        init_para_canonical(H=height, W=width, singularity_type=singularity_type)
    )
    initialize_maps(f_print1)
    print("parameter initialization successful.")

    seed_pos(f_print1, height, width, margin, n_seeds=1000)
    seed_pos(f_print_compare, height, width, margin, n_seeds=1000)

    print("seeding successful.")
    g_cap = set_param_canonical(singularity_type)

    o_map = OrientationMap(
        width,
        height,
        singularity_type,
        delta_positions,
        core_positions,
        g_cap,
        arch_fact1,
        arch_fact2,
        k_arch,
    )

    print("core_positions:", core_positions)
    print("delta_positions:", delta_positions)
    orientation_map = o_map.getOrientationMap()
    print("orientation map generated.")

    freq_map = sel_n_merg_densitymap(H=height, W=width)

    maximum_width_between_ridges = 8.0
    minimum_width_between_ridges = 5.0

    f_min = 1.0 / maximum_width_between_ridges
    f_max = 1.0 / minimum_width_between_ridges
    f_max_over_min = f_max / f_min

    freq_map = f_min * np.pow(
        (f_max_over_min), (freq_map - 1) / 161
    )  # cycles per pixel

    pre_input = f_print1.copy()
    torch_out = None

    f_print1_tensor = torch.as_tensor(f_print1, dtype=torch.float32)
    f_print_compare_tensor = torch.as_tensor(f_print_compare, dtype=torch.float32)

    freq_map_tensor = torch.as_tensor(freq_map, dtype=torch.float32)
    orient_map_tensor = torch.as_tensor(orientation_map, dtype=torch.float32)

    print(freq_map_tensor.shape)
    print(orient_map_tensor.shape)

    print(freq_map.max(), freq_map.min(), freq_map.mean())
    print(orient_map_tensor.max(), orient_map_tensor.min(), orient_map_tensor.mean())

    flayer = ContinuousFilterLayer(
        31,
        soft_binarize=False,
        binarization_threshold=52,
        temperature=20.0,
    )

    torch_out_t = f_print1_tensor
    torch_out_compare_t = f_print_compare_tensor

    for i in range(3):
        with torch.no_grad():
            torch_out_t = flayer(
                torch_out_t,  # seeded image
                orient_map_tensor,  # now 1..180
                freq_map_tensor,  # already 1..100
            )  # tensor

        with torch.no_grad():
            torch_out_compare_t = flayer(
                torch_out_compare_t,  # seeded image
                orient_map_tensor,  # now 1..180
                freq_map_tensor,  # already 1..100
            )  # tensor

    torch_out = torch_out_t.cpu().numpy()
    torch_out_compare = torch_out_compare_t.cpu().numpy()

    # Compute seed centers and store in globals for overlay
    pts = seed_centers_from_mask(f_print1)
    globals()["overlay_points"] = pts

    # Plot seed maps using helper f
    plot_difference(f_print1, f_print_compare)

    # Plot final outputs using helper f (overlay will put seeds on the 'current' subplot)
    plot_difference(torch_out, torch_out_compare)

    plt.subplots(1, 2)
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(torch_out, cmap="gray", origin="lower")

    stride = 4  # plot every 8th pixel for readability
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]

    U = np.cos(np.pi / 2 - orientation_map[::stride, ::stride])
    V = np.sin(np.pi / 2 - orientation_map[::stride, ::stride])
    plt.quiver(
        xs,
        ys,
        U,
        V,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=0.3,
        width=0.003,
        headaxislength=0,
        headlength=0,
        headwidth=0,
    )
    plt.gca().invert_yaxis()
    plt.title("Dense orientation field (theta) as arrows")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    # plot every 8th pixel for readability
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]

    U = np.cos(np.pi / 2 - orientation_map[::stride, ::stride])
    V = np.sin(np.pi / 2 - orientation_map[::stride, ::stride])
    plt.quiver(
        xs,
        ys,
        U,
        V,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=0.3,
        width=0.003,
        headaxislength=0,
        headlength=0,
        headwidth=0,
    )
    plt.gca().invert_yaxis()
    plt.title("Dense orientation field (theta) as arrows")
    plt.axis("off")

    plt.show()
