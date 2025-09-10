from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *
from utils.torch_filter_module import FilterLayer, ContinuousFilterLayer

import time
import torch
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def to_display(img):
    """Convert fingerprint matrix values (0..100) to 8-bit grayscale image for plotting."""
    return np.clip((1 - img / 100.0) * 255, 0, 255).astype(np.uint8)


def stack_filterbank(filterbank_4Dmat):
    """Safely stack nested filterbank lists into a dense numpy array or return None on failure."""
    try:
        return np.stack([np.stack(row, axis=0) for row in filterbank_4Dmat], axis=0)
    except Exception as e:
        print("Failed to stack filterbank for torch FilterLayer:", e)
        return None


def display_filterbank_info(
    filterbank_4Dmat, filter_size_2D, max_filter_size, min_filter_size
):
    """Print compact information about the loaded filterbank."""
    # Avoid printing huge structures; print shape summaries instead
    try:
        fshape = (
            len(filterbank_4Dmat),
            len(filterbank_4Dmat[0]),
            len(filterbank_4Dmat[0][0]),
            len(filterbank_4Dmat[0][0][0]),
        )
    except Exception:
        fshape = None
    print(
        f"filterbank loaded. filterbank shape: {fshape}",
        "filter size shape:",
        filter_size_2D.shape,
        "max filter size:",
        max_filter_size,
        "min filter size:",
        min_filter_size,
    )


def compare_and_visualize(torch_out, numpy_out):
    """Compute basic metrics comparing torch_out and numpy_out and visualize side-by-side.

    Returns a dict of metrics.
    """
    mse = np.mean((torch_out - numpy_out) ** 2)
    mae = np.mean(np.abs(torch_out - numpy_out))
    if np.std(torch_out) > 0 and np.std(numpy_out) > 0:
        corr = np.corrcoef(torch_out.flatten(), numpy_out.flatten())[0, 1]
    else:
        corr = float("nan")
    torch_pos = torch_out > 0
    base_pos = numpy_out > 0
    intersection = np.logical_and(torch_pos, base_pos).sum()
    union = np.logical_or(torch_pos, base_pos).sum()
    jaccard = intersection / union if union > 0 else float("nan")
    accuracy = (torch_pos == base_pos).mean()

    print("FilterLayer vs NumPy filter_image metrics:")
    print(
        f"  MSE:{mse:.2f} MAE:{mae:.2f} Corr:{corr:.3f} Jaccard:{jaccard:.3f} Acc:{accuracy:.3f}"
    )

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(to_display(torch_out), cmap="gray")
    axs[0].set_title("Torch FilterLayer")
    axs[0].axis("off")
    axs[1].imshow(to_display(numpy_out), cmap="gray")
    axs[1].set_title("NumPy filter_image")
    axs[1].axis("off")
    diff_vis = np.clip(np.abs(torch_out - numpy_out), 0, 100)
    axs[2].imshow(diff_vis, cmap="hot")
    axs[2].set_title("Abs Difference")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()

    return {
        "mse": mse,
        "mae": mae,
        "corr": corr,
        "jaccard": jaccard,
        "accuracy": accuracy,
    }


def quiver_plot(mp):
    step = 1

    meshgrid_H, meshgrid_W = mp.shape
    Y, X = np.meshgrid(np.arange(0, meshgrid_H, step), np.arange(0, meshgrid_W, step))

    mp_downscaled = mp[::step, ::step]

    # Compute vector components from orientation
    U = np.cos(mp_downscaled)  # X-component
    V = np.sin(mp_downscaled)  # Y-component

    # Plot the quiver plot
    plt.figure(figsize=(16, 16))
    plt.quiver(
        X,
        Y,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1,  # Reduce scale to make arrows larger
        width=0.0004,  # Increase width to make arrows thicker
        # headlength=0.01,  # Makes arrowheads longer
        headwidth=5,  # Makes arrowheads wider
        color="blue",
    )
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.title("Orientation Map - Quiver Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

    # print(mp)
    np.savetxt("orientation_map.txt", mp, fmt="%f")


def initialize_maps(filt_map, fprint1, fprint2):
    """Vectorized initialization (removed margin usage)."""
    filt_map.fill(0)
    fprint2.fill(100)
    fprint1.fill(0)


def display_fprint(
    f_print1, animate=False, history=None, title="Viewable Fingerprint Image"
):
    img = np.clip((1 - f_print1 / 100.0) * 255, 0, 255).astype(np.uint8)
    if animate and history:
        frames = [
            np.clip((1 - h / 100.0) * 255, 0, 255).astype(np.uint8) for h in history
        ]
        frames.append(img)
        fig, ax = plt.subplots(figsize=(8, 10))
        plt.axis("off")

        def update(k):
            ax.clear()
            ax.imshow(frames[k], cmap="gray")
            ax.set_title(f"Frame {k+1}/{len(frames)}")
            ax.axis("off")
            return (ax,)

        FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
        plt.show()
        return
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_frequency_histogram(matrix):
    flat_values = matrix.flatten()

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_values, bins=100, range=(0, 99), edgecolor="black")
    plt.title("Histogram of Value Distribution")
    plt.xlabel("Value (0–99)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Generate orientation map
    singularity_type = 5
    width = 275
    height = 400
    margin = 0  # kept only for compatibility with existing filter APIs
    padding = 0
    f_print1 = np.zeros((height, width), dtype=np.float32)
    f_print2 = np.zeros_like(f_print1)
    filt_map = np.zeros_like(f_print1)

    history = []

    core_positions, delta_positions, arch_fact1, arch_fact2, k_arch = (
        init_para_canonical(H=height, W=width, singularity_type=singularity_type)
    )
    initialize_maps(filt_map, f_print1, f_print2)
    print("parameter initialization successful.")

    seed_pos(f_print1, filt_map, height, width, margin, n_seeds=1000)
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

    # quiver_plot(orientation_map)

    start = time.time()
    filterbank_4Dmat, filter_size_2D, max_filter_size, min_filter_size = (
        load_gabor_filters()
    )

    print(f"Time taken to load filters:{round(time.time() - start, 2)}s")
    print(
        f"filterbank loaded. filterbank shape: {len(filterbank_4Dmat)}, {len(filterbank_4Dmat[0])}, {len(filterbank_4Dmat[0][0])}, {len(filterbank_4Dmat[0][0][0])}",
        "filter size shape: ",
        filter_size_2D.shape,
        "max filter size: ",
        max_filter_size,
        "min filter size: ",
        min_filter_size,
    )

    start = time.time()
    freq_map = sel_n_merg_densitymap()

    # visualize_frequency_histogram(freq_map)
    # show_density_map(freq_map)

    print(f"Time taken to load density map:{round(time.time() - start, 2)}s")
    print("density map generated.")

    orientation_indices_map, freq_indices_map = pre_filtering(
        orientation_map, freq_map, height, width, margin
    )

    print(np.max(freq_map), np.min(freq_map), np.mean(freq_map))
    print(
        np.max(freq_indices_map),
        np.min(freq_indices_map),
        np.mean(freq_indices_map),
    )

    print(orientation_map)
    print(orientation_indices_map)

    # Preprocess like first-pass (binarize@65 then normalize) and apply region mask
    pre_input = f_print1.copy()
    # Apply region mask (after we build filt_map below) later for fairness

    start = time.time()
    filt_map = set_filter_area(
        filt_map=filt_map,
        H=height,
        W=width,
        margin=margin,
        max_filter_size=max_filter_size,
    )
    print(f"Time taken to set filter area:{round(time.time() - start, 2)}s")

    history.append(f_print1.copy())

    start = time.time()
    f_print1 = filter_image(
        f_print1=f_print1,
        # filt_map=filt_map,
        freq_ind=freq_indices_map,
        orient_ind=orientation_indices_map,
        filter_size_2d=filter_size_2D,
        filterbank=filterbank_4Dmat,
        H=height,
        W=width,
        margin=margin,
        max_filter_size=max_filter_size,
        binarization_threshold=51,
    )
    # print("first pass filtering successful.")
    # display_fprint(f_print1, animate=False, history=history)
    # history.append(f_print1.copy())
    f_print1_alt = f_print1.copy()

    print(f"Time taken for first pass filtering:{round(time.time() - start, 2)}s")

    # --- Torch FilterLayer replication of filter_image (single pass) ---
    # Build dense 4D np array [F,O,K,K]
    try:
        fb_array = np.stack([np.stack(row, axis=0) for row in filterbank_4Dmat], axis=0)
    except Exception as e:
        print("Failed to stack filterbank for torch FilterLayer:", e)
        fb_array = None

    torch_out = None
    torch_out_t = None

    f_print1_tensor = torch.as_tensor(f_print1, dtype=torch.float32)
    freq_map_tensor = torch.as_tensor(freq_map, dtype=torch.long)
    orient_idx_1b = orientation_indices_map + 1
    orient_idx_1b = torch.as_tensor(orient_idx_1b, dtype=torch.long)

    if fb_array is not None:
        flayer = ContinuousFilterLayer(
            max_filter_size,
            soft_binarize=False,
            binarization_threshold=51,
            temperature=20.0,
        )

        # Orientation indices need to be 1-based
        with torch.no_grad():
            torch_out_t = flayer(
                f_print1_tensor,  # seeded image
                orient_idx_1b,  # now 1..180
                freq_map_tensor,  # already 1..100
                show_kernels=True,
                filterbank_4d=fb_array,
            )  # tensor
            torch_out = torch_out_t.cpu().numpy()

        # Metrics vs NumPy version
        mse = np.mean((torch_out - f_print1) ** 2)
        mae = np.mean(np.abs(torch_out - f_print1))
        if np.std(torch_out) > 0 and np.std(f_print1) > 0:
            corr = np.corrcoef(torch_out.flatten(), f_print1.flatten())[0, 1]
        else:
            corr = float("nan")
        torch_pos = torch_out > 0
        base_pos = f_print1 > 0
        intersection = np.logical_and(torch_pos, base_pos).sum()
        union = np.logical_or(torch_pos, base_pos).sum()
        jaccard = intersection / union if union > 0 else float("nan")
        accuracy = (torch_pos == base_pos).mean()
        print("FilterLayer vs NumPy filter_image metrics:")
        print(
            f"  MSE:{mse:.2f} MAE:{mae:.2f} Corr:{corr:.3f} Jaccard:{jaccard:.3f} Acc:{accuracy:.3f}"
        )

        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        axs[0].imshow(
            np.clip((1 - torch_out / 100.0) * 255, 0, 255).astype(np.uint8), cmap="gray"
        )
        axs[0].set_title("Torch FilterLayer")
        axs[0].axis("off")
        axs[1].imshow(
            np.clip((1 - f_print1 / 100.0) * 255, 0, 255).astype(np.uint8), cmap="gray"
        )
        axs[1].set_title("NumPy filter_image")
        axs[1].axis("off")
        diff_vis = np.clip(np.abs(torch_out - f_print1), 0, 100)
        axs[2].imshow(diff_vis, cmap="hot")
        axs[2].set_title("Abs Difference")
        axs[2].axis("off")
        plt.tight_layout()
        plt.show()

    filt_map = set_filter_area(
        filt_map=filt_map,
        H=height,
        W=width,
        margin=margin,
        max_filter_size=max_filter_size,
    )
    plt.imshow(filt_map, cmap="gray")
    plt.title("Filter Map")
    plt.colorbar()
    plt.show()
    b_thresh = 50

    display_fprint(f_print1_alt, animate=False, history=history)
