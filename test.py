from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *

import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


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
    """Initialize maps with default starting values.

    All arrays are assumed preallocated with identical shape (H, W).
    This replaces the previous nested-loop version (much faster & clearer).
    """
    filt_map.fill(0)
    fprint2.fill(100)
    fprint1.fill(0)  # was 100 for white image and black dot noise


def display_fprint(f_print1, animate=False, history=None):
    """Display fingerprint image (static or animated).

    The earlier interface required (height, width, margin); we now infer them
    directly from the array shape and have removed the confusing margin/padding.
    """
    H, W = f_print1.shape
    viewable_image = np.clip((1 - (f_print1 / 100.0)) * 255, 0, 255).astype(np.uint8)

    if animate and history:
        # Vectorized conversion for all history frames
        frames = [
            np.clip((1 - (img / 100.0)) * 255, 0, 255).astype(np.uint8)
            for img in history
        ]
        frames.append(viewable_image)  # ensure current state included

        fig, ax = plt.subplots(figsize=(8, 10))
        plt.title("Fingerprint Generation Progress")
        plt.axis("off")

        def update(frame_num):
            ax.clear()
            ax.imshow(frames[frame_num], cmap="gray")
            ax.set_title(f"Frame {frame_num+1}/{len(frames)}")
            ax.axis("off")
            return (ax,)

        FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
        plt.show()
        return

    plt.imshow(viewable_image, cmap="gray")
    plt.title("Viewable Fingerprint Image")
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
    # Core dimensions (final desired image size)
    singularity_type = 2
    width = 275
    height = 400

    # Margin & padding removed for clarity (set to 0); kept variables in case
    # downstream functions still require a parameter slot.
    margin = 0
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
    )  # margin currently 0; consider refactoring pre_filtering next

    print(orientation_map.shape)
    print(orientation_indices_map.shape)
    print(filterbank_4Dmat.shape)
