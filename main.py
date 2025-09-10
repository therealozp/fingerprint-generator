from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *
from utils.torch_filter_module import ContinuousFilterLayer

import time
import torch
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def initialize_maps(fprint1):
    """Vectorized initialization (removed margin usage)."""
    fprint1.fill(0)


if __name__ == "__main__":
    # Generate orientation map
    singularity_type = 2
    width = 300
    height = 500
    margin = 0  # kept only for compatibility with existing filter APIs
    padding = 0
    f_print1 = np.zeros((height, width), dtype=np.float32)

    history = []

    core_positions, delta_positions, arch_fact1, arch_fact2, k_arch = (
        init_para_canonical(H=height, W=width, singularity_type=singularity_type)
    )
    initialize_maps(f_print1)
    print("parameter initialization successful.")

    seed_pos(f_print1, height, width, margin, n_seeds=1000)
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

    start = time.time()
    freq_map = sel_n_merg_densitymap(H=height, W=width)

    pre_input = f_print1.copy()
    print(f"Time taken to set filter area:{round(time.time() - start, 2)}s")

    print(f"Time taken for first pass filtering:{round(time.time() - start, 2)}s")

    torch_out = None

    f_print1_tensor = torch.as_tensor(f_print1, dtype=torch.float32)
    freq_map_tensor = torch.as_tensor(freq_map, dtype=torch.long)
    orient_map_tensor = torch.as_tensor(orientation_map, dtype=torch.float32)

    print(freq_map_tensor.shape)
    print(orient_map_tensor.shape)

    flayer = ContinuousFilterLayer(
        31,
        soft_binarize=False,
        binarization_threshold=52,
        temperature=20.0,
    )

    torch_out_t = f_print1_tensor
    for i in range(2):
        with torch.no_grad():
            torch_out_t = flayer(
                torch_out_t,  # seeded image
                orient_map_tensor,  # now 1..180
                freq_map_tensor,  # already 1..100
            )  # tensor

    torch_out = torch_out_t.cpu().numpy()

    plt.imshow(
        np.clip((1 - torch_out / 100.0) * 255, 0, 255).astype(np.uint8), cmap="gray"
    )
    plt.title("Torch FilterLayer")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
