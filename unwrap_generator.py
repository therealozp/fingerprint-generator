import os
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *
from utils.unwrapping import unwrap_with_smart_cuts


def generate_surface(
    height: int,
    width: int,
    *,
    min_gauss_sigma: float = 5.0,
    max_gauss_sigma: float = 20.0,
    scaling_range: Tuple[float, float] = (10.0, 30.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    raw_noise = rng.standard_normal((height, width))
    sigma = float(rng.uniform(min_gauss_sigma, max_gauss_sigma))
    surface = gaussian_filter(raw_noise, sigma=sigma)

    std = surface.std()
    if std == 0:
        std = 1.0
    scale = float(rng.uniform(*scaling_range))
    surface = (surface - surface.mean()) / std * scale
    return surface


def wrap_phase(unwrapped_phase: np.ndarray) -> np.ndarray:
    """Wrap a continuous phase into [-pi, pi]."""
    return np.mod(unwrapped_phase + 2 * np.pi, np.pi)


def add_noise(
    wrapped_phase: np.ndarray,
    *,
    noise_level: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(wrapped_phase.shape) * noise_level
    noisy = wrapped_phase + noise
    return wrap_phase(noisy)


def generate_sample(
    height: int,
    width: int,
    *,
    noise_level: float = 0.1,
    min_gauss_sigma: float = 5.0,
    max_gauss_sigma: float = 20.0,
    scaling_range: Tuple[float, float] = (10.0, 30.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    gt = generate_surface(
        height,
        width,
        min_gauss_sigma=min_gauss_sigma,
        max_gauss_sigma=max_gauss_sigma,
        scaling_range=scaling_range,
        rng=rng,
    )
    wrapped = wrap_phase(gt)
    noisy_input = add_noise(wrapped, noise_level=noise_level, rng=rng)
    return noisy_input, gt


def generate_arch(width, height):
    singularity_type = 1

    core_positions, delta_positions, arch_fact1, arch_fact2, k_arch = (
        init_para_canonical(H=height, W=width, singularity_type=singularity_type)
    )
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

    orientation_map = o_map.getOrientationMap()
    unwrapped_orientation = unwrap_with_smart_cuts(
        orientation_map,
        mask=np.ones_like(orientation_map, dtype=bool),
        singularities=[],
    )

    return orientation_map, unwrapped_orientation


# --- Execution Example ---
from tqdm import tqdm

if __name__ == "__main__":
    H, W = 256, 256
    SAMPLES = 10000
    OUTPUT_DIR = "./phase_data_0_pi"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "wrapped"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "ground_truth"), exist_ok=True)

    for i in tqdm(range(SAMPLES)):
        rng = np.random.default_rng()
        random_number = rng.random()

        if random_number < 0.5:
            noisy_input, gt = generate_sample(
                H,
                W,
                noise_level=0.1,
                min_gauss_sigma=30.0,
                max_gauss_sigma=50.0,
                scaling_range=(1.0, 4.5),
                rng=rng,
            )

            np.save(os.path.join(OUTPUT_DIR, "wrapped", f"{i}.npy"), noisy_input)
            np.save(os.path.join(OUTPUT_DIR, "ground_truth", f"{i}.npy"), gt)
        else:
            orientation_map, unwrapped_orientation = generate_arch(W, H)
            np.save(os.path.join(OUTPUT_DIR, "wrapped", f"{i}.npy"), orientation_map)
            np.save(
                os.path.join(OUTPUT_DIR, "ground_truth", f"{i}.npy"),
                unwrapped_orientation,
            )
