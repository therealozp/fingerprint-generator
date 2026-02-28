from utils.orientation_map import *
from utils.generation import *
from utils.filters import *
from utils.density_map import *
from utils.unwrapping import unwrap_with_smart_cuts, unwrap_with_monotonic_cuts
from utils.unwrap_smooth import unwrap_with_curved_cuts

import numpy as np
from collections import deque
from PIL import Image
import os

start_index = 0
to_generate = 7500


def select_and_merge_density_maps(width, height):
    freq1 = load_random_density_map()
    freq2 = load_random_density_map()

    # Possibly load a third map
    flag_3 = rand() > 0.5
    freq3 = load_random_density_map() if flag_3 else None

    # Merge maps
    if flag_3:
        freq = (freq1 + freq2 + freq3) / 3 / 255.0
    else:
        freq = (freq1 + freq2) / 2 / 255.0

    # Resize to target dimensions, normalize to [0, 1]
    f_den = cv2.resize(freq, (width, height), interpolation=cv2.INTER_LINEAR)
    f_den = (f_den - np.min(f_den)) / (np.max(f_den) - np.min(f_den))

    return f_den


def reconstruct_continuous_phase(
    unwrapped_orientation, mask, block_size=8, freq_map=None, f=0.12, initial_phase=0.0
):
    magnitude = 2 * np.pi * freq_map if freq_map is not None else f * 2 * np.pi

    complex_orientation = np.exp(1j * (unwrapped_orientation + np.pi / 2))
    G_complex = magnitude * complex_orientation

    G_cx = np.real(G_complex)
    G_cy = np.imag(G_complex)

    # 2. Initialize Phase Offset P (Eq 14)
    rows, cols = unwrapped_orientation.shape
    P = np.zeros((rows, cols))
    visited = np.zeros((rows, cols), dtype=bool)

    start_node = None
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                start_node = (r, c)
                break
        if start_node:
            break

    if not start_node:
        return np.zeros((rows * block_size, cols * block_size))

    # BFS Initialization
    queue = deque([start_node])
    visited[start_node] = True
    P[start_node] = initial_phase  # Assumption: P(start) = 0 [cite: 35]

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        # [cite_start]Check all 4 connected neighbors [cite: 35]
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc

            # Boundary and Mask Check
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc]:
                if not visited[nr, nc]:
                    if dr == -1:  # Neighbor is Above
                        # Shared border: y = r * block_size
                        border_y = np.full(block_size, r * block_size)
                        border_x = np.arange(c * block_size, (c + 1) * block_size)
                    elif dr == 1:  # Neighbor is Below
                        # Shared border: y = nr * block_size
                        border_y = np.full(block_size, nr * block_size)
                        border_x = np.arange(c * block_size, (c + 1) * block_size)
                    elif dc == -1:  # Neighbor is Left
                        # Shared border: x = c * block_size
                        border_x = np.full(block_size, c * block_size)
                        border_y = np.arange(r * block_size, (r + 1) * block_size)
                    elif dc == 1:  # Neighbor is Right
                        # Shared border: x = nc * block_size
                        border_x = np.full(block_size, nc * block_size)
                        border_y = np.arange(r * block_size, (r + 1) * block_size)

                    # Calculate phase projection from the OLD block (already visited)
                    # Psi_old = G_old * pos + P_old
                    psi_from_old = (
                        G_cx[r, c] * border_x + G_cy[r, c] * border_y + P[r, c]
                    )

                    # Calculate gradient projection from the NEW block
                    # Psi_new_grad = G_new * pos
                    psi_from_new_grad = (
                        G_cx[nr, nc] * border_x + G_cy[nr, nc] * border_y
                    )

                    # The estimated offset P_new = Psi_old - Psi_new_grad
                    # This ensures Psi_new = Psi_old at the border.
                    estimates = psi_from_old - psi_from_new_grad

                    complex_estimates = np.exp(1j * estimates)
                    complex_mean = np.mean(complex_estimates)

                    # Reconvert to phase (extract angle)
                    P[nr, nc] = np.angle(complex_mean)

                    # Mark visited and enqueue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # 4. Construct Final Continuous Phase Image (Eq 14)
    # Psi_C(x,y) = G_cx * x + G_cy * y + P

    # Create meshgrids for pixel coordinates
    H_img, W_img = rows * block_size, cols * block_size
    y_grid, x_grid = np.mgrid[0:H_img, 0:W_img]

    # Upsample G and P to pixel size (repeat values for each block)
    # Using Kronecker product for efficient block repetition
    G_cx_img = np.kron(G_cx, np.ones((block_size, block_size)))
    G_cy_img = np.kron(G_cy, np.ones((block_size, block_size)))
    P_img = np.kron(P, np.ones((block_size, block_size)))

    plt.imshow(P_img, cmap="gray")
    plt.title("Phase Offset P (Upsampled)")
    plt.colorbar()
    plt.show()

    # Calculate Phase
    continuous_phase = G_cx_img * x_grid + G_cy_img * y_grid + P_img

    # Mask out background
    mask_img = np.kron(mask, np.ones((block_size, block_size)))
    continuous_phase[~mask_img.astype(bool)] = 0

    return continuous_phase


def generate_fingerprint(
    continous_images_dir,
    spiral_images_dir,
    minutiae_dir,
    orientation_map_dir,
    freq_map_dir,
    height,
    width,
):
    global start_index
    singularity_type = 1

    if not os.path.exists(continous_images_dir):
        os.makedirs(continous_images_dir)
    if not os.path.exists(spiral_images_dir):
        os.makedirs(spiral_images_dir)
    if not os.path.exists(minutiae_dir):
        os.makedirs(minutiae_dir)
    if not os.path.exists(orientation_map_dir):
        os.makedirs(orientation_map_dir)
    if not os.path.exists(freq_map_dir):
        os.makedirs(freq_map_dir)

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

    singularities = []
    for item in core_positions + delta_positions:
        c, r = int(item.x), int(item.y)
        if r != 0 and c != 0:
            singularities.append((r, c))

    # print(singularities)
    unwrapped_orientation = None

    unwrapped_orientation = unwrap_with_smart_cuts(
        orientation_map,
        mask=np.ones_like(orientation_map, dtype=bool),
        singularities=singularities,
    )
    freq_map = select_and_merge_density_maps(width, height)
    freq_map_jitter = (
        (freq_map - np.min(freq_map)) / (np.max(freq_map) - np.min(freq_map)) * 0.05
    )
    base_freq = np.random.uniform(0.057, 0.153)
    freq_map = base_freq + freq_map_jitter

    phase = reconstruct_continuous_phase(
        unwrapped_orientation=unwrapped_orientation,
        mask=np.ones_like(unwrapped_orientation, dtype=bool),
        block_size=1,
        freq_map=freq_map,
    )
    # minor_noise_factor = phase.std() * 0.2
    # freq_map = select_and_merge_density_maps(width, height) * minor_noise_factor
    # phase += freq_map

    # save_image
    # plt.imsave(
    #     os.path.join(continous_images_dir, f"{start_index}.png"),
    #     np.cos(phase),
    #     cmap="gray",
    # )

    np.save(os.path.join(continous_images_dir, f"{start_index}.npy"), np.cos(phase))

    def add_spiral_phase(psi, points, polarities):
        # points: list of (y,x); polarities: +1 for termination, -1 for bifurcation (convention)
        H, W = psi.shape
        Y, X = np.mgrid[:H, :W]
        out = psi.copy()
        zero = np.zeros_like(psi)
        for (yy, xx), p in zip(points, polarities):
            zero += p * np.arctan2(Y - yy, X - xx)

        out += zero
        return out

    spiral_phase_coords = []
    spiral_phase_polarities = []

    num_minutiae = random.randint(25, 40)
    for _ in range(num_minutiae):
        y = random.randint(12, 230)
        x = random.randint(12, 230)
        spiral_phase_coords.append((y, x))

        polarity = random.choice([1, -1])
        spiral_phase_polarities.append(polarity)

    phase_with_spirals = add_spiral_phase(
        phase, spiral_phase_coords, spiral_phase_polarities
    )
    spiral_image = np.cos(phase_with_spirals)
    # plt.imsave(
    # os.path.join(spiral_images_dir, f"{start_index}.png"), spiral_image, cmap="gray"
    # )

    np.save(os.path.join(spiral_images_dir, f"{start_index}.npy"), spiral_image)
    np.save(os.path.join(orientation_map_dir, f"{start_index}.npy"), orientation_map)
    np.save(os.path.join(freq_map_dir, f"{start_index}.npy"), freq_map)

    # write to "minutiae_locations/.txt" as x, y, type per line
    minutiae_file = os.path.join(minutiae_dir, f"{start_index}.txt")

    with open(minutiae_file, "w") as f:
        for (y, x), p in zip(spiral_phase_coords, spiral_phase_polarities):
            f.write(f"{x},{y},{p}\n")

    start_index += 1


from tqdm import tqdm

if __name__ == "__main__":
    base_path = "/green/data/data_v3"

    IMAGE_DIMS = 299

    for i in tqdm(range(start_index, to_generate)):
        continuous_img_dir = os.path.join(base_path, "cont_images")
        full_img_dir = os.path.join(base_path, "full_images")
        minutiae_dir = os.path.join(base_path, "minutiae_locations")
        orientation_map_dir = os.path.join(base_path, "orientation_maps")
        freq_map_dir = os.path.join(base_path, "freq_maps")

        generate_fingerprint(
            continous_images_dir=continuous_img_dir,
            spiral_images_dir=full_img_dir,
            minutiae_dir=minutiae_dir,
            orientation_map_dir=orientation_map_dir,
            freq_map_dir=freq_map_dir,
            height=IMAGE_DIMS,
            width=IMAGE_DIMS,
        )
