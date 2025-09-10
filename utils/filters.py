import numpy as np
import os
import cv2
from utils.gabors import generate_custom_gabor, generate_gabor_kernel
from gabor_torch import generate_custom_gabor_torch

import matplotlib.pyplot as plt

distnct_f = 100
distnct_o = 180
path_filter = "../Filterbank"

PI = np.pi


def pre_filtering(orientation_map, freq_map, height, width, margin):
    """
    Converts orientation and frequency maps to index maps for filter lookup.

    Parameters:
        orientation_map: 2D array of orientation values in radians
        freq_map: 2D array of frequency values
        height, width: dimensions of the original image
        margin: additional margin around the image

    Returns:
        Tuple of (orientation_indices_map, frequencies_indices_map)
    """
    # Create the maps with proper dimensions
    orientation_indices_map = np.zeros(
        (height + margin, width + margin), dtype=np.int32
    )
    frequencies_indices_map = np.zeros(
        (height + margin, width + margin), dtype=np.int32
    )

    # Convert the entire maps at once using vectorized operations
    orientation_indices_map = ((orientation_map * 180 / PI) % distnct_o).astype(
        np.int32
    )
    frequencies_indices_map = (100 - (freq_map * 100) % distnct_f).astype(np.int32)

    return orientation_indices_map, frequencies_indices_map


def normalize2Df_print(f_print):
    """
    Normalizes a 2D numpy array to the range [0, 100].
    """
    maxval = np.max(f_print)
    minval = np.min(f_print)

    # Shift values if there are negatives
    if minval < 0.0:
        f_print += abs(minval)
    else:
        f_print -= minval

    # Recompute max after shifting
    maxval = np.max(f_print)

    # Normalize to [0, 100]
    if maxval != 0:
        f_print = (f_print / maxval) * 100

    return f_print


def binarize2Df_print(f_print, val):
    """
    Binarizes a 2D numpy array based on a threshold.
    Values above the threshold become 100, others become 0.
    """
    # Binarization using Numpy's vectorized operations
    f_print = np.where(f_print > val, 100, 0)
    return f_print


def copy2Df_print(src):
    """
    Returns a copy of the source 2D numpy array.
    """
    return src.copy()


def load_gabor_filters():
    # Define cache file path
    cache_file = f"gabor_filters_cache.npz"

    # Check if cache exists
    if os.path.exists(cache_file):
        try:
            # Load from cache
            cached_data = np.load(cache_file, allow_pickle=True)
            # Remove .item() since filterbank is not a scalar array
            filterbank_4Dmat = cached_data["filterbank"]
            filter_size_2Dmat = cached_data["filter_sizes"]
            # Convert to int explicitly
            max_filter_size = int(cached_data["max_size"])
            min_filter_size = int(cached_data["min_size"])

            print(f"Loaded Gabor filters from cache: {cache_file}")
            return filterbank_4Dmat, filter_size_2Dmat, max_filter_size, min_filter_size
        except Exception as e:
            print(f"Error loading from cache: {e}. Regenerating filters...")

    # Create a 4D list to hold the filter bank: [frequency][orientation][height][width]
    filterbank_4Dmat = [[None for _ in range(distnct_o)] for _ in range(distnct_f)]
    filter_size_2Dmat = np.zeros((distnct_f, distnct_o), dtype=int)
    max_filter_size = 0
    min_filter_size = float("inf")

    # Load and process filters
    for freq_ind in range(distnct_f):
        for orient_ind in range(distnct_o):
            # Construct file path
            file_path = f"{path_filter}/{freq_ind + 1}/{orient_ind + 1}.bmp"
            if not os.path.exists(file_path):
                print(f"Error: Not able to load Filter Bank at {file_path}")
                exit(0)

            # Load image in grayscale
            filter_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if filter_img is None:
                print(f"Failed to load image: {file_path}")
                exit(0)

            # Convert to float and normalize to range [0, 1]
            mat = filter_img.astype(np.float32) / 255.0

            # Normalize and scale as per original code
            mat = mat * 100 - 46

            # Save the processed filter in the 4D matrix
            filterbank_4Dmat[freq_ind][orient_ind] = mat
            filter_size = mat.shape[0]
            filter_size_2Dmat[freq_ind, orient_ind] = filter_size

            # Track maximum filter size
            if filter_size > max_filter_size:
                max_filter_size = filter_size
            min_filter_size = min(min_filter_size, filter_size)

    # Save to cache file
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(
            cache_file,
            filterbank=filterbank_4Dmat,  # Save the complete nested list structure directly
            filter_sizes=filter_size_2Dmat,
            max_size=max_filter_size,
            min_size=min_filter_size,
        )
        print(f"Saved Gabor filters to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return filterbank_4Dmat, filter_size_2Dmat, max_filter_size, min_filter_size


def filter_image(
    f_print1,
    freq_ind,
    orient_ind,
    filter_size_2d,
    filterbank,
    H,
    W,
    margin,
    max_filter_size,
    binarization_threshold=55,
):
    """
    Applies the filtering process to the entire image.

    For each pixel (excluding border areas), this function:
      - Selects a filter based on the frequency and orientation indices.
      - Applies the filter over a local neighborhood.
      - Stores the filtered value in an output image.
    After filtering, the result is:
      1. Copied back to the input image.
      2. Normalized to the range [0, 100].
      3. Binarized with threshold 55.
      4. Normalized again.

    Parameters:
      f_print1       : Input image as a 2D NumPy array.
      freq_ind       : 2D NumPy array of frequency indices.
      orient_ind     : 2D NumPy array of orientation indices.
      filter_size_2d : 2D NumPy array mapping (freq, orient) to filter size.
      filterbank     : 4D NumPy array of filters indexed by (freq, orient, r, c).
      H, W, margin   : Image dimensions and margin.
      max_filter_size: Maximum filter size (used to compute border limits).

    Returns:
      Processed image as a 2D NumPy array.
    """
    # Create output image
    f_print2 = np.zeros_like(f_print1, dtype=np.float32)

    # Compute iteration boundaries to avoid border issues
    i_start = int(np.floor(max_filter_size / 2.0))
    i_end = int(H + margin - np.ceil(max_filter_size / 2.0))
    j_start = int(np.floor(max_filter_size / 2.0))
    j_end = int(W + margin - np.ceil(max_filter_size / 2.0))

    # Process all pixels in valid region
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            freq_idx = int(freq_ind[i, j]) - 1
            orient_idx = int(orient_ind[i, j]) - 1
            filter_size = int(filter_size_2d[freq_idx, orient_idx])
            filt_size_half = int(np.floor(filter_size / 2.0))

            # Extract neighborhood and apply filter in one operation
            # Calculate neighborhood bounds
            i_min = i - filt_size_half
            i_max = i + filt_size_half + 1
            j_min = j - filt_size_half
            j_max = j + filt_size_half + 1

            # Clamp bounds to valid image dimensions
            i_min_clamped = max(i_min, 0)
            i_max_clamped = min(i_max, H + margin)
            j_min_clamped = max(j_min, 0)
            j_max_clamped = min(j_max, W + margin)

            # Get the neighborhood and corresponding filter portion
            neighborhood = f_print1[
                i_min_clamped:i_max_clamped, j_min_clamped:j_max_clamped
            ]

            # Calculate filter offsets if neighborhood is at image boundary
            filter_i_start = max(0, -i_min)
            filter_j_start = max(0, -j_min)
            filter_i_end = filter_size - max(0, i_max - (H + margin))
            filter_j_end = filter_size - max(0, j_max - (W + margin))

            # Get corresponding portion of filter
            filter_portion = filterbank[freq_idx][orient_idx][
                filter_i_start:filter_i_end, filter_j_start:filter_j_end
            ]

            # Element-wise multiply and sum
            f_print2[i, j] = np.sum(neighborhood * filter_portion)

    # Post-process: copy, normalize, binarize, and normalize again
    f_out = normalize2Df_print(f_print2.copy())
    f_out = binarize2Df_print(f_out, binarization_threshold)
    f_out = normalize2Df_print(f_out)

    return f_out


def filter_image_withmap(
    f_print1,
    filt_map,
    freq_ind,
    orient_ind,
    filter_size_2d,
    filterbank,
    H,
    W,
    margin,
    max_filter_size,
    binarization_threshold=55,
):
    """
    Applies filtering only at pixels where filt_map equals 1.

    For each pixel in the valid region:
      - If filt_map[i, j] == 1, selects a filter based on the frequency and orientation indices.
      - Applies the filter over the local neighborhood.
      - Stores the filtered value in an output image.

    After processing, the filtered image is copied back, normalized to [0, 100],
    and then binarized using a threshold of 55.

    Parameters:
      f_print1       : 2D numpy array representing the input image.
      filt_map       : 2D numpy array (same shape as f_print1) with flags (0 or 1).
      freq_ind       : 2D numpy array of frequency indices.
      orient_ind     : 2D numpy array of orientation indices.
      filter_size_2d : 2D numpy array mapping (freq, orient) to a filter size.
      filterbank     : 4D numpy array of filters indexed by (freq, orient, r, c).
      H, W, margin   : Dimensions of the original image and margin.
      max_filter_size: Maximum filter size (used for border calculations).

    Returns:
      Processed image as a 2D numpy array.
    """
    # Create an output image (float32 for accumulation)
    f_print2 = np.zeros_like(f_print1, dtype=np.float32)

    # Determine iteration boundaries (avoid border issues)
    i_start = int(np.floor(max_filter_size / 2.0))
    i_end = int(H + margin - np.ceil(max_filter_size / 2.0))
    j_start = int(np.floor(max_filter_size / 2.0))
    j_end = int(W + margin - np.ceil(max_filter_size / 2.0))

    # Find coordinates where filtering should be applied
    filter_coords = np.where(filt_map[i_start:i_end, j_start:j_end] == 1)
    i_coords = filter_coords[0] + i_start
    j_coords = filter_coords[1] + j_start

    # Process only pixels where filt_map is 1
    for idx in range(len(i_coords)):
        i, j = i_coords[idx], j_coords[idx]

        freq_idx = int(freq_ind[i, j]) - 1
        orient_idx = int(orient_ind[i, j]) - 1
        filter_size = int(filter_size_2d[freq_idx][orient_idx])
        filt_size_half = int(np.floor(filter_size / 2.0))

        # Calculate neighborhood bounds
        i_min = i - filt_size_half
        i_max = i + filt_size_half + 1
        j_min = j - filt_size_half
        j_max = j + filt_size_half + 1

        # Clamp bounds to valid image dimensions
        i_min_clamped = max(i_min, 0)
        i_max_clamped = min(i_max, H + margin)
        j_min_clamped = max(j_min, 0)
        j_max_clamped = min(j_max, W + margin)

        # Get the neighborhood
        neighborhood = f_print1[
            i_min_clamped:i_max_clamped, j_min_clamped:j_max_clamped
        ]

        # Calculate filter offsets if neighborhood is at image boundary
        filter_i_start = max(0, -i_min)
        filter_j_start = max(0, -j_min)
        filter_i_end = filter_size - max(0, i_max - (H + margin))
        filter_j_end = filter_size - max(0, j_max - (W + margin))

        # Get corresponding portion of filter
        filter_portion = filterbank[freq_idx][orient_idx][
            filter_i_start:filter_i_end, filter_j_start:filter_j_end
        ]

        # Apply filter: element-wise multiply and sum
        f_print2[i, j] = np.sum(neighborhood * filter_portion)

    # Copy filtered result and post-process: normalize then binarize
    f_out = normalize2Df_print(f_print2.copy())  # Normalize to [0, 100]
    f_out = binarize2Df_print(
        f_out, binarization_threshold
    )  # Binarize with threshold 55

    return f_out


def filter_with_math_gabor(
    f_print1,
    filt_map,
    freq_ind,
    orient_ind,
    filter_size_2d,
    H,
    W,
    margin,
    max_filter_size,
    binarization_threshold=55,
):
    f_print2 = np.zeros_like(f_print1, dtype=np.float32)

    # Determine iteration boundaries (avoid border issues)
    i_start = int(np.floor(max_filter_size / 2.0))
    i_end = int(H + margin - np.ceil(max_filter_size / 2.0))
    j_start = int(np.floor(max_filter_size / 2.0))
    j_end = int(W + margin - np.ceil(max_filter_size / 2.0))

    # Find coordinates where filtering should be applied
    filter_coords = np.where(filt_map[i_start:i_end, j_start:j_end] == 1)
    i_coords = filter_coords[0] + i_start
    j_coords = filter_coords[1] + j_start

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            freq_idx = int(freq_ind[i, j])
            orient_idx = int(orient_ind[i, j])

            # Compute orientation in radians (assuming 8 orientations for example)
            filter_size = int(filter_size_2d[freq_idx - 1][orient_idx - 1])
            filt_size_half = int(np.floor(filter_size / 2.0))

            # Compute neighborhood bounds
            i_min = max(i - filt_size_half, 0)
            i_max = min(i + filt_size_half + 1, H + margin)
            j_min = max(j - filt_size_half, 0)
            j_max = min(j + filt_size_half + 1, W + margin)

            # Extract local patch
            neighborhood = f_print1[i_min:i_max, j_min:j_max]

            # Generate Gabor kernel for this pixel
            # Define a spatial frequency mapping (you can tune this mapping)
            theta = (orient_idx / 180) * np.pi
            freq = 0.025 + 0.0015 * freq_idx

            gamma_0 = 1  # + np.random.uniform(-0.3, 0.3)
            gamma_delta = 0.6  # + np.random.uniform(-0.2, 0.3)
            sigma = 6.0

            kernel = generate_custom_gabor(
                size=filter_size,
                theta=theta,
                freq_0=freq,
                freq_delta=freq / 3,
                sigma=sigma,
                gamma_0=gamma_0,
                gamma_delta=gamma_delta,
                phase=0,
            )

            kernel_torch = generate_custom_gabor_torch(
                size=filter_size,
                theta=theta,
                freq_0=freq,
                freq_delta=freq / 3,
                sigma=sigma,
                gamma_0=gamma_0,
                gamma_delta=gamma_delta,
            )

            # Visualization disabled in refactored run to avoid blocking GUI
            # plt.tight_layout()
            # plt.show()

            # Adjust kernel slice if we’re near the edge
            f_i_min = max(filt_size_half - (i - i_min), 0)
            f_i_max = f_i_min + (i_max - i_min)
            f_j_min = max(filt_size_half - (j - j_min), 0)
            f_j_max = f_j_min + (j_max - j_min)

            kernel_slice = kernel[f_i_min:f_i_max, f_j_min:f_j_max]

            # Convolve (element-wise multiply and sum)
            f_print2[i, j] = np.sum(neighborhood * kernel_slice)

    f_out = normalize2Df_print(f_print2.copy())  # Normalize to [0, 100]
    f_out = binarize2Df_print(
        f_out, binarization_threshold
    )  # Binarize with threshold 55

    return f_out


def filter_image_firstpass(
    f_print1,
    filt_map,
    freq_ind,
    orient_ind,
    filter_size_2d,
    filterbank,
    H,
    W,
    margin,
    max_filter_size,
    binarization_threshold=55,
):
    """
    First-pass filtering with optimized NumPy operations:
      - First, binarizes f_print1 with threshold 65 and normalizes it to [0, 100].
      - Then, for each pixel (excluding borders) where filt_map is 1, applies a local filter.
      - The filtered result is stored and then copied back, normalized, and binarized.

    Parameters:
      f_print1       : Input image as a 2D NumPy array.
      filt_map       : 2D NumPy array (same shape as f_print1) with flags (0 or 1).
      freq_ind       : 2D NumPy array of frequency indices.
      orient_ind     : 2D NumPy array of orientation indices.
      filter_size_2d : 2D NumPy array mapping (freq, orient) to filter size.
      filterbank     : 4D NumPy array of filters indexed by (freq, orient, r, c).
      H, W, margin   : Image dimensions and margin.
      max_filter_size: Maximum filter size (for computing border limits).

    Returns:
      Processed image as a 2D NumPy array.
    """
    # Pre-process: binarize with threshold 65 and then normalize
    f_processed = binarize2Df_print(f_print1, 65)
    f_processed = normalize2Df_print(f_processed)

    # Create output image
    f_print2 = np.zeros_like(f_processed, dtype=np.float32)

    # Compute iteration boundaries
    i_start = int(np.floor(max_filter_size / 2.0))
    i_end = int(H + margin - np.ceil(max_filter_size / 2.0))
    j_start = int(np.floor(max_filter_size / 2.0))
    j_end = int(W + margin - np.ceil(max_filter_size / 2.0))

    # Get coordinates where filtering should be applied
    filter_coords = np.where(filt_map[i_start:i_end, j_start:j_end] == 1)
    i_coords = filter_coords[0] + i_start
    j_coords = filter_coords[1] + j_start

    # Process each pixel where filt_map is 1
    for idx in range(len(i_coords)):
        i, j = i_coords[idx], j_coords[idx]

        freq_idx = int(freq_ind[i, j]) - 1
        orient_idx = int(orient_ind[i, j]) - 1
        filter_size = int(filter_size_2d[freq_idx][orient_idx])
        filt_size_half = int(np.floor(filter_size / 2.0))

        # Define the neighborhood boundaries
        i_min = max(i - filt_size_half, 0)
        i_max = min(i + filt_size_half + 1, H + margin)
        j_min = max(j - filt_size_half, 0)
        j_max = min(j + filt_size_half + 1, W + margin)

        # Extract neighborhood
        neighborhood = f_processed[i_min:i_max, j_min:j_max]

        # Adjust filter indices based on boundary conditions
        f_i_min = max(filt_size_half - (i - i_min), 0)
        f_i_max = min(filter_size - (i + filt_size_half + 1 - i_max), filter_size)
        f_j_min = max(filt_size_half - (j - j_min), 0)
        f_j_max = min(filter_size - (j + filt_size_half + 1 - j_max), filter_size)

        # Extract and apply filter
        filter_section = filterbank[freq_idx][orient_idx][
            f_i_min:f_i_max, f_j_min:f_j_max
        ]
        f_print2[i, j] = np.sum(neighborhood * filter_section)

    # Post-process: normalize and binarize
    f_out = normalize2Df_print(f_print2)
    f_out = binarize2Df_print(f_out, binarization_threshold)

    return f_out


def filter_image_preserve_variation(
    f_print1,
    freq_ind,
    orient_ind,
    filter_size_2d,
    filterbank,
    H,
    W,
    margin,
    max_filter_size,
    final_binarization_threshold=55,
    passes=3,
    jitter_freq=False,
    adaptive_thresholding=False,
):
    f_curr = f_print1.astype(np.float32).copy()

    i_start = int(np.floor(max_filter_size / 2.0))
    i_end = int(H + margin - np.ceil(max_filter_size / 2.0))
    j_start = int(np.floor(max_filter_size / 2.0))
    j_end = int(W + margin - np.ceil(max_filter_size / 2.0))

    for pass_num in range(passes):
        f_next = np.zeros_like(f_curr, dtype=np.float32)

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                freq_idx = int(freq_ind[i, j]) - 1
                orient_idx = int(orient_ind[i, j]) - 1

                if jitter_freq:
                    freq_idx = np.clip(
                        freq_idx + np.random.randint(-1, 2), 0, filterbank.shape[0] - 1
                    )

                filter_size = int(filter_size_2d[freq_idx][orient_idx])
                filt_size_half = int(np.floor(filter_size / 2.0))

                i_min = max(i - filt_size_half, 0)
                i_max = min(i + filt_size_half + 1, H + margin)
                j_min = max(j - filt_size_half, 0)
                j_max = min(j + filt_size_half + 1, W + margin)

                neighborhood = f_curr[i_min:i_max, j_min:j_max]

                f_i_min = max(filt_size_half - (i - i_min), 0)
                f_i_max = min(
                    filter_size - (i + filt_size_half + 1 - i_max), filter_size
                )
                f_j_min = max(filt_size_half - (j - j_min), 0)
                f_j_max = min(
                    filter_size - (j + filt_size_half + 1 - j_max), filter_size
                )

                filter_section = filterbank[freq_idx][orient_idx][
                    f_i_min:f_i_max, f_j_min:f_j_max
                ]

                f_next[i, j] = np.sum(neighborhood * filter_section)

        # Update current frame for next iteration
        f_curr = f_next

    # Normalize once at the end
    f_out = normalize2Df_print(f_curr)

    if adaptive_thresholding:
        # Compute threshold using local mean ± k * stddev
        local_mean = np.mean(f_out)
        local_std = np.std(f_out)
        threshold = local_mean + 0.5 * local_std
    else:
        threshold = final_binarization_threshold

    f_out = binarize2Df_print(f_out, threshold)
    f_out = normalize2Df_print(f_out)
    return f_out


def seed_pos(f_print1, H, W, margin, n_seeds=None):
    """
    Seeds the image with blobs by setting corresponding positions in
    filt_map and f_print1.

    Parameters:
        f_print1 (np.ndarray): 2D array representing the image values.
        filt_map (np.ndarray): 2D array representing the filter map.
        H (int): Height of the original image (without margin).
        W (int): Width of the original image (without margin).
        margin (int): Margin added around the image.
        n_seeds (int, optional): Number of seeds to add. If None, it will be
                                 computed as 750 + floor(rand * 100).

    Returns:
        tuple: A tuple containing the modified (f_print1, filt_map) arrays.
    """
    # Compute number of seeds if not provided
    if n_seeds is None:
        n_seeds = 750 + int(np.floor(np.random.rand() * 100))

    print("number of seeds in image: {}".format(n_seeds))
    total_rows = H + margin
    total_cols = W + margin

    # Generate random positions for all seeds at once
    i_blobs = np.random.randint(5, total_rows - 5, n_seeds)
    j_blobs = np.random.randint(5, total_cols - 5, n_seeds)

    # Create 4x4 blocks for all seeds
    for i_offset in range(4):
        for j_offset in range(4):
            # Calculate all positions at once with broadcasting
            i_positions = i_blobs[:, np.newaxis] + i_offset
            j_positions = j_blobs[np.newaxis, :] + j_offset

            # Use advanced indexing to set values
            idx = (i_positions.flatten(), j_positions.flatten())
            f_print1[idx] = 100


# def set_filter_area(filt_map, H, W, margin, max_filter_size):
#     half_filter = int(np.floor(max_filter_size / 2.0))

#     # Process rows
#     for i in range(H + margin):
#         # Find transitions from 0 to 1 (start points)
#         row = filt_map[i]
#         transitions_to_1 = np.where((row[:-1] == 0) & (row[1:] == 1))[0] + 1

#         # Find transitions from 1 to 0 (end points)
#         transitions_to_0 = np.where((row[:-1] == 1) & (row[1:] == 0))[0] + 1

#         # Handle case where row starts with 1
#         if row[0] == 1 and transitions_to_1.size > 0 and transitions_to_1[0] > 0:
#             transitions_to_1 = np.insert(transitions_to_1, 0, 0)

#         # Handle case where row ends with 1
#         if row[-1] == 1 and (
#             transitions_to_0.size == 0 or transitions_to_0[-1] < W + margin - 1
#         ):
#             transitions_to_0 = np.append(transitions_to_0, W + margin)

#         # Fill areas around each segment
#         for start, end in zip(transitions_to_1, transitions_to_0):
#             left_idx = max(0, start - half_filter)
#             right_idx = min(W + margin, end + half_filter)
#             filt_map[i, left_idx:right_idx] = 1

#     # Process columns
#     for j in range(W + margin):
#         # Find transitions from 0 to 1 (start points)
#         col = filt_map[:, j]
#         transitions_to_1 = np.where((col[:-1] == 0) & (col[1:] == 1))[0] + 1

#         # Find transitions from 1 to 0 (end points)
#         transitions_to_0 = np.where((col[:-1] == 1) & (col[1:] == 0))[0] + 1

#         # Handle case where column starts with 1
#         if col[0] == 1 and transitions_to_1.size > 0 and transitions_to_1[0] > 0:
#             transitions_to_1 = np.insert(transitions_to_1, 0, 0)

#         # Handle case where column ends with 1
#         if col[-1] == 1 and (
#             transitions_to_0.size == 0 or transitions_to_0[-1] < H + margin - 1
#         ):
#             transitions_to_0 = np.append(transitions_to_0, H + margin)

#         # Fill areas around each segment
#         for start, end in zip(transitions_to_1, transitions_to_0):
#             top_idx = max(0, start - half_filter)
#             bottom_idx = min(H + margin, end + half_filter)
#             filt_map[top_idx:bottom_idx, j] = 1

#     return filt_map


def set_filter_area(filt_map, H, W, margin, max_filter_size):
    pad = max_filter_size // 2

    # — Horizontal pass —
    i = 0
    while i < H + margin:
        flag_in = False
        j = 0
        while j < W + margin:
            if filt_map[i][j] == 1 and not flag_in:
                for k in range(j - pad, j):
                    if 0 <= k < W + margin:
                        filt_map[i][k] = 1
                flag_in = True

            elif filt_map[i][j] == 0 and flag_in:
                for k in range(j, j + pad + 1):
                    if 0 <= k < W + margin:
                        filt_map[i][k] = 1
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
            if filt_map[i][j] == 1 and not flag_in:
                for k in range(i - pad, i):
                    if 0 <= k < H + margin:
                        filt_map[k][j] = 1
                flag_in = True

            elif filt_map[i][j] == 0 and flag_in:
                for k in range(i, i + pad + 1):
                    if 0 <= k < H + margin:
                        filt_map[k][j] = 1
                i += pad + 1
                flag_in = False
                continue

            i += 1
        j += 1

    return filt_map
