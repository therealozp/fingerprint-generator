import cv2
import numpy as np
import os

# Example variables (set your own paths and counts)
path_density_maps = "density_maps"
density_map_count = 2000
margin = 30
W, H = 275 - margin, 400 - margin


def rand():
    return np.random.rand()


def load_random_density_map():
    idx = int(1 + rand() * (density_map_count - 1))
    filepath = os.path.join(path_density_maps, f"{idx}.jpeg")
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def sel_n_merg_densitymap():
    # Load two random maps
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

    # Resize to target dimensions
    f_den = cv2.resize(freq, (W + margin, H + margin), interpolation=cv2.INTER_LINEAR)

    # Convert to integer matrix and normalize to [0, 99]
    f_den_int = np.clip(f_den * 100, 0, 99999).astype(np.int32)
    min_val, max_val = f_den_int.min(), f_den_int.max()
    f_den_int = ((f_den_int - min_val) / (max_val - min_val) * 99).astype(np.int32)

    return f_den_int


def show_density_map(density_map):
    # Convert back to [0, 255] range for visualization if needed
    f_den_scaled = (density_map * 255).astype(np.uint8)
    cv2.imshow("Density Map", f_den_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("merged_density_map.png", f_den_scaled)


# # Run and get the result
# f_den_int = sel_n_merg_densitymap()
# np.savetxt("merged_density_map.txt", f_den_int, fmt="%d")
# print("Merged density map saved as 'merged_density_map.png'")
