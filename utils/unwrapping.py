import numpy as np
from collections import deque
import random


def get_monotonic_branch_cuts(orientation_map, mask, singularities):
    """
    Generates branch cuts by tracing from singularities to the bottom-left or bottom-right
    edges monotonically.
    """
    rows, cols = orientation_map.shape
    branch_cuts = np.zeros((rows, cols), dtype=bool)

    # Assign a unique ID to each cut to distinguish between "self" and "other"
    cut_ids = np.zeros((rows, cols), dtype=int)

    for i, (start_r, start_c) in enumerate(singularities):
        if not (0 <= start_r < rows and 0 <= start_c < cols):
            continue

        current_id = i + 1

        # 1. Determine Monotonic Direction (Left-Down or Right-Down)
        # If closer to left edge, go Left. Otherwise, Right.
        dist_left = start_c
        dist_right = (cols - 1) - start_c

        target_is_left = dist_left < dist_right

        # Define the "Ideal" vector for this singularity
        # We want to go DOWN (positive r) and Left/Right (negative/positive c)
        if target_is_left:
            ideal_vr, ideal_vc = 1.0, -1.0  # Down-Left
        else:
            ideal_vr, ideal_vc = 1.0, 1.0  # Down-Right

        # 2. Trace
        curr_r, curr_c = float(start_r), float(start_c)

        # Mark start
        branch_cuts[start_r, start_c] = True
        cut_ids[start_r, start_c] = current_id

        step_size = 0.5  # Small step size prevents jumping over pixels
        max_steps = max(rows, cols) * 3  # Allow long paths

        for _ in range(max_steps):
            r_idx, c_idx = int(round(curr_r)), int(round(curr_c))

            # --- A. Check Stop Conditions ---

            # 1. Hit Image Boundary
            if not (0 <= r_idx < rows and 0 <= c_idx < cols):
                break

            # 2. Hit Background Mask
            if not mask[r_idx, c_idx]:
                break

            # 3. Hit a DIFFERENT branch cut (Merge)
            existing_id = cut_ids[r_idx, c_idx]
            if existing_id != 0 and existing_id != current_id:
                # We hit another cut. Stop here to merge them.
                break

            # Mark current pixel
            branch_cuts[r_idx, c_idx] = True
            cut_ids[r_idx, c_idx] = current_id

            # --- B. Calculate Next Step ---

            # Get local orientation
            angle = orientation_map[r_idx, c_idx]
            vr, vc = np.sin(angle), np.cos(angle)

            # Check alignment with our "Ideal" Monotonic direction.
            # Dot product: (vr*ideal_vr + vc*ideal_vc)
            # If negative, the vector is pointing Up/Opposite. Flip it.
            if (vr * ideal_vr + vc * ideal_vc) < 0:
                vr, vc = -vr, -vc

            # Move
            curr_r += vr * step_size
            curr_c += vc * step_size

            # Optimization: If we actually hit the target column, we can stop
            if (target_is_left and curr_c <= 0) or (
                not target_is_left and curr_c >= cols - 1
            ):
                break

    return branch_cuts


def unwrap_with_monotonic_cuts(orientation_map, mask, singularities):
    rows, cols = orientation_map.shape
    unwrapped = np.zeros_like(orientation_map)
    visited = np.zeros((rows, cols), dtype=bool)

    # 1. Generate Monotonic Cuts
    cuts = get_monotonic_branch_cuts(orientation_map, mask, singularities)

    # 2. Unwrap the "Walkable" regions (Standard BFS)
    walkable = mask & (~cuts)
    start_points = np.argwhere(walkable)

    if len(start_points) > 0:
        seed = tuple(start_points[len(start_points) // 2])  # Start near center
        queue = deque([seed])
        visited[seed] = True
        unwrapped[seed] = orientation_map[seed]

        while queue:
            r, c = queue.popleft()

            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    if walkable[nr, nc] and not visited[nr, nc]:
                        diff = orientation_map[nr, nc] - unwrapped[r, c]
                        k = -np.round(diff / np.pi)
                        unwrapped[nr, nc] = orientation_map[nr, nc] + k * np.pi
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    # 3. Unwrap the Cuts (Nearest Neighbor fill)
    # Using a queue to iteratively fill cut pixels from valid neighbors
    cut_pixels = np.argwhere(cuts & mask)
    cut_queue = deque([tuple(p) for p in cut_pixels])

    while cut_queue:
        r, c = cut_queue.popleft()

        if visited[r, c]:
            continue

        # Find a visited neighbor
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        valid_neighbor = None

        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc]:
                valid_neighbor = (nr, nc)
                break

        if valid_neighbor:
            nr, nc = valid_neighbor
            diff = orientation_map[r, c] - unwrapped[nr, nc]
            k = -np.round(diff / np.pi)
            unwrapped[r, c] = orientation_map[r, c] + k * np.pi
            visited[r, c] = True
        else:
            # If no visited neighbor yet, put back in queue
            cut_queue.append((r, c))

    return unwrapped


def get_smart_branch_cuts(orientation_map, mask, singularities):
    rows, cols = orientation_map.shape
    branch_cuts = np.zeros((rows, cols), dtype=bool)

    for start_r, start_c in singularities:
        # Check bounds
        if not (0 <= start_r < rows and 0 <= start_c < cols):
            continue

        # 1. Identify Target Side and Point
        dist_left = start_c
        dist_right = (cols - 1) - start_c

        # Determine closer edge
        if dist_left < dist_right:
            target_col = 0
        else:
            target_col = cols - 1

        # Heuristic: 20-60 indices lower (higher up in image terms usually, or just lower index)
        offset = random.randint(20, 60)
        target_row = max(0, start_r - offset)

        # 2. Determine Initial Direction
        # Get orientation at singularity
        angle = orientation_map[int(start_r), int(start_c)]
        vr, vc = np.sin(angle), np.cos(angle)  # Orientation vector

        # Vector towards the target point
        tr, tc = target_row - start_r, target_col - start_c

        # Check dot product to see if we are facing the target or away
        dot_prod = vr * tr + vc * tc

        # If facing away, flip the vector so we start tracing TOWARDS the edge
        if dot_prod < 0:
            vr, vc = -vr, -vc

        # 3. Trace the Path
        curr_r, curr_c = float(start_r), float(start_c)
        current_trace_pixels = set()  # To track pixels in THIS specific cut
        current_trace_pixels.add((int(start_r), int(start_c)))

        # Mark start on main map
        branch_cuts[int(start_r), int(start_c)] = True

        step_size = 0.8  # Smaller than 1.0 to ensure we don't skip thin barriers
        max_steps = max(rows, cols) * 2

        for _ in range(max_steps):
            # Move
            curr_r += vr * step_size
            curr_c += vc * step_size

            r_idx, c_idx = int(round(curr_r)), int(round(curr_c))

            # Boundary check
            if not (0 <= r_idx < rows and 0 <= c_idx < cols):
                break  # Hit image edge

            # Mask check
            if not mask[r_idx, c_idx]:
                break  # Hit background

            if branch_cuts[r_idx, c_idx] and (r_idx, c_idx) not in current_trace_pixels:
                break

            # Update maps
            branch_cuts[r_idx, c_idx] = True
            current_trace_pixels.add((r_idx, c_idx))

            # 4. Update Direction (Smoothness Constraint)
            # Retrieve new angle at current position
            new_angle = orientation_map[r_idx, c_idx]
            new_vr, new_vc = np.sin(new_angle), np.cos(new_angle)

            # Momentum check: Ensure new vector is roughly in same direction as previous
            # (Dot product > 0). If < 0, it means the orientation flipped 180 (which is valid
            # in orientation maps), so we flip our vector to maintain the flow.
            if (vr * new_vr + vc * new_vc) < 0:
                new_vr = -new_vr
                new_vc = -new_vc

            vr, vc = new_vr, new_vc

            # Stop if we actually reached the target column (optional, but good for efficiency)
            if (target_col == 0 and c_idx <= 0) or (
                target_col == cols - 1 and c_idx >= cols - 1
            ):
                break

    return branch_cuts


def unwrap_with_smart_cuts(orientation_map, mask, singularities):
    """
    Standard unwrapping using the computed branch cuts as barriers.
    """
    rows, cols = orientation_map.shape
    unwrapped = np.zeros_like(orientation_map)
    visited = np.zeros((rows, cols), dtype=bool)

    # 1. Generate Cuts
    cuts = get_smart_branch_cuts(orientation_map, mask, singularities)

    # 2. Unwrap non-cut regions
    walkable = mask & (~cuts)

    start_points = np.argwhere(walkable)
    if len(start_points) > 0:
        # Pick a seed point (try center of mass or just first valid point)
        seed = tuple(start_points[len(start_points) // 2])
        queue = deque([seed])
        visited[seed] = True
        unwrapped[seed] = orientation_map[seed]

        while queue:
            r, c = queue.popleft()

            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    if walkable[nr, nc] and not visited[nr, nc]:
                        diff = orientation_map[nr, nc] - unwrapped[r, c]
                        k = -np.round(diff / np.pi)
                        unwrapped[nr, nc] = orientation_map[nr, nc] + k * np.pi
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    # 3. Unwrap the cuts themselves (simple nearest neighbor fill)
    cut_pixels = np.argwhere(cuts & mask)
    # We might need multiple passes or a queue if cuts are thick,
    # but iterating a few times usually clears them.
    cut_queue = deque([tuple(p) for p in cut_pixels])

    while cut_queue:
        r, c = cut_queue.popleft()

        # If we visited it (perhaps from a previous loop iteration), skip
        if visited[r, c]:
            continue

        # Check neighbors for a visited value
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        valid_neighbor = None

        for nr, nc in neighbors:
            if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc]:
                valid_neighbor = (nr, nc)
                break

        if valid_neighbor:
            nr, nc = valid_neighbor
            diff = orientation_map[r, c] - unwrapped[nr, nc]
            k = -np.round(diff / np.pi)
            unwrapped[r, c] = orientation_map[r, c] + k * np.pi
            visited[r, c] = True
        else:
            # Re-queue to try later when neighbors are filled
            cut_queue.append((r, c))

    return unwrapped
