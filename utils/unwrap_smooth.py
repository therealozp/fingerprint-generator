import numpy as np
from collections import deque


def get_quadratic_bezier_points(start, end, control, num_points=100):
    """
    Generates points for a Quadratic Bezier curve defined by P0(start), P1(control), P2(end).
    Formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    """
    r0, c0 = start
    r1, c1 = control
    r2, c2 = end

    points = []
    for t in np.linspace(0, 1, num_points):
        r = (1 - t) ** 2 * r0 + 2 * (1 - t) * t * r1 + t**2 * r2
        c = (1 - t) ** 2 * c0 + 2 * (1 - t) * t * c1 + t**2 * c2
        points.append((r, c))
    return points


def get_smooth_curved_cuts(shape, mask, singularities):
    rows, cols = shape
    branch_cuts = np.zeros((rows, cols), dtype=bool)

    # Filter valid singularities
    valid_sings = [s for s in singularities if 0 <= s[0] < rows and 0 <= s[1] < cols]

    if not valid_sings:
        return branch_cuts

    cuts_to_draw = []

    # CASE 1: Single Singularity
    if len(valid_sings) == 1:
        s1 = valid_sings[0]
        r, c = s1

        # Distances to edges
        dist_left = c
        dist_right = (cols - 1) - c

        # Pick closer edge
        if dist_left < dist_right:
            target_col = 0
        else:
            target_col = cols - 1

        # Drop 20-60 pixels down (random or fixed average)
        # We ensure we don't go off the bottom of the image
        drop = min(60, rows - 1 - r)
        target_row = r + drop

        cuts_to_draw.append(
            {"start": s1, "end": (target_row, target_col), "type": "edge_drop"}
        )

    # CASE 2: Two Singularities (Core & Delta usually)
    elif len(valid_sings) >= 2:
        s1 = valid_sings[0]
        s2 = valid_sings[1]

        # Calculate distances
        dist_between = np.hypot(s1[0] - s2[0], s1[1] - s2[1])

        # Min distances to ANY edge for each singularity
        d1_L = s1[1]
        d1_R = (cols - 1) - s1[1]
        d1_min_edge = min(d1_L, d1_R)

        d2_L = s2[1]
        d2_R = (cols - 1) - s2[1]
        d2_min_edge = min(d2_L, d2_R)

        min_distance_to_edge = min(d1_min_edge, d2_min_edge)

        # --- SUB-CASE A: Singularities are closer to each other than to any edge ---
        if dist_between < min_distance_to_edge:
            # Bridge them directly
            cuts_to_draw.append({"start": s1, "end": s2, "type": "bridge"})

        # --- SUB-CASE B: Split to edges ---
        else:
            # Which singularity is closer to an edge?
            if d1_min_edge < d2_min_edge:
                priority_sing = s1
                other_sing = s2
                p_d_L, p_d_R = d1_L, d1_R
            else:
                priority_sing = s2
                other_sing = s1
                p_d_L, p_d_R = d2_L, d2_R

            # 1. Priority Singularity picks its closest edge
            drop_p = min(60, rows - 1 - priority_sing[0])
            target_r_p = priority_sing[0] + drop_p

            if p_d_L < p_d_R:
                priority_target_c = 0  # Left
                other_must_go = "right"  # Force other to Right
            else:
                priority_target_c = cols - 1  # Right
                other_must_go = "left"  # Force other to Left

            cuts_to_draw.append(
                {
                    "start": priority_sing,
                    "end": (target_r_p, priority_target_c),
                    "type": "edge_drop",
                }
            )

            # 2. Other Singularity goes to opposite edge
            drop_o = min(60, rows - 1 - other_sing[0])
            target_r_o = other_sing[0] + drop_o

            if other_must_go == "left":
                other_target_c = 0
            else:
                other_target_c = cols - 1

            cuts_to_draw.append(
                {
                    "start": other_sing,
                    "end": (target_r_o, other_target_c),
                    "type": "edge_drop",
                }
            )

    # --- Draw the Curves ---
    for item in cuts_to_draw:
        start = item["start"]
        end = item["end"]

        if item["type"] == "bridge":
            control = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            num_points = int(np.hypot(end[0] - start[0], end[1] - start[1]) * 1.5)

        elif item["type"] == "edge_drop":
            control = (end[0], start[1])

            # Calculate points
            dist = np.hypot(end[0] - start[0], end[1] - start[1])
            num_points = int(dist * 1.5)

        # Generate Curve Points
        curve_points = get_quadratic_bezier_points(
            start, end, control, max(10, num_points)
        )

        for r_float, c_float in curve_points:
            r_idx, c_idx = int(round(r_float)), int(round(c_float))

            # Bounds check
            if 0 <= r_idx < rows and 0 <= c_idx < cols:
                # Mask check (optional: stop if hitting background)
                if mask[r_idx, c_idx]:
                    branch_cuts[r_idx, c_idx] = True

    print(cuts_to_draw)
    return branch_cuts


def unwrap_with_curved_cuts(orientation_map, mask, singularities):
    """
    Unwraps the orientation map using the generated smooth curved cuts.
    """
    rows, cols = orientation_map.shape
    unwrapped = np.zeros_like(orientation_map)
    visited = np.zeros((rows, cols), dtype=bool)

    # 1. Generate Cuts
    cuts = get_smooth_curved_cuts((rows, cols), mask, singularities)

    # 2. Unwrap Walkable Regions
    walkable = mask & (~cuts)
    start_points = np.argwhere(walkable)
    if len(start_points) > 0:
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

    # 3. Fill Cuts
    cut_pixels = np.argwhere(cuts & mask)
    cut_queue = deque([tuple(p) for p in cut_pixels])

    while cut_queue:
        r, c = cut_queue.popleft()
        if visited[r, c]:
            continue

        found = False
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc]:
                # Simple unwrap relative to neighbor
                diff = orientation_map[r, c] - unwrapped[nr, nc]
                k = -np.round(diff / np.pi)
                unwrapped[r, c] = orientation_map[r, c] + k * np.pi
                visited[r, c] = True
                found = True
                break
        if not found:
            cut_queue.append((r, c))

    return unwrapped, cuts
