import numpy as np


def unwrap_orientation(O_init, mask=None):
    if mask is None:
        mask = np.ones_like(O_init, dtype=bool)

    h, w = O_init.shape
    O_u = np.zeros_like(O_init)
    visited = np.zeros_like(O_init, dtype=bool)

    start_r, start_c = 0, 0
    found = False
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                start_r, start_c = r, c
                found = True
                break
        if found:
            break

    if not found:
        return O_u

    O_u[start_r, start_c] = O_init[start_r, start_c]
    visited[start_r, start_c] = True

    queue = [(start_r, start_c)]

    while queue:
        r, c = queue.pop(0)

        neighbors = []
        if r > 0:
            neighbors.append((r - 1, c))
        if r < h - 1:
            neighbors.append((r + 1, c))
        if c > 0:
            neighbors.append((r, c - 1))
        if c < w - 1:
            neighbors.append((r, c + 1))

        for nr, nc in neighbors:
            if mask[nr, nc] and not visited[nr, nc]:
                diff = O_init[nr, nc] - O_u[r, c]
                k = np.round(diff / np.pi)
                O_u[nr, nc] = O_init[nr, nc] - k * np.pi
                visited[nr, nc] = True
                queue.append((nr, nc))

    return O_u


def reconstruct_continuous_phase(G_cx, G_cy, mask=None, block_size=8):
    if mask is None:
        mask = np.ones_like(G_cx, dtype=bool)

    h_blocks, w_blocks = G_cx.shape
    h_img, w_img = h_blocks * block_size, w_blocks * block_size

    Psi_c = np.zeros((h_img, w_img))
    P = np.zeros((h_blocks, w_blocks))
    reconstructed = np.zeros((h_blocks, w_blocks), dtype=bool)

    start_r, start_c = 0, 0
    found = False
    for r in range(h_blocks):
        for c in range(w_blocks):
            if mask[r, c]:
                start_r, start_c = r, c
                found = True
                break
        if found:
            break

    if not found:
        return Psi_c

    P[start_r, start_c] = 0
    reconstructed[start_r, start_c] = True
    queue = [(start_r, start_c)]

    while queue:
        m, n = queue.pop(0)

        neighbors = []
        if m > 0:
            neighbors.append(((m - 1, n), "top"))
        if m < h_blocks - 1:
            neighbors.append(((m + 1, n), "bottom"))
        if n > 0:
            neighbors.append(((m, n - 1), "left"))
        if n < w_blocks - 1:
            neighbors.append(((m, n + 1), "right"))

        for (nm, nn), pos in neighbors:
            if mask[nm, nn] and not reconstructed[nm, nn]:
                offsets = []

                check_neighbors = []
                if nm > 0:
                    check_neighbors.append(((nm - 1, nn), "top"))
                if nm < h_blocks - 1:
                    check_neighbors.append(((nm + 1, nn), "bottom"))
                if nn > 0:
                    check_neighbors.append(((nm, nn - 1), "left"))
                if nn < w_blocks - 1:
                    check_neighbors.append(((nm, nn + 1), "right"))

                for (rm, rn), rpos in check_neighbors:
                    if reconstructed[rm, rn]:

                        if rpos == "top":  # Neighbor is above current
                            ys = np.array([nm * block_size - 1] * block_size)
                            xs = np.arange(nm * block_size, (nm + 1) * block_size)
                        elif rpos == "bottom":
                            ys = np.array([(nm + 1) * block_size] * block_size)
                            xs = np.arange(nm * block_size, (nm + 1) * block_size)
                        elif rpos == "left":
                            xs = np.array(
                                [nm * block_size - 1] * block_size
                            )  # boundary is logic dependent
                            ys = np.arange(nn * block_size, (nn + 1) * block_size)
                            # Simply: use border pixels
                            pass

                        # Simplified border pixel logic per Eq (15)
                        # We need boundary pixels between block (nm, nn) and (rm, rn)
                        if rm == nm - 1:  # Neighbor is top
                            border_y = nm * block_size
                            border_x = np.arange(nn * block_size, (nn + 1) * block_size)
                            # Use previous block (rm, rn) parameters at border
                            prev_val = (
                                G_cx[rm, rn] * border_x
                                + G_cy[rm, rn] * (border_y - 1)
                                + P[rm, rn]
                            )
                            curr_pred = (
                                G_cx[nm, nn] * border_x + G_cy[nm, nn] * border_y
                            )

                        elif rm == nm + 1:  # Neighbor is bottom
                            border_y = (nm + 1) * block_size
                            border_x = np.arange(nn * block_size, (nn + 1) * block_size)
                            prev_val = (
                                G_cx[rm, rn] * border_x
                                + G_cy[rm, rn] * border_y
                                + P[rm, rn]
                            )
                            curr_pred = G_cx[nm, nn] * border_x + G_cy[nm, nn] * (
                                border_y - 1
                            )

                        elif rn == nn - 1:  # Neighbor is left
                            border_x = nn * block_size
                            border_y = np.arange(nm * block_size, (nm + 1) * block_size)
                            prev_val = (
                                G_cx[rm, rn] * (border_x - 1)
                                + G_cy[rm, rn] * border_y
                                + P[rm, rn]
                            )
                            curr_pred = (
                                G_cx[nm, nn] * border_x + G_cy[nm, nn] * border_y
                            )

                        elif rn == nn + 1:  # Neighbor is right
                            border_x = (nn + 1) * block_size
                            border_y = np.arange(nm * block_size, (nm + 1) * block_size)
                            prev_val = (
                                G_cx[rm, rn] * border_x
                                + G_cy[rm, rn] * border_y
                                + P[rm, rn]
                            )
                            curr_pred = (
                                G_cx[nm, nn] * (border_x - 1) + G_cy[nm, nn] * border_y
                            )

                        diffs = prev_val - curr_pred
                        offsets.extend(diffs)

                if offsets:
                    # Average complex numbers as per text [365]
                    complex_offsets = np.exp(1j * np.array(offsets))
                    mean_complex = np.mean(complex_offsets)
                    P[nm, nn] = np.angle(mean_complex)
                    # Note: Since P is an absolute offset, usually simple mean is enough if not wrapped
                    # But paper says "phase values cannot be averaged directly" [365]
                    # However, Eq 14 implies P is a scalar offset to a plane.
                    # If P is just a scalar added to Gx + Gy, it might not be wrapped
                    # but let's stick to the paper's explicit instruction.
                    P[nm, nn] = np.mean(
                        offsets
                    )  # Using simple mean for linear offset P

                    reconstructed[nm, nn] = True
                    queue.append((nm, nn))

    # Generate full image
    y_grid, x_grid = np.mgrid[0:h_img, 0:w_img]

    # Map blocks to pixels
    G_cx_img = np.repeat(np.repeat(G_cx, block_size, axis=0), block_size, axis=1)
    G_cy_img = np.repeat(np.repeat(G_cy, block_size, axis=0), block_size, axis=1)
    P_img = np.repeat(np.repeat(P, block_size, axis=0), block_size, axis=1)

    Psi_c = G_cx_img * x_grid + G_cy_img * y_grid + P_img

    return Psi_c
