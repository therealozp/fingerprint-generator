import numpy as np
from utils.complex import *

# Constants
PI = np.pi


def rand():
    return np.random.rand()


# Helper function to set g_cap values
def set_g_cap(g_cap, index, u, v, f1, f2):
    g_cap[index][1][1] = -PI + u
    g_cap[index][1][2] = -3 * PI / 4 + f1 * u
    g_cap[index][1][3] = -PI / 2
    g_cap[index][1][4] = -PI / 4 + f1 * v
    g_cap[index][1][5] = v
    g_cap[index][1][6] = PI / 4 + f2 * v
    g_cap[index][1][7] = PI / 2
    g_cap[index][1][8] = 3 * PI / 4 + f2 * u
    g_cap[index][1][9] = PI + u

    # Set delta values
    g_cap[index][2][1] = -PI
    g_cap[index][2][2] = -3 * PI / 4
    g_cap[index][2][3] = -PI / 2
    g_cap[index][2][4] = -PI / 4
    g_cap[index][2][5] = 0
    g_cap[index][2][6] = PI / 4
    g_cap[index][2][7] = PI / 2
    g_cap[index][2][8] = 3 * PI / 4
    g_cap[index][2][9] = PI


def set_param(singularity_type):
    g_cap = np.zeros((3, 3, 10))
    u1, v1 = 0, 0

    if singularity_type in {2, 3, 4, 5, 6}:
        # Define f1, f2, u, v based on singularity type
        f1 = 2.0 / 3.0
        f2 = 2.0 / 3.0 if singularity_type != 4 else 1.0

        u = -PI / 2 + rand() * (PI / 4)
        v = PI / 4 + rand() * (PI / 4)

        if singularity_type == 4:
            u = -2 * PI / 3 + rand() * (PI / 4)
            v = 5 * PI / 18 + rand() * (PI / 4)
        elif singularity_type == 5 or singularity_type == 6:
            u = -PI / 3 + rand() * (PI / 12)
            v = 2 * PI / 9 + rand() * (PI / 12)

        set_g_cap(g_cap, 1, u, v, f1, f2)

        # Additional settings for double-loop patterns
        if singularity_type in {5, 6}:
            u = PI / 18 + rand() * (PI / 12)
            v = PI / 6 + rand() * (PI / 12)

            g_cap[2][1][1] = -PI + u
            g_cap[2][1][2] = -3 * PI / 4 + f1 * u
            g_cap[2][1][3] = -PI / 2
            g_cap[2][1][4] = -PI / 4 + f1 * v
            g_cap[2][1][5] = v
            g_cap[2][1][6] = PI / 4 + f2 * v
            g_cap[2][1][7] = PI / 2
            g_cap[2][1][8] = 3 * PI / 4 + f2 * u
            g_cap[2][1][9] = PI + u

            g_cap[2][2][1] = -PI + 2 * u1 / 3
            g_cap[2][2][2] = -3 * PI / 4
            g_cap[2][2][3] = -PI / 2
            g_cap[2][2][4] = -PI / 4
            g_cap[2][2][5] = 2 * v1 / 3
            g_cap[2][2][6] = PI / 4 + v1
            g_cap[2][2][7] = PI / 2 + 2 * (u1 + v1) / 3
            g_cap[2][2][8] = 3 * PI / 4 + u1
            g_cap[2][2][9] = PI + 2 * u1 / 3

    return g_cap.copy()


def set_param_canonical(singularity_type):
    g_cap = np.zeros((3, 3, 10))

    if singularity_type == 2:
        f1 = 2.0 / 3.0
        f2 = 2.0 / 3.0
        u = -90 * PI / 180.0 + rand() * (45 * PI / 180.0)
        v = 45 * PI / 180.0 + rand() * (45 * PI / 180.0)

        g_cap[1][1][1] = -PI + u
        g_cap[1][1][2] = -3 * PI / 4 + f1 * u
        g_cap[1][1][3] = -PI / 2
        g_cap[1][1][4] = -PI / 4 + f1 * v
        g_cap[1][1][5] = v
        g_cap[1][1][6] = PI / 4 + f1 * v
        g_cap[1][1][7] = PI / 2
        g_cap[1][1][8] = 3 * PI / 4 + f1 * u
        g_cap[1][1][9] = PI + u

        u1 = 0
        v1 = 0

        g_cap[1][2][1] = -PI + 2 * u1 / 3
        g_cap[1][2][2] = -3 * PI / 4
        g_cap[1][2][3] = -PI / 2
        g_cap[1][2][4] = -PI / 4
        g_cap[1][2][5] = 2 * v1 / 3
        g_cap[1][2][6] = PI / 4 + v1
        g_cap[1][2][7] = PI / 2 + 2 * (u1 + v1) / 3
        g_cap[1][2][8] = 3 * PI / 4 + u1
        g_cap[1][2][9] = PI + 2 * u1 / 3
    elif singularity_type == 3:
        f1 = 2.0 / 3.0
        u = -90 * PI / 180.0 + rand() * (45 * PI / 180.0)
        v = 60 * PI / 180.0 + rand() * (45 * PI / 180.0)

        g_cap[1][1][1] = -PI + u
        g_cap[1][1][2] = -3 * PI / 4 + f1 * u
        g_cap[1][1][3] = -PI / 2
        g_cap[1][1][4] = -PI / 4 + f1 * v
        g_cap[1][1][5] = v
        g_cap[1][1][6] = PI / 4 + f1 * v
        g_cap[1][1][7] = PI / 2
        g_cap[1][1][8] = 3 * PI / 4 + f1 * u
        g_cap[1][1][9] = PI + u

        u1 = 0
        v1 = 0

        g_cap[1][2][1] = -PI + 2 * u1 / 3
        g_cap[1][2][2] = -3 * PI / 4
        g_cap[1][2][3] = -PI / 2
        g_cap[1][2][4] = -PI / 4
        g_cap[1][2][5] = 2 * v1 / 3
        g_cap[1][2][6] = PI / 4 + v1
        g_cap[1][2][7] = PI / 2 + 2 * (u1 + v1) / 3
        g_cap[1][2][8] = 3 * PI / 4 + u1
        g_cap[1][2][9] = PI + 2 * u1 / 3

    if singularity_type == 4:
        f1 = 2.0 / 3.0
        f2 = 3.0 / 3.0
        u = -120 * PI / 180.0 + rand() * (45 * PI / 180.0)
        v = 50 * PI / 180.0 + rand() * (45 * PI / 180.0)

        g_cap[1][1][1] = -PI + u
        g_cap[1][1][2] = -3 * PI / 4 + f1 * u
        g_cap[1][1][3] = -PI / 2
        g_cap[1][1][4] = -PI / 4 + f1 * v
        g_cap[1][1][5] = v
        g_cap[1][1][6] = PI / 4 + f1 * v
        g_cap[1][1][7] = PI / 2
        g_cap[1][1][8] = 3 * PI / 4 + f1 * u
        g_cap[1][1][9] = PI + u

        u1 = 0
        v1 = 0

        g_cap[1][2][1] = -PI + 2 * u1 / 3
        g_cap[1][2][2] = -3 * PI / 4
        g_cap[1][2][3] = -PI / 2
        g_cap[1][2][4] = -PI / 4
        g_cap[1][2][5] = 2 * v1 / 3
        g_cap[1][2][6] = PI / 4 + v1
        g_cap[1][2][7] = PI / 2 + 2 * (u1 + v1) / 3
        g_cap[1][2][8] = 3 * PI / 4 + u1
        g_cap[1][2][9] = PI + 2 * u1 / 3

    if singularity_type == 5 or singularity_type == 6:
        f1 = 2.0 / 3.0
        f2 = 2.0 / 3.0

        u = -60 * PI / 180.0 + rand() * (15 * PI / 180.0)
        v = 40 * PI / 180.0 + rand() * (15 * PI / 180.0)

        g_cap[1][1][1] = -PI + u
        g_cap[1][1][2] = -3 * PI / 4 + f1 * u
        g_cap[1][1][3] = -PI / 2
        g_cap[1][1][4] = -PI / 4 + f1 * v
        g_cap[1][1][5] = v
        g_cap[1][1][6] = PI / 4 + f2 * v
        g_cap[1][1][7] = PI / 2
        g_cap[1][1][8] = 3 * PI / 4 + f2 * u
        g_cap[1][1][9] = PI + u

        u1 = 0
        v1 = 0

        g_cap[1][2][1] = -PI + 2 * u1 / 3
        g_cap[1][2][2] = -3 * PI / 4
        g_cap[1][2][3] = -PI / 2
        g_cap[1][2][4] = -PI / 4
        g_cap[1][2][5] = 2 * v1 / 3
        g_cap[1][2][6] = PI / 4 + v1
        g_cap[1][2][7] = PI / 2 + 2 * (u1 + v1) / 3
        g_cap[1][2][8] = 3 * PI / 4 + u1
        g_cap[1][2][9] = PI + 2 * u1 / 3

        u = 10 * PI / 180.0 + rand() * (15 * PI / 180.0)
        v = 30 * PI / 180.0 + rand() * (15 * PI / 180.0)
        g_cap[2][1][1] = -PI + u
        g_cap[2][1][2] = -3 * PI / 4 + f1 * u
        g_cap[2][1][3] = -PI / 2
        g_cap[2][1][4] = -PI / 4 + f1 * v
        g_cap[2][1][5] = v
        g_cap[2][1][6] = PI / 4 + f2 * v
        g_cap[2][1][7] = PI / 2
        g_cap[2][1][8] = 3 * PI / 4 + f2 * u
        g_cap[2][1][9] = PI + u

        u1 = 0
        v1 = 0

        g_cap[2][2][1] = -PI + 2 * u1 / 3
        g_cap[2][2][2] = -3 * PI / 4
        g_cap[2][2][3] = -PI / 2
        g_cap[2][2][4] = -PI / 4
        g_cap[2][2][5] = 2 * v1 / 3
        g_cap[2][2][6] = PI / 4 + v1
        g_cap[2][2][7] = PI / 2 + 2 * (u1 + v1) / 3
        g_cap[2][2][8] = 3 * PI / 4 + u1
        g_cap[2][2][9] = PI + 2 * u1 / 3

    return g_cap.copy()


def init_para(H, W, singularity_type: int):
    """
    Initialize core and delta positions based on singularity type.
    """
    core_positions = [Complex(0, 0) for _ in range(3)]
    delta_positions = [Complex(0, 0) for _ in range(3)]

    arch_fact1 = 0.0
    arch_fact2 = 0.0
    k_arch = 0.0
    # Define core and delta regions
    core_region = {
        "crx1": int(W * 0.4),
        "crx2": int(W - W * 0.4),
        "cry1": int(H * 0.35),
        "cry2": int(H - H * 0.45),
    }
    delta_region = {
        "d1x1": int(W * 0.12),
        "d1x2": int(W - W * 0.675),
        "d1y1": int(H * 0.625),
        "d1y2": int(H - H * 0.185),
    }
    delta_region2 = {
        "d2x1": int(W * 0.675),
        "d2x2": int(W - W * 0.12),
        "d2y1": int(H * 0.625),
        "d2y2": int(H - H * 0.185),
    }

    if singularity_type == 1:
        arch_fact1 = 0.8 + 0.4 * rand()
        arch_fact2 = 0.6 + 0.8 * rand()
        k_arch = 1.2 + rand() * 1.5

    elif singularity_type == 2:
        core_positions[1].x = int(
            core_region["crx1"] + rand() * (core_region["crx2"] - core_region["crx1"])
        )
        core_positions[1].y = int(
            core_region["cry1"] + rand() * (core_region["cry2"] - core_region["cry1"])
        )
        delta_positions[1].x = core_positions[1].x
        delta_positions[1].y = int(
            delta_region["d1y1"]
            + rand() * (delta_region["d1y2"] - delta_region["d1y1"])
        )

    elif singularity_type == 3:
        core_positions[1].x = int(
            core_region["crx1"] + rand() * (core_region["crx2"] - core_region["crx1"])
        )
        core_positions[1].y = int(
            core_region["cry1"] + rand() * (core_region["cry2"] - core_region["cry1"])
        )
        delta_positions[1].x = int(
            delta_region["d1x1"]
            + rand() * (delta_region["d1x2"] - delta_region["d1x1"])
        )
        delta_positions[1].y = int(
            delta_region["d1y1"]
            + rand() * (delta_region["d1y2"] - delta_region["d1y1"])
        )

    elif singularity_type == 4:
        core_positions[1].x = int(
            core_region["crx1"] + rand() * (core_region["crx2"] - core_region["crx1"])
        )
        core_positions[1].y = int(
            core_region["cry1"] + rand() * (core_region["cry2"] - core_region["cry1"])
        )
        delta_positions[1].x = int(
            delta_region2["d2x1"]
            + rand() * (delta_region2["d2x2"] - delta_region2["d2x1"])
        )
        delta_positions[1].y = int(
            delta_region2["d2y1"]
            + rand() * (delta_region2["d2y2"] - delta_region2["d2y1"])
        )

    elif singularity_type == 5:
        core_positions[1].x = int(W * 0.4 + rand() * (W * 0.5 - W * 0.4))
        core_positions[1].y = int(H * 0.325 + rand() * (H * 0.45 - H * 0.325))
        core_positions[2].x = int(W * 0.5 + rand() * (W * 0.6 - W * 0.5))
        core_positions[2].y = int(H * 0.475 + rand() * (H * 0.6 - H * 0.475))

        delta_positions[1].x = int(W * 0.05 + rand() * (W - W * 0.75 - W * 0.05))
        delta_positions[1].y = int(H * 0.625 + rand() * (H * 0.875 - H * 0.625))
        delta_positions[2].x = int(W * 0.75 + rand() * (W - W * 0.05 - W * 0.75))
        delta_positions[2].y = int(H * 0.65 + rand() * (H * 0.85 - H * 0.65))

    elif singularity_type == 6:
        tmp1 = int(H * 0.4 + rand() * (H * 0.575 - H * 0.4))
        tmp2 = int(H * 0.4 + rand() * (H * 0.575 - H * 0.4))
        core_positions[1].x = core_positions[2].x = int(
            W * 0.4 + rand() * (W * 0.6 - W * 0.4)
        )
        if tmp1 < tmp2:
            core_positions[1].y, core_positions[2].y = tmp1, tmp2
        else:
            core_positions[1].y, core_positions[2].y = tmp2, tmp1

        tmp1 = int(
            delta_region["d1y1"]
            + rand() * (delta_region["d1y2"] - delta_region["d1y1"])
        )
        tmp2 = int(
            delta_region2["d2y1"]
            + rand() * (delta_region2["d2y2"] - delta_region2["d2y1"])
        )
        delta_positions[1].x = int(
            delta_region["d1x1"]
            + rand() * (delta_region["d1x2"] - delta_region["d1x1"])
        )
        delta_positions[2].x = int(
            delta_region2["d2x1"]
            + rand() * (delta_region2["d2x2"] - delta_region2["d2x1"])
        )
        if tmp1 < tmp2:
            delta_positions[1].y, delta_positions[2].y = tmp1, tmp2
        else:
            delta_positions[1].y, delta_positions[2].y = tmp2, tmp1

    return core_positions, delta_positions, arch_fact1, arch_fact2, k_arch


def init_para_canonical(H: int, W: int, singularity_type: int):
    core_positions = [Complex(0, 0) for _ in range(3)]
    delta_positions = [Complex(0, 0) for _ in range(3)]

    arch_fact1 = 0
    arch_fact2 = 0
    k_arch = 0

    crx1 = np.floor(W * 0.4)
    crx2 = np.floor(W - W * 0.4)
    cry1 = np.floor(H * 0.35)
    cry2 = np.floor(H - H * 0.45)

    cwx1 = np.floor(W * 0.4)
    cwx2 = np.floor(W - W * 0.4)
    cwy1 = np.floor(H * 0.40)
    cwy2 = np.floor(H * 0.575)

    cw1x1 = np.floor(W * 0.4)
    cw1x2 = np.floor(W * 0.5)
    cw1y1 = np.floor(H * 0.325)
    cw1y2 = np.floor(H * 0.45)

    cw2x1 = np.floor(W * (0.5))
    cw2x2 = np.floor(W * (0.6))
    cw2y1 = np.floor(H * 0.475)
    cw2y2 = np.floor(H * 0.6)

    dw1x1 = np.floor(W * 0.05)
    dw1x2 = np.floor(W - W * 0.75)
    dw1y1 = np.floor(H * 0.625)
    dw1y2 = np.floor(H * 0.875)

    dw2x1 = np.floor(W * 0.75)
    dw2x2 = np.floor(W - W * 0.05)
    dw2y1 = np.floor(H * 0.65)
    dw2y2 = np.floor(H * 0.85)

    d0x1 = crx1
    d0x2 = crx2
    d0y1 = np.floor(H * 0.6)
    d0y2 = np.floor(H - H * 0.3)

    d1x1 = np.floor(W * 0.12)
    d1x2 = np.floor(W - W * 0.675)
    d1y1 = np.floor(H * 0.625)
    d1y2 = np.floor(H - H * 0.185)

    d2x1 = np.floor(W * 0.675)
    d2x2 = np.floor(W - W * 0.12)
    d2y1 = np.floor(H * 0.625)
    d2y2 = np.floor(H - H * 0.185)

    if singularity_type == 1:
        arch_fact1 = 0.8 + 0.4 * rand()
        arch_fact2 = 0.6 + 0.8 * rand()
        k_arch = 1.2 + rand() * 1.5
    elif singularity_type == 2:
        core_positions[1].x = np.floor(crx1 + rand() * (crx2 - crx1))
        core_positions[1].y = np.floor(cry1 + rand() * (cry2 - cry1))

        delta_positions[1].x = core_positions[1].x
        delta_positions[1].y = np.floor(d0y1 + rand() * (d0y2 - d0y1))

    elif singularity_type == 3:
        core_positions[1].x = np.floor(crx1 + rand() * (crx2 - crx1))
        core_positions[1].y = np.floor(cry1 + rand() * (cry2 - cry1))

        delta_positions[1].x = np.floor(d1x1 + rand() * (d1x2 - d1x1))
        delta_positions[1].y = np.floor(d1y1 + rand() * (d1y2 - d1y1))

    elif singularity_type == 4:
        core_positions[1].x = np.floor(crx1 + rand() * (crx2 - crx1))
        core_positions[1].y = np.floor(cry1 + rand() * (cry2 - cry1))

        delta_positions[1].x = np.floor(d2x1 + rand() * (d2x2 - d2x1))
        delta_positions[1].y = np.floor(d2y1 + rand() * (d2y2 - d2y1))

    elif singularity_type == 5:
        core_positions[1].x = np.floor(cw1x1 + rand() * (cw1x2 - cw1x1))
        core_positions[1].y = np.floor(cw1y1 + rand() * (cw1y2 - cw1y1))

        core_positions[2].x = np.floor(cw2x1 + rand() * (cw2x2 - cw2x1))
        core_positions[2].y = np.floor(cw2y1 + rand() * (cw2y2 - cw2y1))

        delta_positions[1].x = np.floor(dw1x1 + rand() * (dw1x2 - dw1x1))
        delta_positions[1].y = np.floor(dw1y1 + rand() * (dw1y2 - dw1y1))

        delta_positions[2].x = np.floor(dw2x1 + rand() * (dw2x2 - dw2x1))
        delta_positions[2].y = np.floor(dw2y1 + rand() * (dw2y2 - dw2y1))
    elif singularity_type == 6:
        core_positions[1].x = np.floor(cwx1 + rand() * (cwx2 - cwx1))
        core_positions[2].x = core_positions[1].x

        tmp1 = np.floor(cwy1 + rand() * (cwy2 - cwy1))
        tmp2 = np.floor(cwy1 + rand() * (cwy2 - cwy1))
        if tmp1 < tmp2:
            core_positions[1].y = tmp1
            core_positions[2].y = tmp2
        else:
            core_positions[1].y = tmp2
            core_positions[2].y = tmp1

        delta_positions[1].x = np.floor(d1x1 + rand() * (d1x2 - d1x1))

        delta_positions[2].x = np.floor(d2x1 + rand() * (d2x2 - d2x1))

        tmp1 = np.floor(d1y1 + rand() * (d1y2 - d1y1))
        tmp2 = np.floor(d2y1 + rand() * (d2y2 - d2y1))

        if tmp1 < tmp2:
            delta_positions[1].y = tmp1
            delta_positions[2].y = tmp2
        else:
            delta_positions[1].y = tmp2
            delta_positions[2].y = tmp1

    return core_positions, delta_positions, arch_fact1, arch_fact2, k_arch
