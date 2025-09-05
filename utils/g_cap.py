import numpy as np
import math

PI = np.pi


def get_q_alphaI(alpha):
    q = math.floor(4 * (PI + alpha) / PI)
    if q == 8:
        q = 7

    alpha_i = -PI + (PI * q / 4)

    return q, alpha_i


def g_loop1(g_cap, alpha):
    q, alpha_i = get_q_alphaI(alpha)

    return g_cap[1][1][q + 1] + ((4 * (alpha - alpha_i)) / PI) * (
        g_cap[1][1][q + 2] - g_cap[1][1][q + 1]
    )


def g_delta1(g_cap, alpha):
    q, alpha_i = get_q_alphaI(alpha)

    return g_cap[1][2][q + 1] + ((4 * (alpha - alpha_i)) / PI) * (
        g_cap[1][2][q + 2] - g_cap[1][2][q + 1]
    )


def g_loop2(g_cap, alpha):
    q, alpha_i = get_q_alphaI(alpha)

    return g_cap[2][1][q + 1] + ((4 * (alpha - alpha_i)) / PI) * (
        g_cap[2][1][q + 2] - g_cap[2][1][q + 1]
    )


def g_delta2(g_cap, alpha):
    q, alpha_i = get_q_alphaI(alpha)

    return g_cap[2][2][q + 1] + ((4 * (alpha - alpha_i)) / PI) * (
        g_cap[2][2][q + 2] - g_cap[2][2][q + 1]
    )
