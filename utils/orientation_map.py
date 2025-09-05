import random
import math

from utils.complex import *
from utils.g_cap import *
import numpy as np

# Constants
PI = np.pi
rad_deg_fact = 180 / PI
deg_rad_fact = PI / 180


class OrientationMap:

    def __init__(
        self,
        width: int,
        height: int,
        singularity_type,
        delta_positions,
        core_positions,
        g_cap,
        arch_factor_1,
        arch_factor_2,
        k_arch,
    ):
        """Orientation map without external margin/padding complexity."""
        self.singularity_type = singularity_type
        self.o_map = np.zeros((height, width), dtype=float)

        self.width = width
        self.height = height

        self.delta_pos = delta_positions
        self.core_pos = core_positions

        self.g_cap = g_cap

        if self.singularity_type == 1:
            self.arch_factor_1 = arch_factor_1
            self.arch_factor_2 = arch_factor_2
            self.k_arch = k_arch

    def getOrientationAtPoint(self, i, j):
        local_orient = 0
        if self.singularity_type == 1:
            local_orient = np.arctan(
                max(
                    0.0,
                    (
                        self.k_arch
                        - self.k_arch * i / (self.height * self.arch_factor_2)
                    ),
                )
                * np.cos(j * PI / (self.width * self.arch_factor_1))
            )

        elif self.singularity_type in {2, 3, 4}:
            z = Complex(j, i)
            v1 = Complex(z.x - self.delta_pos[1].x, z.y - self.delta_pos[1].y)
            u1 = Complex(z.x - self.core_pos[1].x, z.y - self.core_pos[1].y)

            local_orient = 0.5 * (
                g_delta1(self.g_cap, arg(v1)) - g_loop1(self.g_cap, arg(u1))
            )

        elif self.singularity_type in {5, 6}:
            z = Complex(j, i)
            v1 = Complex(
                z.x - self.delta_pos[1].x,
                z.y - self.delta_pos[1].y,
            )
            u1 = Complex(
                z.x - self.core_pos[1].x,
                z.y - self.core_pos[1].y,
            )
            v2 = Complex(
                z.x - self.delta_pos[2].x,
                z.y - self.delta_pos[2].y,
            )
            u2 = Complex(
                z.x - self.core_pos[2].x,
                z.y - self.core_pos[2].y,
            )

            local_orient = 0.5 * (
                g_delta1(self.g_cap, arg(v1)) - g_loop1(self.g_cap, arg(u1))
            )
            local_orient += 0.5 * (
                g_delta2(self.g_cap, arg(v2)) - g_loop2(self.g_cap, arg(u2))
            )

        else:
            raise ValueError("Invalid singularity type")

        # Convert to degrees
        degrees = int(local_orient * rad_deg_fact)

        # Normalize degrees to [0, 180)
        if degrees > 0:
            degrees = degrees % 180
        elif degrees < 0:
            degrees = -((-degrees) % 180) + 180

        # Convert back to radians
        local_orient = degrees * deg_rad_fact

        return local_orient

    def fillOrientationMap(self):
        for i in range(self.height):
            for j in range(self.width):
                self.o_map[i, j] = self.getOrientationAtPoint(i, j)

    def getOrientationMap(self):
        self.fillOrientationMap()
        return self.o_map
