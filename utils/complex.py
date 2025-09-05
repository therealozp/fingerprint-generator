import numpy as np


def dot(v, w):
    return v.x * w.x + v.y * w.y


def angle(v, w):
    l1 = v.length()
    l2 = w.length()

    dot_prod = dot(v, w)

    return np.arccos(dot_prod / (l1 * l2))


def arg(v):
    return np.atan2(v.y, v.x)


class Complex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def length(self):
        return (self.x**2 + self.y**2) ** 0.5

    def __repr__(self):
        return f"({self.x}, {self.y})"
