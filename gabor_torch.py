import torch
import numpy as np


def gabor_kernel_torch(theta, freq, sigma, gamma, size):
    half = size // 2
    y, x = torch.meshgrid(
        torch.linspace(-half, half, size, device=theta.device),
        torch.linspace(-half, half, size, device=theta.device),
        indexing="ij",
    )
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    envelope = torch.exp(-0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / sigma**2)
    wave = torch.cos(2 * np.pi * freq * x_theta)

    return envelope * wave  # shape: [size, size]


def generate_custom_gabor_torch(
    size,
    theta,
    freq_0,
    freq_delta,
    gamma_0,
    gamma_delta,
    sigma=3.0,
    phase=0.0,
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    theta = torch.tensor(theta, device=device)

    half = size // 2
    coords = torch.arange(-half, half + 1, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="xy")

    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    taper_x = 1 - (torch.abs(x) / half)
    freq = freq_0 + freq_delta * taper_x
    gamma = gamma_0 + gamma_delta * taper_x

    envelope = torch.exp(-0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / sigma**2)
    wave = torch.cos(2 * torch.pi * freq * x_theta + phase)

    return envelope * wave
