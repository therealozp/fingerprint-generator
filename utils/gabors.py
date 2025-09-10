import numpy as np
import torch


def generate_custom_gabor(
    size, theta, freq_0, freq_delta, gamma_0, gamma_delta, sigma=3.0, phase=0.0
):
    half = size // 2
    y, x = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    taper_x = 1 - (np.abs(x) / half)
    freq = freq_0 + freq_delta * taper_x
    gamma = gamma_0 + gamma_delta * taper_x

    envelope = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / sigma**2)
    wave = np.cos(2 * np.pi * freq * x_theta + phase)
    return envelope * wave


def generate_gabor_kernel(size, theta, freq, sigma=3.0, gamma=0.5, phase=0):
    """
    Generate a 2D Gabor kernel mathematically.

    Parameters:
        size  : kernel size (odd number)
        theta : orientation in radians
        freq  : spatial frequency (cycles per pixel)
        sigma : std deviation of Gaussian envelope
        gamma : spatial aspect ratio (y/x)
        phase : phase offset in radians

    Returns:
        2D NumPy array (size x size)
    """
    half = size // 2
    y, x = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / sigma**2)
    sinusoid = np.cos(2 * np.pi * freq * x_theta + phase)

    return gaussian * sinusoid


def torch_gabor(
    size,
    theta,
    freq_0,
    freq_delta,
    gamma_0,
    gamma_delta,
    sigma=3.0,
    phase=0,
):
    """
    Generate a 2D Gabor kernel mathematically.

    Parameters:
        size  : kernel size (odd number)
        theta : orientation in radians
        freq  : spatial frequency (cycles per pixel)
        sigma : std deviation of Gaussian envelope
        gamma : spatial aspect ratio (y/x)
        phase : phase offset in radians

    Returns:
        2D NumPy array (size x size)
    """
    half = size // 2
    y, x = torch.meshgrid(
        torch.linspace(-half, half + 1), torch.linspace(-half, half + 1)
    )

    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    taper_x = 1 - (torch.abs(x) / half)
    freq = freq_0 + freq_delta * taper_x
    gamma = gamma_0 + gamma_delta * taper_x

    gaussian = torch.exp(-0.5 * (x_theta**2 + (gamma * y_theta) ** 2) / sigma**2)
    sinusoid = torch.cos(2 * torch.pi * freq * x_theta + phase)

    return gaussian * sinusoid
