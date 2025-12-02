import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2
import os

from plotting_utils import plot_results


class DecomposedPhase(nn.Module):
    def __init__(self, H=256, W=256, init_freq=0.1, init_theta=None, smooth=True):
        super(DecomposedPhase, self).__init__()
        self.phase = nn.Parameter(torch.randn(H, W))

        if smooth:
            x_coords = torch.linspace(torch.pi, 2 * torch.pi, W)
            y_coords = torch.linspace(torch.pi, 2 * torch.pi, H)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            phase_init = xx
            self.phase = nn.Parameter(phase_init.clone())

        if init_theta is not None:
            self.theta_cos = nn.Parameter(init_theta)
        else:
            init_angle = torch.pi / 4.0
            self.theta_cos = nn.Parameter(torch.full((H, W), init_angle))
        self.freq = nn.Parameter(torch.full((H, W), init_freq))

    def forward(self, x=None):
        theta_cos = torch.cos(self.theta_cos)
        theta_sin = torch.sin(self.theta_cos)

        phase_gradient_x = 2.0 * torch.pi * torch.abs(self.freq) * theta_cos
        phase_gradient_y = 2.0 * torch.pi * torch.abs(self.freq) * theta_sin

        I_pred = 0.5 * (1.0 + torch.cos(self.phase))

        theta = torch.atan2(theta_sin, theta_cos)

        return {
            "I_pred": I_pred,
            "phase": self.phase,
            "phase_gradient_x": phase_gradient_x,
            "phase_gradient_y": phase_gradient_y,
            "freq": torch.abs(self.freq),
            "theta": theta,
            "theta_cos": theta_cos,
        }
