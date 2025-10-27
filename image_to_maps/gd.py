import math
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# -------- helpers: finite differences (reflective edges) --------
def grad_xy(u):
    # u: [1,1,H,W]
    gx = u[:, :, :, 1:] - u[:, :, :, :-1]
    gy = u[:, :, 1:, :] - u[:, :, :-1, :]
    # pad reflect to keep size
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    return gx, gy


def laplacian(u):
    # 4-neighbor Laplacian with reflective edges
    up = F.pad(u[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
    down = F.pad(u[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
    left = F.pad(u[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
    right = F.pad(u[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
    return (up + down + left + right) - 4.0 * u


# -------- main module: optimize a phase map psi --------
class PhaseOptim(nn.Module):
    def __init__(self, H, W, init="zeros"):
        super().__init__()
        if init == "zeros":
            psi0 = torch.zeros(1, 1, H, W)
        elif init == "randn":
            psi0 = 0.01 * torch.randn(1, 1, H, W)
        else:
            psi0 = init.clone().view(1, 1, H, W)
        self.psi = nn.Parameter(psi0)

    def forward(self):
        # I_pred in [0,1] (dark ridges if you flip the sign)
        I_pred = 0.5 * (1.0 + torch.cos(self.psi))
        return I_pred


class DecomposedPhaseOptim(nn.Module):
    def __init__(self, H, W, init="zeros"):
        super().__init__()
        if init == "zeros":
            thc = torch.zeros(1, 1, H, W)
        elif init == "randn":
            thc = math.pi * torch.randn(1, 1, H, W)
        else:
            thc = init.clone().view(1, 1, H, W)

        self.theta_cos = nn.Parameter(torch.cos(thc))
        self.freq = nn.Parameter(torch.ones(1, 1, H, W) * 0.1)  # default freq

    def forward(self):
        # compute the composite phase gradients based on the orientation and frequency
        theta_sin = torch.sqrt(1.0 - self.theta_cos**2 + 1e-8)
        exp_term = torch.complex(
            self.theta_cos + torch.pi / 2, self.theta_sin + torch.pi / 2
        )
        composite_phase_gradient = 2.0 * torch.pi * self.freq * exp_term
        print("composite_phase datatype", composite_phase_gradient.dtype)
        return composite_phase_gradient


# -------- loss builder (switch components on/off as needed) --------
def phase_losses(
    psi,
    I_tgt,
    f=None,
    th=None,
    ridge_theta=True,
    w_pix=1.0,
    w_grad=0.0,
    w_smooth=0.0,
    w_seed=0.0,
    seed_mask=None,
):
    """
    psi: [1,1,H,W] trainable phase
    I_tgt: [1,1,H,W] target image in [0,1]
    f: [1,1,H,W] frequency (cycles/pixel), optional if w_grad=0
    th: [1,1,H,W] orientation (radians), ridge or gradient depending on ridge_theta
    ridge_theta: True if th is ridge orientation; we convert to gradient orientation
    seed_mask: [1,1,H,W] in {0,1} where psi should be anchored to 0 (or desired value)
    """
    I_pred = 0.5 * (1.0 + torch.cos(psi))
    L_pix = F.l1_loss(I_pred, I_tgt)

    L_grad = torch.tensor(0.0, device=psi.device)
    if w_grad > 0.0:
        # build target phase gradient v = 2π f [cos θg, sin θg]
        assert f is not None and th is not None
        if ridge_theta:
            thg = (th + math.pi / 2.0) % (2 * math.pi)
        else:
            thg = th
        vx = 2.0 * math.pi * f * torch.cos(thg)
        vy = 2.0 * math.pi * f * torch.sin(thg)
        gx, gy = grad_xy(psi)
        L_grad = F.mse_loss(gx, vx) + F.mse_loss(gy, vy)

    L_smooth = torch.tensor(0.0, device=psi.device)
    if w_smooth > 0.0:
        lap = laplacian(psi)
        L_smooth = torch.mean(lap * lap)

    L_seed = torch.tensor(0.0, device=psi.device)
    if w_seed > 0.0 and seed_mask is not None:
        L_seed = torch.mean((psi * seed_mask) ** 2)

    L = w_pix * L_pix + w_grad * L_grad + w_smooth * L_smooth + w_seed * L_seed
    return L, {
        "L_pix": L_pix.item(),
        "L_grad": L_grad.item(),
        "L_smooth": L_smooth.item(),
        "L_seed": L_seed.item(),
    }


def unit_circle_regularization(
    theta_cos: torch.Tensor, theta_sin: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the loss term to enforce the unit circle constraint: cos^2 + sin^2 = 1.

    Args:
        theta_cos: Tensor representing the cosine component.
        theta_sin: Tensor representing the sine component.

    Returns:
        A scalar tensor representing the mean squared error loss.
    """
    # Calculate cos^2 + sin^2
    sum_of_squares = theta_cos**2 + theta_sin**2

    # Calculate the squared error from 1.0, and take the mean across all elements
    # This heavily penalizes deviations from the unit circle.
    L_constraint = F.mse_loss(sum_of_squares, torch.ones_like(sum_of_squares))

    return L_constraint


# -------- example training loop --------
def train_phase(
    I_tgt_np,
    f_np=None,
    th_np=None,
    ridge_theta=True,
    steps=2000,
    lr=1e-2,
    w_pix=1.0,
    w_grad=10.0,
    w_smooth=0.1,
    w_seed=0.0,
    seed_mask_np=None,
    init="zeros",
    device="cpu",
):
    """
    I_tgt_np: HxW in [0,1]
    f_np, th_np: HxW (optional if w_grad=0); th_np in radians
    seed_mask_np: HxW in {0,1} for optional phase anchoring
    """
    H, W = I_tgt_np.shape
    I_tgt = torch.as_tensor(I_tgt_np, dtype=torch.float32, device=device).view(
        1, 1, H, W
    )

    f = th = seed_mask = None
    if w_grad > 0.0:
        assert f_np is not None and th_np is not None
        f = torch.as_tensor(f_np, dtype=torch.float32, device=device).view(1, 1, H, W)
        th = torch.as_tensor(th_np, dtype=torch.float32, device=device).view(1, 1, H, W)
    if w_seed > 0.0 and seed_mask_np is not None:
        seed_mask = torch.as_tensor(
            seed_mask_np, dtype=torch.float32, device=device
        ).view(1, 1, H, W)

    model = PhaseOptim(H, W, init=init).to(device)
    opt = torch.optim.Adam([model.psi], lr=lr)

    pbar = tqdm(range(steps), desc="train_phase")
    for t in pbar:
        opt.zero_grad()
        psi = model.psi
        L, parts = phase_losses(
            psi, I_tgt, f, th, ridge_theta, w_pix, w_grad, w_smooth, w_seed, seed_mask
        )
        L.backward()
        opt.step()
        if (t + 1) % 200 == 0:
            msg = (
                f"step {t+1:4d}  L={L.item():.6f}  "
                f"pix={parts['L_pix']:.5f} grad={parts['L_grad']:.5f}  "
                f"smooth={parts['L_smooth']:.5f} seed={parts['L_seed']:.5f}"
            )
            pbar.write(msg)
            pbar.set_postfix(L=L.item(), pix=parts["L_pix"], grad=parts["L_grad"])

    with torch.no_grad():
        I_pred = 0.5 * (1.0 + torch.cos(model.psi))
    return (
        model.psi.detach().cpu().squeeze(0).squeeze(0).numpy(),
        I_pred.detach().cpu().squeeze(0).squeeze(0).numpy(),
    )


def train_decomposed_model(
    I_tgt_np,
    f_np=None,
    th_np=None,
    ridge_theta=True,
    steps=2000,
    lr=1e-2,
    w_pix=1.0,
    w_grad=10.0,
    w_smooth=0.1,
    w_seed=0.0,
    seed_mask_np=None,
    init="zeros",
    device="cpu",
):
    H, W = I_tgt_np.shape
    I_tgt = torch.as_tensor(I_tgt_np, dtype=torch.float32, device=device).view(
        1, 1, H, W
    )

    f = th = seed_mask = None
    if w_grad > 0.0:
        assert f_np is not None and th_np is not None
        f = torch.as_tensor(f_np, dtype=torch.float32, device=device).view(1, 1, H, W)
        th = torch.as_tensor(th_np, dtype=torch.float32, device=device).view(1, 1, H, W)
    if w_seed > 0.0 and seed_mask_np is not None:
        seed_mask = torch.as_tensor(
            seed_mask_np, dtype=torch.float32, device=device
        ).view(1, 1, H, W)

    model = DecomposedPhaseOptim(H, W, init=init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(steps), desc="train_phase")
    for t in pbar:
        opt.zero_grad()
        psi = model.forward()
        L, parts = phase_losses(
            psi, I_tgt, f, th, ridge_theta, w_pix, w_grad, w_smooth, w_seed, seed_mask
        )
        regularization_losses = unit_circle_regularization(
            model.theta_cos, model.theta_sin
        )
        L = L + regularization_losses
        L.backward()
        opt.step()
        if (t + 1) % 200 == 0:
            msg = (
                f"step {t+1:4d}  L={L.item():.6f}  "
                f"pix={parts['L_pix']:.5f} grad={parts['L_grad']:.5f}  "
                f"smooth={parts['L_smooth']:.5f} seed={parts['L_seed']:.5f}"
            )
            pbar.write(msg)
            pbar.set_postfix(L=L.item(), pix=parts["L_pix"], grad=parts["L_grad"])

    with torch.no_grad():
        I_pred = 0.5 * (1.0 + torch.cos(model.forward()))

    return (
        model.forward().detach().cpu().squeeze(0).squeeze(0).numpy(),
        I_pred.detach().cpu().squeeze(0).squeeze(0).numpy(),
    )


if __name__ == "__main__":
    # H, W = 256, 256
    # yy, xx = np.mgrid[:H, :W]
    # cx, cy = W // 2, H // 2
    # X = xx - cx
    # Y = yy - cy

    # kappa = 0.16  # controls spacing of rings (ridges)
    # psi_c = kappa * np.sqrt(X * X + Y * Y)
    # img = 0.5 * (1.0 - np.cos(psi_c))

    img = cv2.imread(
        "C:\\Users\\Oz\\code\\fingerprint-generator\\images\\50_whorl.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    rescaled_dims = img.shape[0] * 0.4
    img = cv2.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))
    img = img / 255.0

    H = img.shape[0]
    W = img.shape[1]

    psi_opt, I_pred = train_phase(
        img,
        w_pix=1.0,
        w_grad=0.0,
        w_smooth=0.001,
        steps=2000,
        lr=5e-3,
        init="randn",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # psi_opt, I_pred = train_decomposed_model(
    #     img,
    #     w_pix=1.0,
    #     w_grad=0.0,
    #     w_smooth=0.1,
    #     steps=2000,
    #     lr=5e-3,
    #     init="randn",
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("target image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(psi_opt, cmap="gray")
    plt.title("optimized phase")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(I_pred, cmap="gray")
    plt.title("predicted image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
