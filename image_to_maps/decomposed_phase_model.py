import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2
import os

from plotting_utils import (
    plot_results,
    plot_first_order_grads,
    plot_second_order_grads,
    plot_change_in_value_per_stripe,
)


class DecomposedPhase(nn.Module):
    def __init__(
        self,
        H=256,
        W=256,
        init_freq=0.1,
        init_theta=None,
        init_phase=None,
        smooth=True,
        spiral_phase_coords=[],
        spiral_phase_polarities=[],
    ):
        assert len(spiral_phase_coords) == len(
            spiral_phase_polarities
        ), "spiral_phase_coords and spiral_phase_polarities must have the same length"

        super(DecomposedPhase, self).__init__()

        if init_phase is not None:
            self.phase = nn.Parameter(torch.tensor(init_phase, dtype=torch.float32))
        elif smooth:
            x_coords = torch.linspace(torch.pi, 2 * torch.pi, W)
            y_coords = torch.linspace(torch.pi, 2 * torch.pi, H)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            phase_init = xx
            self.phase = nn.Parameter(phase_init.clone())
        else:
            self.phase = nn.Parameter(torch.zeros(H, W))
        spiral_phase = torch.zeros(H, W)
        y_range = torch.arange(H)
        x_range = torch.arange(W)

        Y, X = torch.meshgrid(y_range, x_range, indexing="ij")
        for (yy, xx), polarity in zip(spiral_phase_coords, spiral_phase_polarities):
            spiral_phase += polarity * torch.arctan2(Y - yy, X - xx)

        self.spiral_phase = nn.Parameter(spiral_phase, requires_grad=False)

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

        I_pred = 0.5 * (1.0 - torch.cos(self.phase + self.spiral_phase))

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


def get_blockwise_orientation(
    img: torch.Tensor,
    block_size: int = 8,
):
    dev = img.device
    dtype = img.dtype

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=dev
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=dev
    ).view(1, 1, 3, 3)

    Gx = F.conv2d(img, sobel_x, padding=1)
    Gy = F.conv2d(img, sobel_y, padding=1)

    Gx2 = Gx * Gx
    Gy2 = Gy * Gy
    Gxy = Gx * Gy

    kernel = torch.ones(
        (1, 1, block_size, block_size), dtype=Gx2.dtype, device=Gx2.device
    )

    sum_Gx2 = F.avg_pool2d(Gx2, kernel_size=block_size, stride=block_size) * (
        block_size * block_size
    )
    sum_Gy2 = F.avg_pool2d(Gy2, kernel_size=block_size, stride=block_size) * (
        block_size * block_size
    )
    sum_Gxy = F.avg_pool2d(Gxy, kernel_size=block_size, stride=block_size) * (
        block_size * block_size
    )

    Vx = 2 * sum_Gxy.squeeze(0).squeeze(0)
    Vy = sum_Gx2.squeeze(0).squeeze(0) - sum_Gy2.squeeze(0).squeeze(0)

    theta = 0.5 * torch.atan2(Vx, Vy + 1e-8)
    orientation_map = torch.remainder(theta, torch.pi)

    return orientation_map


def laplacian_2d(u: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian for [H, W] tensor."""
    u_4d = u.unsqueeze(0).unsqueeze(0)
    u_padded = F.pad(u_4d, (1, 1, 1, 1), mode="replicate")

    up = u_padded[0, 0, :-2, 1:-1]
    down = u_padded[0, 0, 2:, 1:-1]
    left = u_padded[0, 0, 1:-1, :-2]
    right = u_padded[0, 0, 1:-1, 2:]
    center = u

    return (up + down + left + right) - 4.0 * center


import torch
import torch.nn.functional as F


def alignment_loss(phase_image, epsilon=1e-8):
    # Ensure input is 4D for padding operations if needed, or handle slicing directly
    if phase_image.dim() == 2:
        phase_image = phase_image.unsqueeze(0)  # Add Batch dim

    # dy: Gradient in vertical direction
    dy = phase_image[:, 1:, :] - phase_image[:, :-1, :]
    # dx: Gradient in horizontal direction
    dx = phase_image[:, :, 1:] - phase_image[:, :, :-1]

    # Align dimensions (trim the last row/col to match sizes)
    # We create a vector field G of shape (Batch, 2, H-1, W-1)
    g_y = dy[:, :, :-1]
    g_x = dx[:, :-1, :]

    # Stack to create gradient vectors: shape [Batch, 2, H-1, W-1]
    gradients = torch.stack((g_x, g_y), dim=1)

    # 2. Normalize Vectors (We only care about Direction, not Magnitude)
    # This prevents the massive magnitude of a "cliff" from dominating the loss
    magnitudes = torch.norm(gradients, dim=1, keepdim=True)
    normalized_grads = gradients / (magnitudes + epsilon)

    # 3. Compute Cosine Similarity with Neighbors
    # We compare pixel (i, j) with (i+1, j) and (i, j+1)

    # Slice the normalized field
    current_pixel = normalized_grads[:, :, :-1, :-1]  # Top-Left portion
    neighbor_right = normalized_grads[:, :, :-1, 1:]  # Shifted Right
    neighbor_down = normalized_grads[:, :, 1:, :-1]  # Shifted Down

    # Dot product of normalized vectors = Cosine Similarity
    # dim=1 is the channel dimension (x,y components)
    sim_right = torch.sum(current_pixel * neighbor_right, dim=1)
    sim_down = torch.sum(current_pixel * neighbor_down, dim=1)

    # 4. The Loss Function
    # We want similarity to be +1.
    # Loss = 1 - similarity.
    # Range: 0 (perfect alignment) to 2 (perfect reversal/fold)
    loss_x = 1.0 - sim_right
    loss_y = 1.0 - sim_down

    return torch.mean(loss_x + loss_y)


def loss_fn(
    model_output,
    target_image,
    w_reconstruction=1.0,
    w_orientation_correctness=1.0,
    w_phase_grad_x=1.0,
    w_phase_grad_y=1.0,
    w_orientation_smoothness=0.1,
    w_phase_smoothness=0.01,
    w_frequency_smoothness=0.1,
    w_phase_alignment=0.1,
):
    """
    Compute total loss with components.

    Args:
        model_output: Dictionary of model outputs
        target_image: Ground truth image tensor
        w_reconstruction: Weight for image reconstruction loss
        w_phase_grad_x: Weight for phase gradient x consistency
        w_phase_grad_y: Weight for phase gradient y consistency
        w_orientation_smoothness: Weight for orientation field smoothness
        w_phase_smoothness: Weight for phase field smoothness
        w_frequency_smoothness: Weight for frequency field smoothness
    """
    # Image reconstruction loss
    I_pred = model_output["I_pred"]
    recon_loss = F.mse_loss(I_pred, target_image)

    # Gradient comparison (phase gradient should match computed gradients)
    predicted_phase_grad_x = model_output["phase_gradient_x"]
    actual_phase_grad_x = torch.gradient(model_output["phase"], dim=1)[0]
    grad_loss_x = F.mse_loss(predicted_phase_grad_x, actual_phase_grad_x)

    predicted_phase_grad_y = model_output["phase_gradient_y"]
    actual_phase_grad_y = torch.gradient(model_output["phase"], dim=0)[0]
    grad_loss_y = F.mse_loss(predicted_phase_grad_y, actual_phase_grad_y)

    # Smoothness in orientation
    theta = model_output["theta"]
    laplacian_theta = laplacian_2d(theta)
    theta_smoothness_loss = torch.mean(laplacian_theta**2)

    # Smoothness in phase
    phase = model_output["phase"]
    laplacian_phase = laplacian_2d(phase)
    phase_smoothness_loss = torch.mean(laplacian_phase**2)

    # Smoothness in frequency
    freq = model_output["freq"]
    laplacian_freq = laplacian_2d(freq)
    freq_smoothness_loss = torch.mean(laplacian_freq**2)

    phase_alignment_loss = alignment_loss(phase)

    # Orientation correctness loss
    with torch.no_grad():
        target_orientation = (
            get_blockwise_orientation(
                target_image.unsqueeze(0).unsqueeze(0), block_size=8
            )
            .squeeze(0)
            .squeeze(0)
        )

    theta = model_output["theta"]
    cos2 = torch.cos(2.0 * theta)
    sin2 = torch.sin(2.0 * theta)

    cos2_pool = (
        F.avg_pool2d(cos2.unsqueeze(0).unsqueeze(0), kernel_size=8, stride=8)
        .squeeze(0)
        .squeeze(0)
    )
    sin2_pool = (
        F.avg_pool2d(sin2.unsqueeze(0).unsqueeze(0), kernel_size=8, stride=8)
        .squeeze(0)
        .squeeze(0)
    )

    orientation_windows = 0.5 * torch.atan2(sin2_pool, cos2_pool)
    orientation_windows = torch.remainder(orientation_windows, torch.pi)

    pred_cos2 = torch.cos(2.0 * orientation_windows)
    pred_sin2 = torch.sin(2.0 * orientation_windows)

    target_cos2 = torch.cos(2.0 * target_orientation)
    target_sin2 = torch.sin(2.0 * target_orientation)

    # Calculate the MSE between the vectors
    loss_cos = F.mse_loss(pred_cos2, target_cos2)
    loss_sin = F.mse_loss(pred_sin2, target_sin2)

    orientation_correctness_loss = loss_cos + loss_sin

    # Total loss with tunable weights
    total_loss = (
        w_reconstruction * recon_loss
        + w_phase_grad_x * grad_loss_x
        + w_phase_grad_y * grad_loss_y
        + w_orientation_smoothness * theta_smoothness_loss
        + w_phase_smoothness * phase_smoothness_loss
        + w_frequency_smoothness * freq_smoothness_loss
        + w_orientation_correctness * orientation_correctness_loss
        + w_phase_alignment * phase_alignment_loss
    )

    loss_dict = {
        "total": total_loss.item(),
        "recon": recon_loss.item(),
        "grad_x": grad_loss_x.item(),
        "grad_y": grad_loss_y.item(),
        "theta_smooth": theta_smoothness_loss.item(),
        "phase_smooth": phase_smoothness_loss.item(),
        "freq_smooth": freq_smoothness_loss.item(),
    }

    return total_loss, loss_dict


def train(
    model,
    target_image,
    steps=3000,
    lr=1e-3,
    device="cpu",
    w_reconstruction=1.0,
    w_phase_grad_x=1.0,
    w_phase_grad_y=1.0,
    w_orientation_smoothness=0.1,
    w_phase_smoothness=0.01,
    w_frequency_smoothness=0.1,
    w_orientation_correctness=1.0,
    w_phase_alignment=1.0,
):
    """
    Training loop.

    Args:
        model: DecomposedPhase model
        target_image: Ground truth image
        steps: Number of training iterations
        lr: Learning rate
        device: 'cpu' or 'cuda'
        w_reconstruction: Weight for image reconstruction loss
        w_phase_grad_x: Weight for phase gradient x consistency
        w_phase_grad_y: Weight for phase gradient y consistency
        w_orientation_smoothness: Weight for orientation field smoothness
        w_phase_smoothness: Weight for phase field smoothness
        w_frequency_smoothness: Weight for frequency field smoothness
    """
    model = model.to(device)
    target_image = target_image.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {steps} steps...")

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        output = model()
        loss, loss_dict = loss_fn(
            output,
            target_image,
            w_reconstruction=w_reconstruction,
            w_phase_grad_x=w_phase_grad_x,
            w_phase_grad_y=w_phase_grad_y,
            w_orientation_smoothness=w_orientation_smoothness,
            w_phase_smoothness=w_phase_smoothness,
            w_frequency_smoothness=w_frequency_smoothness,
            w_orientation_correctness=w_orientation_correctness,
            w_phase_alignment=w_phase_alignment,
        )

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1 or step == steps:
            print(
                f"[{step:5d}/{steps}] "
                f"Loss={loss_dict['total']:.5f} | "
                f"recon={loss_dict['recon']:.4f} "
                f"grad_x={loss_dict['grad_x']:.4f} "
                f"grad_y={loss_dict['grad_y']:.4f} "
                f"θ_s={loss_dict['theta_smooth']:.4f} "
                f"φ_s={loss_dict['phase_smooth']:.4f} "
                f"f_s={loss_dict['freq_smooth']:.4f} "
                f"orientation_correctness={loss_dict.get('orientation_correctness', 0):.4f}"
            )

    return model


def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 256, 256
    STEPS = 300
    LR = 1e-2
    INIT_FREQ = 0.1
    QUIVER_STRIDE = 16

    W_RECONSTRUCTION = 6.0
    W_PHASE_GRAD_X = 0.0
    W_PHASE_GRAD_Y = 0.0
    W_ORIENTATION_SMOOTHNESS = 0.0
    W_PHASE_SMOOTHNESS = 2.0
    W_FREQUENCY_SMOOTHNESS = 0.0
    W_ORIENTATION_CORRECTNESS = 0.0
    W_PHASE_ALIGNMENT = 0.0

    print(f"Using device: {DEVICE}")

    # Generate synthetic fingerprint (replace with real image loading)
    # yy, xx = np.mgrid[:H, :W]
    # cx, cy = W // 2, H // 2
    # X = xx - cx
    # Y = yy - cy

    # kappa = 0.16
    # psi_c = kappa * np.sqrt(X * X + Y * Y)
    # img = 0.5 * (1.0 - np.cos(psi_c))

    img = cv2.imread(os.path.join("images", "spiral_phase.jpg"), cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    H, W = img.shape

    # Convert to tensor
    target_image = torch.from_numpy(img).float()

    # Create model
    model = DecomposedPhase(
        H=H,
        W=W,
        init_freq=INIT_FREQ,
        # smooth=True,
        # init_phase=get_init_phase(H, W),
        spiral_phase_coords=[(95, 128), (40, 55), (128, 200)],
        spiral_phase_polarities=[+1, -1, -1],
    )

    # Train
    model = train(
        model,
        target_image,
        steps=STEPS,
        lr=LR,
        device=DEVICE,
        w_reconstruction=W_RECONSTRUCTION,
        w_phase_grad_x=W_PHASE_GRAD_X,
        w_phase_grad_y=W_PHASE_GRAD_Y,
        w_orientation_smoothness=W_ORIENTATION_SMOOTHNESS,
        w_phase_smoothness=W_PHASE_SMOOTHNESS,
        w_frequency_smoothness=W_FREQUENCY_SMOOTHNESS,
        w_orientation_correctness=W_ORIENTATION_CORRECTNESS,
        w_phase_alignment=W_PHASE_ALIGNMENT,
    )

    # Plot results
    plot_results(model, target_image, quiver_stride=QUIVER_STRIDE)
    plot_first_order_grads(model)
    plot_second_order_grads(model)
    plot_change_in_value_per_stripe(model)

    plt.figure(figsize=(12, 6))
    image_without_spirals = 0.5 * (1.0 - torch.cos(model.phase.detach()))

    plt.subplot(1, 2, 1)
    plt.title("Reconstructed Image without Spiral Phase")
    plt.imshow(image_without_spirals.cpu(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Spiral Phase Component")
    plt.imshow(model.spiral_phase.cpu(), cmap="gray")
    plt.axis("off")
    plt.show()


def get_init_phase(H, W):
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    X = xx - cx
    Y = yy - cy

    # radial phase
    kappa = 0.35  # controls spacing of rings (ridges)
    psi_c = kappa * np.sqrt(X * X + Y * Y)
    return psi_c


if __name__ == "__main__":
    torch.manual_seed(1337)
    np.random.seed(1337)
    main()
