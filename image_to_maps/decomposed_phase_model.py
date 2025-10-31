import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2


class DecomposedPhase(nn.Module):
    def __init__(self, H=256, W=256, init_freq=0.1):
        super(DecomposedPhase, self).__init__()
        self.phase = nn.Parameter(torch.randn(H, W))
        self.theta_cos = nn.Parameter(torch.randn(H, W) * 0.1)
        self.freq = nn.Parameter(torch.full((H, W), init_freq))

    def forward(self, x=None):
        # Normalize theta_cos to [-1, 1]
        theta_cos = torch.tanh(self.theta_cos)
        theta_sin = torch.sqrt(1 - theta_cos**2 + 1e-6)

        phase_gradient_x = 2.0 * torch.pi * torch.abs(self.freq) * theta_cos
        phase_gradient_y = 2.0 * torch.pi * torch.abs(self.freq) * theta_sin

        I_pred = 0.5 * (1.0 + torch.cos(self.phase))

        # Reconstruct theta for visualization
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


def loss_fn(
    model_output,
    target_image,
    w_reconstruction=1.0,
    w_phase_grad_x=1.0,
    w_phase_grad_y=1.0,
    w_orientation_smoothness=0.1,
    w_phase_smoothness=0.01,
    w_frequency_smoothness=0.1,
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

    # Total loss with tunable weights
    total_loss = (
        w_reconstruction * recon_loss
        + w_phase_grad_x * grad_loss_x
        + w_phase_grad_y * grad_loss_y
        + w_orientation_smoothness * theta_smoothness_loss
        + w_phase_smoothness * phase_smoothness_loss
        + w_frequency_smoothness * freq_smoothness_loss
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
                f"f_s={loss_dict['freq_smooth']:.4f}"
            )

    return model


def plot_results(model, target_image, quiver_stride=16):
    """Plot results similar to dgd.py."""
    with torch.no_grad():
        output = model()

        # Move to CPU and convert to numpy
        phase = output["phase"].cpu().numpy()
        I_pred = output["I_pred"].cpu().numpy()
        theta = output["theta"].cpu().numpy()
        freq = output["freq"].cpu().numpy()
        target = target_image.cpu().numpy()

        H, W = phase.shape

        # Wrap phase to [0, 1] for visualization
        phase_vis = ((phase + np.pi) % (2 * np.pi)) / (2 * np.pi)

        # Prepare quiver plot
        s = quiver_stride
        Y, X = np.mgrid[0:H:s, 0:W:s]

        # Ridge tangent direction
        U = np.cos(theta)[::s, ::s]
        V = np.sin(theta)[::s, ::s]

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Original image
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(target, cmap="gray")
        plt.axis("off")

        # Predicted phase (wrapped)
        plt.subplot(2, 3, 2)
        plt.title("Predicted Phase (wrapped to [0,1])")
        plt.imshow(phase_vis, cmap="gray")
        plt.axis("off")

        # Predicted image
        plt.subplot(2, 3, 3)
        plt.title("Predicted Image from Phase")
        plt.imshow(I_pred, cmap="gray")
        plt.axis("off")

        # Orientation field (quiver)
        plt.subplot(2, 3, 4)
        plt.title(f"Orientation Field (stride={s})")
        plt.imshow(target, cmap="gray")
        plt.quiver(
            X,
            Y,
            U,
            V,
            pivot="mid",
            scale=40,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            width=0.002,
            color="red",
        )
        plt.axis("off")

        # Frequency map
        plt.subplot(2, 3, 5)
        plt.title("Ridge Frequency Map")
        plt.imshow(freq, cmap="hot")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        # Reconstruction error
        plt.subplot(2, 3, 6)
        plt.title("Reconstruction Error (absolute)")
        error = np.abs(I_pred - target)
        plt.imshow(error, cmap="hot")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 256, 256
    STEPS = 3000
    LR = 1e-2
    INIT_FREQ = 0.1
    QUIVER_STRIDE = 8

    W_RECONSTRUCTION = 6.0
    W_PHASE_GRAD_X = 10.0
    W_PHASE_GRAD_Y = 10.0
    W_ORIENTATION_SMOOTHNESS = 2.9
    W_PHASE_SMOOTHNESS = 3.3
    W_FREQUENCY_SMOOTHNESS = 0.1

    print(f"Using device: {DEVICE}")

    # Generate synthetic fingerprint (replace with real image loading)
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    X = xx - cx
    Y = yy - cy

    kappa = 0.16
    psi_c = kappa * np.sqrt(X * X + Y * Y)
    img = 0.5 * (1.0 - np.cos(psi_c))

    # Or load real image:
    # img = cv2.imread("fingerprint.png", cv2.IMREAD_GRAYSCALE)
    # img = img / 255.0

    # Convert to tensor
    target_image = torch.from_numpy(img).float()

    # Create model
    model = DecomposedPhase(H=H, W=W, init_freq=INIT_FREQ)

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
    )

    # Plot results
    plot_results(model, target_image, quiver_stride=QUIVER_STRIDE)


if __name__ == "__main__":
    torch.manual_seed(1337)
    np.random.seed(1337)
    main()
