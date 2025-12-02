import torch
import numpy as np
import matplotlib.pyplot as plt


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
        # phase_vis = ((phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
        phase_vis = phase

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
        plt.gca().invert_yaxis()
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

        # figure to compare the phase gradients
        fig = plt.figure(figsize=(12, 5))
        # normalize before plot
        phase_gradient_x = output["phase_gradient_x"].cpu().numpy()
        phase_gradient_y = output["phase_gradient_y"].cpu().numpy()

        actual_phase_grad_x = np.gradient(phase, axis=1)
        actual_phase_grad_y = np.gradient(phase, axis=0)

        # normalize arrays to [0,1] for visualization
        def _normalize(arr):
            a = np.nan_to_num(arr)
            mn = a.min()
            mx = a.max()
            if mx - mn < 1e-8:
                return np.zeros_like(a)
            return (a - mn) / (mx - mn)

        phase_gradient_x = _normalize(phase_gradient_x)
        phase_gradient_y = _normalize(phase_gradient_y)
        actual_phase_grad_x = _normalize(actual_phase_grad_x)
        actual_phase_grad_y = _normalize(actual_phase_grad_y)

        plt.subplot(2, 2, 1)
        plt.title("Phase Gradient X from Orientation")
        plt.imshow(phase_gradient_x, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("Phase Gradient X from Phase")
        plt.imshow(actual_phase_grad_x, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Phase Gradient Y from Orientation")
        plt.imshow(phase_gradient_y, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.title("Phase Gradient Y from Phase")
        plt.imshow(actual_phase_grad_y, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
