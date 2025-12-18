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
        fig = plt.figure(figsize=(12, 5))

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


def plot_first_order_grads(model):
    # figure to compare the phase gradients
    fig = plt.figure(figsize=(12, 5))
    with torch.no_grad():
        output = model()
        # normalize before plot
        phase_gradient_x = output["phase_gradient_x"].cpu().numpy()
        phase_gradient_y = output["phase_gradient_y"].cpu().numpy()

        actual_phase_grad_x = np.gradient(model.phase.cpu().numpy(), axis=1)
        actual_phase_grad_y = np.gradient(model.phase.cpu().numpy(), axis=0)

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


def plot_second_order_grads(model):
    # figure to compare the phase gradients
    fig = plt.figure(figsize=(12, 5))
    with torch.no_grad():
        output = model()
        # normalize before plot
        phase_gradient_x = output["phase_gradient_x"].cpu().numpy()
        phase_gradient_y = output["phase_gradient_y"].cpu().numpy()

        actual_phase_grad_x = np.gradient(model.phase.cpu().numpy(), axis=1)
        actual_phase_grad_y = np.gradient(model.phase.cpu().numpy(), axis=0)

        phase_gradient_xx = np.gradient(phase_gradient_x, axis=1)
        phase_gradient_yy = np.gradient(phase_gradient_y, axis=0)

        actual_phase_grad_xx = np.gradient(actual_phase_grad_x, axis=1)
        actual_phase_grad_yy = np.gradient(actual_phase_grad_y, axis=0)

        plt.subplot(2, 2, 1)
        plt.title("Phase Gradient XX from Orientation")
        plt.imshow(phase_gradient_xx, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.title("Phase Gradient XX from Phase")
        plt.imshow(actual_phase_grad_xx, cmap="gray")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.title("Phase Gradient YY from Orientation")
        plt.imshow(phase_gradient_yy, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 2, 4)

        plt.title("Phase Gradient YY from Phase")
        plt.imshow(actual_phase_grad_yy, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def plot_change_in_value_per_stripe(model):
    with torch.no_grad():
        output = model()
        phase = output["phase"].cpu().numpy()
        H, W = phase.shape

        stripe_width = 1
        num_stripes = W // stripe_width

        if num_stripes == 0:
            return

        # Compute mean pixel value for each horizontal row within each vertical stripe
        stripe_means = np.zeros((num_stripes, H))
        for i in range(num_stripes):
            x0 = i * stripe_width
            x1 = x0 + stripe_width
            stripe = phase[:, x0:x1]
            stripe_means[i] = stripe.mean(axis=1)

        # Plot the phase image with stripe boundaries and the per-row stripe means
        fig, (ax_img, ax_plot) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]}
        )

        ax_img.set_title("Phase with Stripe Boundaries")
        im = ax_img.imshow(phase, cmap="gray")
        for i in range(num_stripes + 1):
            ax_img.axvline(i * stripe_width, color="red", linewidth=0.6)
        ax_img.axis("off")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        ax_plot.set_title("Mean Pixel Value per Row for Each Stripe")
        ax_plot.set_xlabel("Row (y)")
        ax_plot.set_ylabel("Mean pixel value (across stripe width)")
        x = np.arange(H)

        colors = plt.cm.get_cmap("tab20", num_stripes)
        for i in range(num_stripes):
            ax_plot.plot(x, stripe_means[i], color=colors(i), label=f"stripe {i}")

        # Avoid overcrowding legend if many stripes
        if num_stripes <= 10:
            ax_plot.legend(loc="upper right", fontsize="small", ncol=1)
        ax_plot.grid(True)

        plt.tight_layout()
        plt.show()
