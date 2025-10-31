# fingerprint_phase_simple.py
import math, numpy as np, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# User-editable knobs
IMAGE_PATH = "your_fingerprint.png"  # grayscale fingerprint
STEPS = 3000
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights (interpretation below)
W_PIX = 6.0  # match I_pred(phi) to target image
W_GRAD = 10.0  # match ∇phi to 2π f n_hat
W_THETA_S = 0.5  # smooth theta locally (not globally flat, just no jitter)
W_FREQ_S = 0.5  # smooth frequency locally
W_LAPLACE = 0.1  # tiny stabilizer on phi

QUIVER_STRIDE = 8  # for visualization of orientation
INIT_FREQ = 0.12  # initial guess for ridge frequency (cycles/pixel)
INIT_PHASE = "randn"  # "randn" or "zeros"
SEED = 1337
# ============================================================


# -----------------------------
# wrap-aware finite differences for phase
# -----------------------------
def wrap_diff(phi: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Forward difference on a wrapped phase field.
    phi: [N,1,H,W]
    Returns dphi/d(dim) with shortest angular diff in (-pi, pi].
    """
    rolled = torch.roll(phi, shifts=-1, dims=dim)
    d = torch.atan2(torch.sin(rolled - phi), torch.cos(rolled - phi))
    return d


def grad_phi_wrap(phi: torch.Tensor) -> torch.Tensor:
    """
    Compute ∇phi = (phi_x, phi_y) with wrap-aware differences.
    Returns [N,2,H,W] = (gx, gy).
    """
    gx = wrap_diff(phi, dim=3)  # x-direction (W)
    gy = wrap_diff(phi, dim=2)  # y-direction (H)
    return torch.cat([gx, gy], dim=1)


# -----------------------------
# simple first-derivative smoothness (Dirichlet energy)
# This says: "be locally smooth", not "be constant everywhere".
# -----------------------------
def smoothness_dirichlet(field: torch.Tensor) -> torch.Tensor:
    """
    Penalize squared spatial derivatives of a scalar field.
    field: [N,1,H,W]
    Returns sum of squared forward diffs.
    """
    dx = field[:, :, :, 1:] - field[:, :, :, :-1]
    dy = field[:, :, 1:, :] - field[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


# -----------------------------
# Laplacian regularizer for phi (tiny stabilizer)
# -----------------------------
def laplacian_4nbr(u: torch.Tensor) -> torch.Tensor:
    """
    4-neighbor Laplacian with replicate padding.
    u: [N,1,H,W]
    Return same shape.
    """
    up = F.pad(u[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
    down = F.pad(u[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
    left = F.pad(u[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
    right = F.pad(u[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
    return (up + down + left + right) - 4.0 * u


# -----------------------------
# Model
# -----------------------------
class DecomposedPhaseOptim(nn.Module):
    """
    Learn:
      - phi(x,y): phase field (wrapped)
      - orientation field theta(x,y), via normalized (cos2θ, sin2θ)
      - frequency field f(x,y) >= 0

    This is consistent with AM-FM modeling:
      I(x,y) ≈ 0.5*(1+cos(phi(x,y)))
      ∇phi(x,y) ≈ 2π f(x,y) n_hat(x,y),
    where n_hat = (-sin θ, cos θ) is perpendicular to ridge direction.
    """

    def __init__(self, H, W, init_phase="randn", init_freq=0.12):
        super().__init__()

        # initialize phi
        if init_phase == "zeros":
            init_phi = torch.zeros(1, 1, H, W)
        else:
            # random in [-pi, pi]
            init_phi = (2 * torch.pi) * torch.rand(1, 1, H, W) - torch.pi

        # raw orientation parameters (unconstrained)
        # we'll interpret them as [cos(2θ), sin(2θ)] after normalization
        init_o = torch.randn(1, 2, H, W) * 0.05

        # raw frequency param -> softplus -> f>=0
        init_f = torch.full((1, 1, H, W), float(init_freq))

        self.phi = nn.Parameter(init_phi)
        self.o_raw = nn.Parameter(init_o)  # [1,2,H,W]
        self.f_raw = nn.Parameter(init_f)  # [1,1,H,W]

    def forward(self):
        # normalize orientation 2-vector per pixel
        # u_raw = (ux, uy) ~ supposed to represent (cos(2θ), sin(2θ))
        norm = torch.clamp(
            torch.sqrt((self.o_raw**2).sum(dim=1, keepdim=True)), min=1e-6
        )
        u = self.o_raw / norm  # [1,2,H,W], now unit length

        cos2t = u[:, 0:1]  # cos(2θ)
        sin2t = u[:, 1:2]  # sin(2θ)
        # recover θ in (-π/2, π/2]
        theta = 0.5 * torch.atan2(sin2t, cos2t)  # [1,1,H,W]

        # ridge normal n_hat = (-sin θ, cos θ)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        n_hat_x = -sin_t
        n_hat_y = cos_t
        n_hat = torch.cat([n_hat_x, n_hat_y], dim=1)  # [1,2,H,W]

        # frequency >= 0
        f = F.softplus(self.f_raw) + 1e-6  # [1,1,H,W]

        # predicted intensity from phi
        I_pred = 0.5 * (1.0 + torch.cos(self.phi))  # [1,1,H,W]

        return {
            "phi": self.phi,  # [1,1,H,W]
            "I_pred": I_pred,  # [1,1,H,W]
            "theta": theta,  # [1,1,H,W] ridge tangent orientation
            "n_hat": n_hat,  # [1,2,H,W] ridge normal unit vector
            "f": f,  # [1,1,H,W] ridge frequency (cycles/pixel)
            "cos2t": cos2t,
            "sin2t": sin2t,
        }


# -----------------------------
# Losses (aligned with the math)
# -----------------------------
def compute_losses(model_out, I_tgt):
    """
    model_out is dict from forward()

    We enforce:
    1. pixel reconstruction  (phi -> cos(phi) matches target image)
    2. phase gradient consistency (∇phi matches 2π f n_hat)
    3. smoothness of theta   (local orientation flow should vary slowly)
    4. smoothness of f       (local ridge frequency smooth)
    5. small Laplacian on phi (regularity)
    """

    phi = model_out["phi"]  # [1,1,H,W]
    I_pred = model_out["I_pred"]  # [1,1,H,W]
    theta = model_out["theta"]  # [1,1,H,W]
    n_hat = model_out["n_hat"]  # [1,2,H,W]
    f = model_out["f"]  # [1,1,H,W]

    # (1) Pixel reconstruction loss
    # this is hypothesis (1): "if phi alone can render the fingerprint, phi is good
    I_tgt = I_tgt.to(dtype=I_pred.dtype, device=I_pred.device)
    L_pix = F.mse_loss(
        I_pred,
        I_tgt,
    )

    # (2) Phase-gradient consistency with AM-FM model
    # ∇phi_est from the learned phi, using wrap-aware diff
    G_phi = grad_phi_wrap(phi)  # [1,2,H,W] = (phi_x, phi_y)
    # target instantaneous frequency vector: 2π f n_hat
    # expand f to 2 channels to match n_hat
    f2 = torch.cat([f, f], dim=1)  # [1,2,H,W]
    G_tar = (2 * torch.pi) * (f2 * n_hat)

    # this enforces hypothesis (2): "phi's gradient should match what θ and f predict"
    L_grad = F.mse_loss(G_phi, G_tar)

    # (3) Smoothness of theta
    # We don't want theta CONSTANT, we just don't want pixel-to-pixel jitter.
    # This is the Dirichlet energy ∫ |∇θ|^2, which penalizes high-frequency noise
    # but allows slow bending over many pixels.
    L_theta_smooth = smoothness_dirichlet(theta)

    # (4) Smoothness of frequency f
    # same logic: frequency should be spatially smooth so ridge spacing isn't chaotic
    L_f_smooth = smoothness_dirichlet(f)

    # (5) Laplacian on phi: keeps phi from getting salt-and-pepper (stabilizer only)
    lap_phi = laplacian_4nbr(phi)
    L_lap = (lap_phi.pow(2)).mean()

    # total loss
    L_total = (
        W_PIX * L_pix
        + W_GRAD * L_grad
        + W_THETA_S * L_theta_smooth
        + W_FREQ_S * L_f_smooth
        + W_LAPLACE * L_lap
    )

    parts = {
        "L_total": float(L_total.item()),
        "L_pix": float(L_pix.item()),
        "L_grad": float(L_grad.item()),
        "L_theta_s": float(L_theta_smooth.item()),
        "L_f_smooth": float(L_f_smooth.item()),
        "L_lap": float(L_lap.item()),
    }

    return L_total, parts


# -----------------------------
# training
# -----------------------------
def main():
    # load grayscale image -> [0,1]
    H, W = 256, 256
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    X = xx - cx
    Y = yy - cy

    kappa = 0.16  # controls spacing of rings (ridges)
    psi_c = kappa * np.sqrt(X * X + Y * Y)
    img = 0.5 * (1.0 - np.cos(psi_c))

    # img = cv2.imread(
    #     "C:\\Users\\Oz\\code\\fingerprint-generator\\images\\50_whorl.jpg",
    #     cv2.IMREAD_GRAYSCALE,
    # )
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # rescaled_dims = img.shape[0] * 0.4
    # img = cv2.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))
    # img = img / 255.0

    I_tgt = torch.from_numpy(img)[None, None].to(DEVICE)
    model = DecomposedPhaseOptim(H, W).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(1, STEPS + 1):
        opt.zero_grad()
        out = model()
        L, parts = compute_losses(out, I_tgt)
        L.backward()
        opt.step()

        if step % 100 == 0 or step == 1 or step == STEPS:
            print(
                f"[{step:5d}/{STEPS}] "
                f"L={parts['L_total']:.5f} | "
                f"pix={parts['L_pix']:.4f} grad={parts['L_grad']:.4f} "
                f"θ_s={parts['L_theta_s']:.4f} f_s={parts['L_f_smooth']:.4f} lap={parts['L_lap']:.4f}"
            )

    # -----------------------------
    # plots requested
    # -----------------------------
    with torch.no_grad():
        out = model()
        phi = out["phi"].cpu().numpy()[0, 0]
        I_pred = out["I_pred"].cpu().numpy()[0, 0]
        theta = out["theta"].cpu().numpy()[0, 0]  # orientation angle

    # (2) visualize phase mod 2π for readability
    phi_vis = ((phi + torch.pi) % (2 * torch.pi)) / (2 * torch.pi)

    # (4) quiver of orientation (we'll plot ridge tangent or normal—choose one)
    # ridge tangent vector t_hat = (cosθ, sinθ)
    t_x = np.cos(theta)
    t_y = np.sin(theta)

    # stride for legibility
    s = QUIVER_STRIDE
    Y, X = np.mgrid[0:H:s, 0:W:s]
    U = t_x[::s, ::s]
    V = t_y[::s, ::s]

    # plot
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Predicted phase (wrapped to [0,1])")
    plt.imshow(phi_vis, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Predicted image from phase")
    plt.imshow(I_pred, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title(f"Orientation field (quiver, stride={s})")
    plt.imshow(img, cmap="gray")
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1337)
    np.random.seed(1337)
    print(f"Using device: {DEVICE}")
    main()
