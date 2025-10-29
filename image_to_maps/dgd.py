# fingerprint_phase_simple.py
import math, numpy as np, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# user: set this to your fingerprint image (grayscale)
IMAGE_PATH = "your_fingerprint.png"  # <-- put your path here
STEPS = 2000
LR = 1e-2

# loss weights (only the 3 you asked for)
W_PIX = 6.0  # image reconstruction from phase
W_GRAD = 10.0  # ∇φ vs 2π f n_hat
W_O_TV = 0.5  # smoothness of (cos2θ, sin2θ)
W_F_TV = 0.5  # smoothness of frequency

QUIVER_STRIDE = 8  # subsampling for the orientation quiver
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================


# -----------------------------
# small helpers (wrap-aware diffs)
# -----------------------------
def wrap_diff(phi: torch.Tensor, dim: int) -> torch.Tensor:
    """Forward difference for phase with 2π wrapping."""
    rolled = torch.roll(phi, shifts=-1, dims=dim)
    d = torch.atan2(torch.sin(rolled - phi), torch.cos(rolled - phi))
    return d


def grad_phi_wrap(phi: torch.Tensor) -> torch.Tensor:
    """∇φ (wrap-aware). Returns [N,2,H,W] = (gx, gy)."""
    gx = wrap_diff(phi, dim=3)  # x = width
    gy = wrap_diff(phi, dim=2)  # y = height
    return torch.cat([gx, gy], dim=1)


def tv_l2(x: torch.Tensor) -> torch.Tensor:
    """Smooth TV (L2)."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


# -----------------------------
# model: phase φ, orientation via (cos2θ,sin2θ), frequency via softplus
# -----------------------------
class DecomposedPhaseOptim(nn.Module):
    def __init__(self, H, W, init="randn", init_freq=0.12):
        super().__init__()
        if init == "zeros":
            init_phase = torch.zeros(1, 1, H, W)
        else:
            init_phase = (2 * math.pi) * torch.rand(1, 1, H, W) - math.pi

        self.phi = nn.Parameter(init_phase)
        self.theta_sin = nn.Parameter(torch.randn(1, 1, H, W) * 0.05)
        self.theta_cos = nn.Parameter(torch.randn(1, 1, H, W) * 0.05)

        self.fraw = nn.Parameter(torch.full((1, 1, H, W), float(init_freq)))  # freq raw

    def forward(self):
        # orientation: normalize to unit circle -> (cos2θ, sin2θ)
        norm = torch.clamp(
            torch.sqrt(
                (self.theta_cos**2 + self.theta_sin**2).sum(dim=1, keepdim=True)
            ),
            min=1e-6,
        )
        c2 = self.theta_cos / norm
        s2 = self.theta_sin / norm

        theta = 0.5 * torch.atan2(s2, c2)  # θ in (-π/2, π/2]

        # unit normal to ridges: n_hat = (-sinθ, cosθ)
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        n_hat = torch.cat([-sin_th, cos_th], dim=1)  # [1,2,H,W]

        # frequency >= 0
        f = F.softplus(self.fraw) + 1e-6

        # predicted image from phase
        I_pred = 0.5 * (1.0 + torch.cos(self.phi))

        return {
            "phi": self.phi,
            "I_pred": I_pred,
            "theta": theta,
            "c2": c2,
            "s2": s2,
            "f": f,
            "n_hat": n_hat,
        }


# -----------------------------
# losses (exactly your 3 ideas)
# -----------------------------
def compute_losses_simple(out, I_tgt):
    phi = out["phi"]  # [1,1,H,W]
    I_pred = out["I_pred"]
    c2 = out["c2"]
    s2 = out["s2"]
    f = out["f"]
    n_hat = out["n_hat"]  # [1,2,H,W]

    # (1) pixel reconstruction (L1 to match ridges from phase)
    L_pix = (I_pred - I_tgt).abs().mean()

    # (2) phase-gradient consistency
    G_phi = grad_phi_wrap(phi)  # [1,2,H,W]
    G_tar = torch.cat(
        [(2 * math.pi) * f * n_hat[:, 0:1], (2 * math.pi) * f * n_hat[:, 1:2]], dim=1
    )
    L_grad = (G_phi - G_tar).pow(2).mean()

    # (3) smoothness on orientation (via cos2θ, sin2θ) and frequency
    O2 = torch.cat([c2, s2], dim=1)  # [1,2,H,W]
    L_o_tv = tv_l2(O2)
    L_f_tv = tv_l2(f)

    # total
    L = W_PIX * L_pix + W_GRAD * L_grad + W_O_TV * L_o_tv + W_F_TV * L_f_tv
    parts = dict(
        L_pix=float(L_pix.item()),
        L_grad=float(L_grad.item()),
        L_o_tv=float(L_o_tv.item()),
        L_f_tv=float(L_f_tv.item()),
    )
    return L, parts


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
    model = DecomposedPhaseOptim(H, W, init="randn", init_freq=0.12).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(1, STEPS + 1):
        opt.zero_grad()
        out = model()
        L, parts = compute_losses_simple(out, I_tgt)
        L.backward()
        opt.step()

        # optional: keep orientation params roughly normalized (helps stability)
        with torch.no_grad():
            nrm = torch.clamp(
                torch.sqrt((model.o2raw**2).sum(dim=1, keepdim=True)), min=1e-6
            )
            model.o2raw[:] = model.o2raw / nrm

        if step % 100 == 0 or step == 1 or step == STEPS:
            print(
                f"[{step:5d}/{STEPS}] "
                f"L={L.item():.5f} | pix={parts['L_pix']:.4f} grad={parts['L_grad']:.4f} "
                f"O_TV={parts['L_o_tv']:.4f} F_TV={parts['L_f_tv']:.4f}"
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
    phi_vis = ((phi + math.pi) % (2 * math.pi)) / (2 * math.pi)

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
