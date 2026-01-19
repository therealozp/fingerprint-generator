import torch
import torch.nn as nn
import torch.nn.functional as F


class InputContinuityTransform(nn.Module):
    """
    Pre-processing layer to convert discontinuous wrapped phase maps
    into continuous sine/cosine vector components.
    """

    def __init__(self, use_double_angle: bool = False):
        super().__init__()
        self.use_double_angle = use_double_angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw wrapped orientation map (B, 1, H, W).
        Returns:
            Tensor containing [cos, sin] components (B, 2, H, W).
        """
        if self.use_double_angle:
            x = x * 2

        cos_x = torch.cos(x)
        sin_x = torch.sin(x)

        return torch.cat([cos_x, sin_x], dim=1)


class DoubleConv(nn.Module):
    """
    Standard double convolution block used in U-Net:
    (Conv2d -> BN -> ReLU) * 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DilatedBottleneck(nn.Module):
    """
    Bottleneck layer using dilated convolutions to expand the receptive field
    and capture global context for resolving large-scale wrap ambiguities.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OrientationUnwrappingNet(nn.Module):
    """
    Residual U-Net for Orientation Unwrapping.
    Predicts the wrap count 'k' to recover the absolute phase:
    Phi_true = Phi_wrapped + 2*pi*k
    """

    def __init__(
        self,
        input_channels: int = 1,
        auxiliary_channels: int = 0,
        use_double_angle: bool = False,
    ):
        super().__init__()

        self.transform = InputContinuityTransform(use_double_angle=use_double_angle)

        unet_input_dim = 2 + auxiliary_channels

        self.inc = DoubleConv(unet_input_dim, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DilatedBottleneck(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.out_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self, wrapped_phase: torch.Tensor, aux_map: torch.Tensor = None
    ) -> dict:
        """
        Args:
            wrapped_phase: Tensor (B, 1, H, W) in range [-pi, pi]
            aux_map: Optional Quality/Coherence map (B, C, H, W)
        Returns:
            Dictionary containing:
                'k_est': The predicted wrap count (continuous)
                'unwrapped': The final reconstructed orientation
        """

        tr_input = self.transform(wrapped_phase)

        if aux_map is not None:
            x = torch.cat([tr_input, aux_map], dim=1)
        else:
            x = tr_input

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bot = self.bottleneck(x4)

        x = self.up1(x_bot)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        k_continuous = self.out_head(x)

        unwrapped = wrapped_phase + (2 * torch.pi * k_continuous)

        return {"k_est": k_continuous, "unwrapped": unwrapped}


class UnwrappingCompositeLoss(nn.Module):
    """
    Composite loss function for phase unwrapping.
    Includes:
    1. Periodic Loss: Enforces re-wrapped consistency.
    2. Gradient Consistency Loss: Enforces smoothness of derivatives.
    """

    def __init__(self, lambda_per: float = 1.0, lambda_grad: float = 0.5):
        super().__init__()
        self.lambda_per = lambda_per
        self.lambda_grad = lambda_grad

    def wrap_operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wraps a value into the range [-pi, pi].
        """
        return torch.angle(torch.exp(1j * x))

    def spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes simple spatial gradients (dy, dx).
        Returns Tensor shape (B, 2, H, W) where channel 0 is dy, 1 is dx.
        """
        b, c, h, w = x.shape

        dy = torch.zeros_like(x)
        dx = torch.zeros_like(x)

        dy[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]

        return torch.cat([dy, dx], dim=1)

    def forward(
        self,
        pred_unwrapped: torch.Tensor,
        gt_unwrapped: torch.Tensor,
        input_wrapped: torch.Tensor,
    ):
        """
        Args:
            pred_unwrapped: The network output (Phi_pred).
            gt_unwrapped: Ground truth unwrapped phase (Phi_true).
            input_wrapped: The noisy wrapped input (phi_input).
        """

        diff = pred_unwrapped - gt_unwrapped
        loss_per = torch.mean(1 - torch.cos(diff))

        pred_grad = self.spatial_gradient(pred_unwrapped)

        # Calculate the "continuous" gradient of the wrapped input
        # by wrapping the raw gradient to remove 2pi jumps.
        input_raw_grad = self.spatial_gradient(input_wrapped)
        target_grad = self.wrap_operator(input_raw_grad)

        loss_grad = F.mse_loss(pred_grad, target_grad)

        total_loss = (self.lambda_per * loss_per) + (self.lambda_grad * loss_grad)

        return total_loss, {"periodic": loss_per.item(), "gradient": loss_grad.item()}
