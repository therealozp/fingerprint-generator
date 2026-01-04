import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def get_blockwise_orientation(
    source_image: torch.Tensor,
    block_size: int = 8,
):
    dev = source_image.device
    dtype = source_image.dtype

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=dev
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=dev
    ).view(1, 1, 3, 3)
    blur = transforms.GaussianBlur(kernel_size=5)

    img = source_image.clone()
    img = blur(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    Gx = F.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
    Gy = F.conv2d(img.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)

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

    Vx = 2.0 * sum_Gxy.squeeze(0).squeeze(0)
    Vy = sum_Gx2.squeeze(0).squeeze(0) - sum_Gy2.squeeze(0).squeeze(0)

    theta = 0.5 * torch.atan2(Vx, Vy + 1e-8)

    Tx = torch.cos(2.0 * theta)
    Ty = torch.sin(2.0 * theta)

    Tprime_x = blur(Tx.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    Tprime_y = blur(Ty.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    theta = 0.5 * torch.atan2(Tprime_y, Tprime_x)
    orientation_map = torch.remainder(theta, torch.pi)

    return orientation_map
