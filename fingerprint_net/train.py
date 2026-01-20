import torch
import torch.nn as nn

from model import FingerprintUNet
from loss_functions import FingerprintLoss
from torch.utils.data import DataLoader
from dataset import FingerprintOrientationDataset

from dataclasses import dataclass
from tqdm import tqdm

import os


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 5e-4
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True


def train(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, optimizer, save_model=False
):
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lowest_validation_loss = float("inf")

    criterion = FingerprintLoss()

    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        running = 0.0
        loss_from_continuous_phase = 0.0
        loss_from_full_phase = 0.0
        sin_cos_normalization_loss = 0.0

        for step, input in enumerate(tqdm(loader)):
            inputs = input["inputs"].to(cfg.device)  # (B, 3, H, W)
            target_c = input["target_continuous"].to(cfg.device)  # (B, 1, H, W)
            target_f = input["target_full"].to(cfg.device)  # (B, 1, H, W)
            spiral_phasor = input["spiral_phasor"].to(cfg.device)  # (B, 2, H, W)

            optimizer.zero_grad()

            pred_c, pred_f, pred_phasor = model(inputs, spiral_phasor)

            # Calculate Loss
            loss, l_cont, l_full, l_norm = criterion(
                pred_c, pred_f, pred_phasor, target_c, target_f
            )

            loss.backward()
            optimizer.step()

            running += loss.item()
            loss_from_continuous_phase += l_cont.item()
            loss_from_full_phase += l_full.item()
            sin_cos_normalization_loss += l_norm.item()

        avg = running / max(1, len(loader))
        if save_model and avg < lowest_validation_loss:
            lowest_validation_loss = avg
            torch.save(
                model.state_dict(),
                f"fingerprint_unet_best.pth",
            )
        print(
            f"epoch {epoch+1:03d}/{cfg.epochs} | total loss = {avg:.6f} | cont phase loss = {loss_from_continuous_phase / max(1, len(loader)):.6f} | full phase loss = {loss_from_full_phase / max(1, len(loader)):.6f} | norm loss = {sin_cos_normalization_loss / max(1, len(loader)):.6f}"
        )


if __name__ == "__main__":
    cfg = TrainConfig(epochs=100)

    original_images_dir = "./single/"
    target_images_dir = "./single/continuous_phase"
    minutiae_dir = "./single/minutiae"
    orientation_maps_dir = "./single/orientations"

    orig_paths = []
    cont_paths = []
    minutiae_paths = []
    orientation_map_paths = []

    for item in os.listdir(original_images_dir):
        if item.endswith(".png"):
            orig_paths.append(os.path.join(original_images_dir, item))
            cont_paths.append(os.path.join(target_images_dir, item))
            minutiae_paths.append(
                os.path.join(minutiae_dir, item.replace(".png", ".txt"))
            )
            orientation_map_paths.append(
                os.path.join(orientation_maps_dir, item.replace(".png", ".npy"))
            )

    dataset = FingerprintOrientationDataset(
        orientation_paths=orientation_map_paths,
        minutiae_paths=minutiae_paths,
        continuous_paths=cont_paths,
        full_paths=orig_paths,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = FingerprintUNet(in_channels=3, out_channels=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train(model, dataloader, cfg, optimizer, save_model=True)
