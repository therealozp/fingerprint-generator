import torch
import torch.nn as nn

from model import FingerprintUNet
from loss_functions import FingerprintLoss
from torch.utils.data import DataLoader
from dataset import OrientationFrequencyDataset

from dataclasses import dataclass
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os


@dataclass
class TrainConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr: float = 5e-4
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    optimizer,
    scheduler=None,
    save_model=False,
    load_best=False,
    load_path=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if load_best:
        assert load_path, "Specify best model paths in `load_path`."
        print("Loading best model weights for training...")
        checkpoint = torch.load(load_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("lr_sched", None):
            scheduler.load_state_dict(checkpoint["lr_sched"])
        print("Best weights loaded.")
    else:
        print("Training from scratch.")

    model.to(cfg.device)
    lowest_validation_loss = float("inf")

    criterion = FingerprintLoss()

    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        running = 0.0
        loss_from_continuous_phase = 0.0
        loss_from_full_phase = 0.0
        sin_cos_normalization_loss = 0.0

        for step, input in enumerate(
            tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.epochs}")
        ):
            inputs = input["inputs"].to(cfg.device)  # (B, 3, H, W)
            target_c = input["target_continuous"].to(cfg.device)  # (B, 1, H, W)
            target_f = input["target_full"].to(cfg.device)  # (B, 1, H, W)
            spiral_phasor = input["spiral_phasor"].to(cfg.device)  # (B, 2, H, W)

            optimizer.zero_grad()

            pred = model(inputs)

            # Calculate Loss
            loss, l_cont, l_full, l_norm = criterion(
                pred=pred,
                spiral_phasor=spiral_phasor,
                target_cont=target_c,
                target_full=target_f,
            )

            loss.backward()
            optimizer.step()

            running += loss.item()
            loss_from_continuous_phase += l_cont.item()
            loss_from_full_phase += l_full.item()
            sin_cos_normalization_loss += l_norm.item()

        avg = running / max(1, len(train_loader))

        print(
            f"epoch {epoch+1:03d}/{cfg.epochs} | total loss = {avg:.6f} | cont phase loss = {loss_from_continuous_phase / max(1, len(train_loader)):.6f} | full phase loss = {loss_from_full_phase / max(1, len(train_loader)):.6f} | norm loss = {sin_cos_normalization_loss / max(1, len(train_loader)):.6f}"
        )

        # eval step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, input in enumerate(
                tqdm(val_loader, desc=f"Validation {epoch+1}/{cfg.epochs}")
            ):
                inputs = input["inputs"].to(cfg.device)  # (B, 3, H, W)
                target_c = input["target_continuous"].to(cfg.device)  # (B, 1, H, W)
                target_f = input["target_full"].to(cfg.device)  # (B, 1, H, W)
                spiral_phasor = input["spiral_phasor"].to(cfg.device)  # (B, 2, H, W)

                pred = model(inputs)

                loss, _, _, _ = criterion(
                    pred=pred,
                    spiral_phasor=spiral_phasor,
                    target_cont=target_c,
                    target_full=target_f,
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print("validation loss:", avg_val_loss)
        print("best validation loss:", lowest_validation_loss)
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if save_model and val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            print("saving best model with validation loss:", lowest_validation_loss)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": scheduler.state_dict() if scheduler else None,
            }
            torch.save(checkpoint, "checkpoints/ckpt.pth")


if __name__ == "__main__":
    cfg = TrainConfig(epochs=5000, batch_size=16)

    original_images_dir = "/green/data/data_v2/full_images"
    target_images_dir = "/green/data/data_v2/cont_images"
    minutiae_dir = "/green/data/data_v2/minutiae_locations"
    orientation_maps_dir = "/green/data/data_v2/orientation_maps"
    freq_maps_dir = "/green/data/data_v2/freq_maps"

    orig_paths = []
    cont_paths = []
    minutiae_paths = []
    orientation_map_paths = []
    frequency_map_paths = []

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
            frequency_map_paths.append(
                os.path.join(freq_maps_dir, item.replace(".png", ".npy"))
            )

    orig_paths.sort()
    cont_paths.sort()
    minutiae_paths.sort()
    orientation_map_paths.sort()
    frequency_map_paths.sort()

    train_split = int(0.85 * len(orig_paths))

    train_dataset = OrientationFrequencyDataset(
        orientation_paths=orientation_map_paths[:train_split],
        frequency_paths=frequency_map_paths[:train_split],
        minutiae_paths=minutiae_paths[:train_split],
        continuous_paths=cont_paths[:train_split],
        full_paths=orig_paths[:train_split],
    )

    val_dataset = OrientationFrequencyDataset(
        orientation_paths=orientation_map_paths[train_split:],
        minutiae_paths=minutiae_paths[train_split:],
        frequency_paths=frequency_map_paths[train_split:],
        continuous_paths=cont_paths[train_split:],
        full_paths=orig_paths[train_split:],
    )

    model = FingerprintUNet(in_channels=3, out_channels=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
    )

    train(
        model,
        train_loader,
        val_loader=val_loader,
        cfg=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        save_model=True,
        load_best=False,
        load_path="checkpoints/checkpoint_2.pth",
    )
