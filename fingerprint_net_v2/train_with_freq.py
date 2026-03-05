import torch
import torch.nn as nn

from model import FingerprintUNet
from loss_functions import FingerprintLossv2
from torch.utils.data import DataLoader
from dataset import FingerprintOrientationDataset

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

    criterion = FingerprintLossv2().to(cfg.device)

    opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        loss_items = {
            "mse_cont": 0.0,
            "mse_full": 0.0,
            "ssim_cont": 0.0,
            "ssim_full": 0.0,
            "sobel_cont": 0.0,
            "sobel_full": 0.0,
            "phasor": 0.0,
        }

        for step, inp in enumerate(
            tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.epochs}")
        ):
            optimizer.zero_grad()
            inputs = inp["inputs"].to(cfg.device)  # (B, 3, H, W)
            pred = model(inputs)

            # Calculate Loss
            loss, loss_object = criterion(
                pred=pred,
                spiral_phasor=inp["spiral_phasor"].to(cfg.device),
                cos_cont=inp["cos_cont"].to(cfg.device),
                cos_full=inp["cos_full"].to(cfg.device),
                sin_cont=inp["sin_cont"].to(cfg.device),
                sin_full=inp["sin_full"].to(cfg.device),
            )

            loss.backward()
            optimizer.step()

            running += loss.item()
            for key in loss_items:
                loss_items[key] += loss_object[key].item()

        avg = running / max(1, len(train_loader))
        avg_items = {
            key: val / max(1, len(train_loader)) for key, val in loss_items.items()
        }

        loss_str = " | ".join(f"{key} = {val:.6f}" for key, val in avg_items.items())
        print(f"epoch {epoch+1:03d}/{cfg.epochs} | total loss = {avg:.6f} | {loss_str}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_sched": scheduler.state_dict() if scheduler else None,
        }
        torch.save(checkpoint, "checkpoints/ckpt.pth")

        # # eval step
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for step, inp in enumerate(
        #         tqdm(val_loader, desc=f"Validation {epoch+1}/{cfg.epochs}")
        #     ):
        #         inputs = inp["inputs"].to(cfg.device)  # (B, 3, H, W)

        #         pred = model(inputs)

        #         # Calculate Loss
        #         loss, loss_object = criterion(
        #             pred=pred,
        #             spiral_phasor=inp["spiral_phasor"].to(cfg.device),
        #             cos_cont=inp["cos_cont"].to(cfg.device),
        #             cos_full=inp["cos_full"].to(cfg.device),
        #             sin_cont=inp["sin_cont"].to(cfg.device),
        #             sin_full=inp["sin_full"].to(cfg.device),
        #         )

        #         running += loss.item()
        #         val_loss += loss.item()

        # avg_val_loss = val_loss / max(1, len(val_loader))
        # print("validation loss:", avg_val_loss)
        # print("best validation loss:", lowest_validation_loss)
        # if scheduler is not None:
        #     scheduler.step(avg_val_loss)

        # if save_model and val_loss < lowest_validation_loss:
        #     lowest_validation_loss = val_loss
        #     print("saving best model with validation loss:", lowest_validation_loss)
        #     checkpoint = {
        #         "epoch": epoch,
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_sched": scheduler.state_dict() if scheduler else None,
        #     }
        #     torch.save(checkpoint, "checkpoints/ckpt.pth")


if __name__ == "__main__":
    cfg = TrainConfig(epochs=5000, batch_size=1, lr=1e-3)

    base_dir = "data_v3_single"

    orientation_dir = "orientation_maps"
    minutiae_dir = "minutiae_locations"
    frequency_dir = "freq_maps"
    cos_cont_dir = "cos_cont"
    cos_full_dir = "cos_full"
    sin_cont_dir = "sin_cont"
    sin_full_dir = "sin_full"

    orientation_paths = []
    minutiae_paths = []
    frequency_paths = []
    cos_cont_paths = []
    cos_full_paths = []
    sin_cont_paths = []
    sin_full_paths = []

    for item in os.listdir(os.path.join(base_dir, orientation_dir)):
        if item.endswith(".npy"):
            cos_cont_paths.append(os.path.join(base_dir, cos_cont_dir, item))
            cos_full_paths.append(os.path.join(base_dir, cos_full_dir, item))
            sin_cont_paths.append(os.path.join(base_dir, sin_cont_dir, item))
            sin_full_paths.append(os.path.join(base_dir, sin_full_dir, item))
            minutiae_paths.append(
                os.path.join(base_dir, minutiae_dir, item.replace(".npy", ".txt"))
            )
            orientation_paths.append(os.path.join(base_dir, orientation_dir, item))
            frequency_paths.append(os.path.join(base_dir, frequency_dir, item))

    assert (
        len(orientation_paths)
        == len(minutiae_paths)
        == len(frequency_paths)
        == len(cos_cont_paths)
        == len(cos_full_paths)
        == len(sin_cont_paths)
        == len(sin_full_paths)
    ), "Mismatch in dataset lengths."

    orientation_paths.sort()
    minutiae_paths.sort()
    frequency_paths.sort()
    cos_cont_paths.sort()
    cos_full_paths.sort()
    sin_cont_paths.sort()
    sin_full_paths.sort()

    train_split = int(1 * len(orientation_paths))

    train_dataset = FingerprintOrientationDataset(
        orientation_paths[:train_split],
        minutiae_paths[:train_split],
        frequency_paths[:train_split],
        cos_cont_paths[:train_split],
        cos_full_paths[:train_split],
        sin_cont_paths[:train_split],
        sin_full_paths[:train_split],
    )

    val_dataset = FingerprintOrientationDataset(
        orientation_paths[train_split:],
        minutiae_paths[train_split:],
        frequency_paths[train_split:],
        cos_cont_paths[train_split:],
        cos_full_paths[train_split:],
        sin_cont_paths[train_split:],
        sin_full_paths[train_split:],
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
        scheduler=False,
        save_model=True,
        load_best=False,
        load_path="checkpoints/checkpoint_2.pth",
    )
