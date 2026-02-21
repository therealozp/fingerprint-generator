import torch
import torch.nn as nn
import os
# NEW IMPORTS
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import FingerprintUNet
from loss_functions import FingerprintLoss
from torch.utils.data import DataLoader
from dataset import OrientationFrequencyDataset

from dataclasses import dataclass
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

@dataclass
class TrainConfig:
    # UPDATED CONFIG
    device: str = "cuda" 
    lr: float = 5e-4
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 2
    amp: bool = True

def split_into_patches(x, patch_size=64):
    B, C, H, W = x.shape
    grid_h = H // patch_size
    grid_w = W // patch_size
    
    x = x.view(B, C, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, C, patch_size, patch_size)

    return x

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    optimizer,
    # ADDED RANK/SAMPLER
    rank: int,
    train_sampler: DistributedSampler,
    patch_size,
    scheduler=None,
    save_model=False,
    load_best=False,
    load_path=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    if load_best:
        assert load_path, "Specify best model paths in `load_path`."
        if rank == 0: print("Loading best model weights...")
        checkpoint = torch.load(load_path, map_location=f"cuda:{rank}")
        model.module.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("lr_sched", None):
            scheduler.load_state_dict(checkpoint["lr_sched"])
    else:
        if rank == 0: print("Training from scratch.")

    lowest_validation_loss = float("inf")
    best_epoch = -1
    criterion = FingerprintLoss()

    for epoch in range(cfg.epochs):
        # NEW: SET SAMPLER EPOCH
        train_sampler.set_epoch(epoch)
        
        model.train()
        running = 0.0
        
        # NEW: ONLY SHOW PROGRESS ON RANK 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0))

        for step, input in enumerate(pbar):
            inputs = input["inputs"].to(rank)
            target_c = input["target_continuous"].to(rank)
            target_f = input["target_full"].to(rank)
            spiral_phasor = input["spiral_phasor"].to(rank)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)

            loss, l_cont, l_full = criterion(
                pred=pred,
                spiral_phasor=spiral_phasor,
                target_cont=target_c,
                target_full=target_f,
            )

            loss.backward()
            optimizer.step()
            running += loss.item()

        avg = running / max(1, len(train_loader))
        
        if rank == 0:
            print(f"epoch {epoch+1:03d}/{cfg.epochs} | cont_loss = {l_cont:.6f} | full_loss = {l_full:.6f} | total loss = {avg:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input in val_loader:
                inputs = input["inputs"].to(rank)
                target_c = input["target_continuous"].to(rank)
                target_f = input["target_full"].to(rank)
                spiral_phasor = input["spiral_phasor"].to(rank)

                inputs_patches = split_into_patches(inputs, patch_size=patch_size)
                target_c_patches = split_into_patches(target_c, patch_size=patch_size)
                target_f_patches = split_into_patches(target_f, patch_size=patch_size)
                spiral_phasor_patches = split_into_patches(spiral_phasor, patch_size=patch_size)

                pred_patches = model(inputs_patches)
                
                loss, _, _ = criterion(
                    pred_patches, 
                    spiral_phasor_patches, 
                    target_c_patches, 
                    target_f_patches
                )

                val_loss += loss.item()

        val_tensor = torch.tensor(val_loss).to(rank)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_tensor.item() / (max(1, len(val_loader)) * dist.get_world_size())

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if rank == 0:
            print(f"validation loss: {avg_val_loss}")

            if save_model and avg_val_loss < lowest_validation_loss:
                lowest_validation_loss = avg_val_loss
                best_epoch = epoch

                checkpoint = {
                    "epoch": epoch,
                    # NEW: SAVE .module.state_dict()
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": scheduler.state_dict() if scheduler else None,
                }
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(checkpoint, f"checkpoints/patches_{patch_size}_1e-4.pth")
                
                print("best validation loss:", lowest_validation_loss)
            else:
                print(f"have not found a better loss since {epoch - best_epoch} epochs ago.")


if __name__ == "__main__":
    # NEW: DDP INITIALIZATION
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    cfg = TrainConfig(epochs=5000, batch_size=64, lr=1e-4)

    # ... [Data Path Definitions remain the same] ...
    original_images_dir = "/green/data/data_v2/full_images"
    target_images_dir = "/green/data/data_v2/cont_images"
    minutiae_dir = "/green/data/data_v2/minutiae_locations"
    orientation_maps_dir = "/green/data/data_v2/orientation_maps"
    freq_maps_dir = "/green/data/data_v2/freq_maps"

    orig_paths = sorted([os.path.join(original_images_dir, i) for i in os.listdir(original_images_dir) if i.endswith(".png")])
    cont_paths = sorted([os.path.join(target_images_dir, i) for i in os.listdir(original_images_dir) if i.endswith(".png")])
    minutiae_paths = sorted([os.path.join(minutiae_dir, i.replace(".png", ".txt")) for i in os.listdir(original_images_dir) if i.endswith(".png")])
    orientation_map_paths = sorted([os.path.join(orientation_maps_dir, i.replace(".png", ".npy")) for i in os.listdir(original_images_dir) if i.endswith(".png")])
    frequency_map_paths = sorted([os.path.join(freq_maps_dir, i.replace(".png", ".npy")) for i in os.listdir(original_images_dir) if i.endswith(".png")])

    train_split = int(0.85 * len(orig_paths))
    patch_size = 64

    train_dataset = OrientationFrequencyDataset(
        orientation_paths=orientation_map_paths[:train_split],
        frequency_paths=frequency_map_paths[:train_split],
        minutiae_paths=minutiae_paths[:train_split],
        continuous_paths=cont_paths[:train_split],
        full_paths=orig_paths[:train_split],
        crop_size=patch_size,
    )

    val_dataset = OrientationFrequencyDataset(
        orientation_paths=orientation_map_paths[train_split:],
        minutiae_paths=minutiae_paths[train_split:],
        frequency_paths=frequency_map_paths[train_split:],
        continuous_paths=cont_paths[train_split:],
        full_paths=orig_paths[train_split:],
    )

    # NEW: WRAP MODEL IN DDP
    model = FingerprintUNet(in_channels=3, out_channels=2).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # NEW: DISTRIBUTED SAMPLERS
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler, # NEW
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler, # NEW
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=50)
    # scheduler = CosineAnnealingWarmRestarts(
    #         optimizer, 
    #         T_0=50,       
    #         T_mult=2,     
    #         eta_min=1e-6
    #     )

    train(
        model,
        train_loader,
        val_loader,
        cfg,
        optimizer,
        rank=rank,
        train_sampler=train_sampler,
        scheduler=None,
        save_model=True,
        load_best=False, 
        patch_size=patch_size,
        # load_path="checkpoints/ckpt_parallel_1e-3.pth",
    )

    dist.destroy_process_group()
