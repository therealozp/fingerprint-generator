import torch
from model import OrientationUnwrappingNet, UnwrappingCompositeLoss
from dataset import OrientationDataset
from torch.utils.data import DataLoader
import os

from tqdm import tqdm

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrientationUnwrappingNet(auxiliary_channels=0).to(DEVICE)
criterion = UnwrappingCompositeLoss(lambda_per=1.0, lambda_grad=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

wrapped_data_dir = "./phase_data/wrapped"
filename_list = sorted(os.listdir(wrapped_data_dir))

ground_truth_data_dir = "./phase_data/ground_truth"
train_split = int(0.8 * len(filename_list))

train_list = filename_list[:train_split]
test_list = filename_list[train_split:]

train_ds = OrientationDataset(
    wrapped_data_dir=wrapped_data_dir,
    unwrapped_data_dir=ground_truth_data_dir,
    wrapped_data_list=train_list,
    unwrapped_data_list=train_list,
)

test_ds = OrientationDataset(
    wrapped_data_dir=wrapped_data_dir,
    unwrapped_data_dir=ground_truth_data_dir,
    wrapped_data_list=test_list,
    unwrapped_data_list=test_list,
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

NUM_EPOCHS = 30
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        input_wrapped = batch["wrapped"].to(DEVICE)  # (B, 1, H, W)
        gt_unwrapped = batch["unwrapped"].to(DEVICE)  # (B, 1, H, W)

        # Forward Pass
        output = model(input_wrapped)
        prediction = output["unwrapped"]

        # Loss Calculation
        loss, _ = criterion(prediction, gt_unwrapped, input_wrapped)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * input_wrapped.size(0)

    avg_train_loss = train_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Training Loss: {avg_train_loss:.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(
            test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"
        ):
            input_wrapped = batch["wrapped"].to(DEVICE)
            gt_unwrapped = batch["unwrapped"].to(DEVICE)

            output = model(input_wrapped)
            prediction = output["unwrapped"]

            loss, _ = criterion(prediction, gt_unwrapped, input_wrapped)
            val_loss += loss.item() * input_wrapped.size(0)

    avg_val_loss = val_loss / len(test_ds)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Validation Loss: {avg_val_loss:.4f}")

# Save the model checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/orientation_unwrapping_net.pth")
