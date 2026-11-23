import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OffroadDataset
from model import get_model
from loss import CombinedLoss
from weights import get_inverse_log_weights
from utils import calculate_iou_dataset
from config import *

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = OffroadDataset(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    is_train_dataset=True,
    perform_augmentation=True
)

val_dataset = OffroadDataset(
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    is_train_dataset=False,
    perform_augmentation=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

model = get_model().to(device)
print(model)

class_weights = get_inverse_log_weights(device)


criterion = CombinedLoss(
    class_weights=class_weights,
    weight_ce=1.0,
    weight_dice=1.0
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler()

best_miou = 0.0

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc="Training")

    for images, masks in train_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    print(f"\nTrain Loss: {train_loss:.4f}")

    # -------- VALIDATION --------
    iou_per_class, mean_iou = calculate_iou_dataset(
        model,
        val_loader,
        NUM_CLASSES,
        device
    )

    print("\nPer-Class IoU:")
    for label, iou in zip(CLASS_LABELS, iou_per_class):
        print(f"{label:15s}: {iou.item():.4f}")

    print(f"\nMean IoU: {mean_iou.item():.4f}")

    scheduler.step()

    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    if mean_iou > best_miou:
        best_miou = mean_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(" Best model saved!")

print("\nTraining Complete.")
