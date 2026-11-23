import torch
from torch.utils.data import DataLoader

from dataset import OffroadDataset
from model import get_model
from utils import calculate_iou_dataset
from config import *

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD TEST DATA =================
test_dataset = OffroadDataset(
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    is_train_dataset=False,
    perform_augmentation=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ================= LOAD MODEL =================
model = get_model().to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print("Model loaded successfully.")

# ================= EVALUATE =================
iou_per_class, mean_iou = calculate_iou_dataset(
    model,
    test_loader,
    NUM_CLASSES,
    device
)

print("\nTest Per-Class IoU:")
for label, iou in zip(CLASS_LABELS, iou_per_class):
    print(f"{label:15s}: {iou.item():.4f}")

print(f"\nTest Mean IoU: {mean_iou.item():.4f}")
