import tensorflow as tf
import numpy as np
import os

from config import *
from data_preprocess_pipeline import build_dataset
from v1_vanilla_unet.model_unet import build_unet
from v1_vanilla_unet.metrics import MeanIoU
from config import HEIGHT, WIDTH, NUM_CLASSES, CLASS_IDS

# ---------------------------
# Directories
# ---------------------------
CHECKPOINT_DIR = "checkpoints_2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# Datasets
# ---------------------------
# Ensure these return images normalized to [0, 1] and masks mapped to [0, 9]
train_ds = build_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, shuffle=True)
val_ds   = build_dataset(VAL_IMG_DIR, VAL_MASK_DIR, shuffle=False)

# ---------------------------
# Model Initialization
# ---------------------------
# 1. Build the fresh architecture
model = build_unet(
    input_shape=(HEIGHT, WIDTH, 3),
    num_classes=NUM_CLASSES
)

# 2. Compile it (This avoids the MeanIoU constructor error)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=[MeanIoU(num_classes=NUM_CLASSES)]
)

# 3. Load weights from the best model if it exists
checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.h5")

if os.path.exists(checkpoint_path):
    print(f"\n--- Found checkpoint. Loading weights from: {checkpoint_path} ---")
    # Using load_weights bypasses the custom_objects deserialization bug
    model.load_weights(checkpoint_path)
    print("--- Weights loaded successfully. Resuming training... ---\n")
else:
    print("\n--- No checkpoint found. Starting training from scratch. ---\n")

# ---------------------------
# GPU memory logger
# ---------------------------
class GPUMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            mem = tf.config.experimental.get_memory_info("GPU:0")
            print(
                f"[GPU] Used: {mem['current']/1e9:.2f} GB | "
                f"Peak: {mem['peak']/1e9:.2f} GB"
            )
        except:
            pass

# ---------------------------
# Callbacks
# ---------------------------
callbacks = [
    # Best model (by val_mean_iou)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.h5"),
        monitor="val_mean_iou",
        mode="max",
        save_best_only=True,
        save_weights_only=False, # Saves full model for backup, but we use load_weights
        verbose=1
    ),

    # Last model (every epoch)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "last_model.h5"),
        save_best_only=False,
        verbose=0
    ),

    # Learning Rate Scheduler to help with the 22% IoU problem
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=2, 
        min_lr=1e-7, 
        verbose=1
    ),

    GPUMemory()
]

# ---------------------------
# Training
# ---------------------------
# Setting initial_epoch=5 means the next epoch run will be labeled "Epoch 6"
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    initial_epoch=5, 
    callbacks=callbacks
)

# ---------------------------
# Save training history
# ---------------------------
history_file = os.path.join(CHECKPOINT_DIR, "training_history.npz")
np.savez(
    history_file,
    loss=history.history["loss"],
    val_loss=history.history["val_loss"],
    mean_iou=history.history["mean_iou"],
    val_mean_iou=history.history["val_mean_iou"]
)

print(f"\nTraining complete. History saved to {history_file}")