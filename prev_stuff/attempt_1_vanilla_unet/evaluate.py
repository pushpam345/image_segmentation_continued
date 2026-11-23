import os
import tensorflow as tf

# Import your local project files
from config import HEIGHT, WIDTH, NUM_CLASSES
from data_preprocess_pipeline import build_dataset
from v1_vanilla_unet.model_unet import build_unet  # Ensure this filename matches your U-Net script
from v1_vanilla_unet.metrics import MeanIoU

# Paths - Ensure these match your local folder structure
TEST_IMG_DIR = "data/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "data/Offroad_Segmentation_testImages/Segmentation"
CHECKPOINT_PATH = os.path.join("checkpoints_2", "best_model.h5")

def main() -> None:
    # 1. Verification
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing model file: {CHECKPOINT_PATH}. Did you download it from Colab?")

    print("--- Initializing Evaluation Logic ---")

    # 2. Build Architecture Fresh
    # This bypasses the Keras version mismatch (batch_shape error)
    model = build_unet((HEIGHT, WIDTH, 3), NUM_CLASSES)
    
    # 3. Load Only the Weights
    # By using bypass_sign_check or just load_weights, we get the 'brains' without the metadata errors
    try:
        print(f"Loading weights from: {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
        print("Successfully loaded weights.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 4. Compile the model
    # We must compile before we can run .evaluate()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[MeanIoU(NUM_CLASSES)]
    )

    # 5. Prepare Test Dataset
    # shuffle=False is critical for evaluation so images/masks stay in order
    test_dataset = build_dataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        shuffle=False
    )

    # 6. Run Evaluation
    print("\nStarting evaluation on test dataset...")
    # verbose=1 shows the progress bar
    results = model.evaluate(test_dataset, verbose=1)

    # 7. Print Results
    # model.metrics_names usually returns ['loss', 'mean_io_u']
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    
    for name, value in zip(model.metrics_names, results):
        print(f"{name.upper()}: {value:.6f}")
    print("="*30)

if __name__ == "__main__":
    # Ensure GPU memory growth is enabled if running locally
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    main()
    