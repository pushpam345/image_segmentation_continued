HEIGHT = 544
WIDTH = 960
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 1e-4
DEVICE = "cuda"   # Colab gives GPU if enabled
TRAIN_CLASS_FREQ = [
    0.03529683,  # Trees
    0.05933981,  # Lush Bushes
    0.18872895,  # Dry Grass
    0.01099030,  # Dry Bushes
    0.04394827,  # Ground Clutter
    0.02807256,  # Flower
    0.00078129,  # Logs
    0.01198249,  # Rocks
    0.24447772,  # Landscape
    0.37638177,  # Sky
]

CLASS_IDS = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_LABELS = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
                "Ground Clutter", "Flower", "Logs", "Rocks", "Landscape", "Sky"]
NUM_CLASSES = len(CLASS_IDS)

TRAIN_IMG_DIR = "data/Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK_DIR = "data/Offroad_Segmentation_Training_Dataset/train/Segmentation"

VAL_IMG_DIR = "data/Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK_DIR = "data/Offroad_Segmentation_Training_Dataset/val/Segmentation"

TEST_IMG_DIR = "data/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "data/Offroad_Segmentation_testImages/Segmentation"

WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY = 5.0
