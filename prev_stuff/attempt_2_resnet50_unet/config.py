HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE=1e-4

CLASS_IDS = [100, 200, 300,500, 550,600, 700, 800, 7100, 10000]
CLASS_LABELS=["Trees", "Lush Bushes","Dry Grass","Dry Bushes","Ground Clutter","Flower","Logs","Rocks","Landscape","Sky"]
NUM_CLASSES = len(CLASS_IDS)

TRAIN_IMG_DIR = "data/Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK_DIR = "data/Offroad_Segmentation_Training_Dataset/train/Segmentation"

VAL_IMG_DIR = "data/Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK_DIR = "data/Offroad_Segmentation_Training_Dataset/val/Segmentation"

TEST_IMG_DIR = "data/Offroad_Segmentation_testImages/Color_Images"
TEST_MASK_DIR = "data/Offroad_Segmentation_testImages/Segmentation"

WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY=5.0 

# in train.ipynb, i just want you to implement dice loss along side with the weighted cross entropy total_loss = weighted_sparse_ce + 0.7 * dice_loss
# . also in train.ipynb cell 10 . the classes implementeation have something wrong. when i save the model after training and load it afterwards. and i pass these metrics in custom objects. i get an error. run it yourself and look at teh error and fix it cuz it is very important for teh model to be loaded so i can training in pieces. finally implement a new metric name DatasetMeanIoUPresent only which only considers the IOU of classes which are actually present in teh mask . if a class is absent and my model doesnt predit it then DatasetMeanIoU uneccesarily tanks it