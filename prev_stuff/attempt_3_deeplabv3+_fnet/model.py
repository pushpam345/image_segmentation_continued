import segmentation_models_pytorch as smp
from config import NUM_CLASSES


def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model
