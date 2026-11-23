import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CLASS_IDS,
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    CLASS_LABELS,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    BATCH_SIZE,
    WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY,
    LEARNING_RATE,
    EPOCHS,
)
from data_preprocess_pipeline import build_dataset, remap_mask, get_sorted_file_paths
from model import make_Unet_with_resnet50
from collections import defaultdict

TRAIN_CLASS_FREQ = [
    0.03529683,
    0.05933981,
    0.18872895,
    0.01099030,
    0.04394827,
    0.02807256,
    0.00078129,
    0.01198249,
    0.24447772,
    0.37638177,
]
print(CLASS_LABELS)
print(TRAIN_CLASS_FREQ)

freq = np.array(TRAIN_CLASS_FREQ, dtype=np.float32)
weights = 1.0 / np.log(1.02 + freq)
print("unclipped weights:", weights)
weights = np.clip(weights, 1.0, WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY)
print("clipped weights", weights)
CLASS_WEIGHTS = tf.constant(weights, tf.float32)
print(CLASS_WEIGHTS.numpy())


def _safe_register_keras_serializable(package, name=None):
    def decorator(obj):
        reg_name = f"{package}>{name or obj.__name__}"
        if reg_name in tf.keras.utils.get_custom_objects():
            return obj
        return tf.keras.utils.register_keras_serializable(package=package, name=name)(obj)

    return decorator


@_safe_register_keras_serializable(package="Custom")
def weighted_sparse_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    pixel_weights = tf.gather(CLASS_WEIGHTS, y_true)
    weighted_ce = ce * pixel_weights
    return tf.reduce_mean(weighted_ce)


@_safe_register_keras_serializable(package="Custom")
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)

    num_classes = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

    axes = [0, 1, 2]
    intersection = tf.reduce_sum(y_true_oh * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true_oh + y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - tf.reduce_mean(dice)


@_safe_register_keras_serializable(package="Custom")
def total_loss(y_true, y_pred):
    return weighted_sparse_ce(y_true, y_pred) + 0.7 * dice_loss(y_true, y_pred)


class DatasetIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="dataset_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(num_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        tp_list = []
        fp_list = []
        fn_list = []

        for c in range(self.num_classes):
            true_c = tf.equal(y_true, c)
            pred_c = tf.equal(y_pred, c)

            tp = tf.reduce_sum(tf.cast(true_c & pred_c, tf.float32))
            fp = tf.reduce_sum(tf.cast(~true_c & pred_c, tf.float32))
            fn = tf.reduce_sum(tf.cast(true_c & ~pred_c, tf.float32))

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        tp_vec = tf.stack(tp_list)
        fp_vec = tf.stack(fp_list)
        fn_vec = tf.stack(fn_list)

        self.tp.assign_add(tp_vec)
        self.fp.assign_add(fp_vec)
        self.fn.assign_add(fn_vec)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class DatasetMeanIoU(DatasetIoU):
    def __init__(self, num_classes, name="dataset_mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        iou = tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn)
        return tf.reduce_mean(iou)

    def get_config(self):
        config = super().get_config()
        return config


class DatasetMeanIoUPresentOnly(DatasetIoU):
    def __init__(self, num_classes, name="dataset_mean_iou_present_only", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        gt_pixels = self.tp + self.fn
        pred_pixels = self.tp + self.fp
        present = tf.logical_or(tf.greater(gt_pixels, 0), tf.greater(pred_pixels, 0))
        iou = tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn)
        present_iou = tf.boolean_mask(iou, present)
        return tf.cond(
            tf.size(present_iou) > 0,
            lambda: tf.reduce_mean(present_iou),
            lambda: tf.constant(0.0, dtype=iou.dtype),
        )

    def get_config(self):
        config = super().get_config()
        return config


class DatasetWorstIoU(DatasetIoU):
    def __init__(self, num_classes, name="dataset_worst_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        iou = tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn)
        return tf.reduce_min(iou)

    def get_config(self):
        config = super().get_config()
        return config


class DatasetClassIoU(DatasetIoU):
    def __init__(self, class_index, class_name, num_classes=10, **kwargs):
        super().__init__(num_classes=num_classes, name=f"dataset_iou_{class_name}", **kwargs)
        self.class_index = class_index
        self.class_name = class_name

    def result(self):
        tp = self.tp[self.class_index]
        fp = self.fp[self.class_index]
        fn = self.fn[self.class_index]
        return tf.math.divide_no_nan(tp, tp + fp + fn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_index": self.class_index,
                "class_name": self.class_name,
            }
        )
        return config


metrics = [
    DatasetMeanIoUPresentOnly(len(CLASS_IDS)),
    DatasetMeanIoU(len(CLASS_IDS)),
    DatasetWorstIoU(len(CLASS_IDS)),
]

for i, label in enumerate(CLASS_LABELS):
    metrics.append(DatasetClassIoU(i, label, len(CLASS_IDS)))


test_dataset = build_dataset(TEST_IMG_DIR, TEST_MASK_DIR, isTrainingDataset=False, performAugment=False)

MODEL_PATH = "to_test.keras"

custom_objects = {
    "weighted_sparse_ce": weighted_sparse_ce,
    "dice_loss": dice_loss,
    "total_loss": total_loss,
    "DatasetIoU": DatasetIoU,
    "DatasetMeanIoU": DatasetMeanIoU,
    "DatasetMeanIoUPresentOnly": DatasetMeanIoUPresentOnly,
    "DatasetWorstIoU": DatasetWorstIoU,
    "DatasetClassIoU": DatasetClassIoU,
}


def load_any_model(path, custom_objects):
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    is_zip = zipfile.is_zipfile(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".keras" and is_zip:
        try:
            import keras

            if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
                return keras.saving.load_model(path, custom_objects=custom_objects, compile=False)
            if hasattr(keras, "models") and hasattr(keras.models, "load_model"):
                return keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        except Exception:
            pass
        h5_path = os.path.splitext(path)[0] + ".h5"
        if tf.io.gfile.exists(h5_path):
            return tf.keras.models.load_model(h5_path, custom_objects=custom_objects, compile=False)
        raise RuntimeError(
            "Unable to load .keras zip model with this TensorFlow/Keras. "
            "Re-save as .h5 (e.g., in Colab) and set MODEL_PATH to that file."
        )
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)


model = load_any_model(MODEL_PATH, custom_objects)
metric = DatasetMeanIoUPresentOnly(len(CLASS_IDS))

for images, masks in test_dataset:
    preds = model(images, training=False)
    metric.update_state(masks, preds)

print("Test Dataset Mean IoU (present only):", metric.result().numpy())

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=total_loss,
    metrics=metrics,
)

eval_results = model.evaluate(test_dataset, return_dict=True)
print("Eval results:", eval_results)


def compute_per_class_iou_single(y_true, y_pred, num_classes):
    ious = []
    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)

        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)

    return np.array(ious)


def visualize_prediction(model, dataset, class_labels):
    for images, masks in dataset.take(1):
        break

    image = images[0:1]
    true_mask = masks[0].numpy()

    pred = model.predict(image)
    pred_mask = np.argmax(pred[0], axis=-1)

    pixel_error = np.mean(pred_mask != true_mask) * 100

    ious = compute_per_class_iou_single(
        true_mask,
        pred_mask,
        num_classes=len(class_labels),
    )

    valid_ious = np.where(np.isnan(ious), 1.0, ious)
    worst_class_idx = np.argmin(valid_ious)
    worst_class_name = class_labels[worst_class_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow((image[0].numpy() + [103.939, 116.779, 123.68])[:, :, ::-1] / 255.0)
    axes[0].set_title("RGB Image")

    axes[1].imshow(true_mask, cmap="tab20")
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(pred_mask, cmap="tab20")
    axes[2].set_title("Predicted Mask")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Pixel Error: {pixel_error:.2f}% | "
        f"Worst Class: {worst_class_name} (IoU={ious[worst_class_idx]:.4f})"
    )

    plt.show()

    print("\nPer-class IoU:")
    for label, iou in zip(class_labels, ious):
        print(f"{label:15s}: {iou:.4f}")

    print(f"\nPixel Error %: {pixel_error:.2f}%")
    print(f"Worst Class: {worst_class_name}")


visualize_prediction(model, test_dataset, CLASS_LABELS)
