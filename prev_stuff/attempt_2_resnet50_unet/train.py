import os
import tensorflow as tf
import numpy as np
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

SAVE_DIR = "/content/drive/My Drive/unet_v4"
os.makedirs(SAVE_DIR, exist_ok=True)

LAST_MODEL_PATH = os.path.join(SAVE_DIR, "last_model.keras")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.keras")
LOG_FILE_PATH = os.path.join(SAVE_DIR, "log.txt")

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


def pred_labels(y_pred):
    return tf.argmax(y_pred, axis=-1, output_type=tf.int32)


def per_class_iou(y_true, y_pred, num_classes=len(CLASS_IDS)):
    y_pred = pred_labels(y_pred=y_pred)
    y_true = tf.cast(y_true, tf.int32)

    ious = []
    for c in range(num_classes):
        true_c = tf.equal(y_true, c)
        pred_c = tf.equal(y_pred, c)
        intersection = tf.reduce_sum(tf.cast(true_c & pred_c, tf.float32))
        union = tf.reduce_sum(tf.cast(true_c | pred_c, tf.float32))
        iou = tf.math.divide_no_nan(intersection, union)
        ious.append(iou)

    return tf.stack(ious)


def mean_iou_metric(y_true, y_pred, num_classes=len(CLASS_IDS)):
    ious = per_class_iou(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    return tf.reduce_mean(ious)


def worst_iou_metric(y_true, y_pred, num_classes=len(CLASS_IDS)):
    ious = per_class_iou(y_true, y_pred, num_classes=num_classes)
    return tf.reduce_min(ious)


def make_class_iou_metric(class_id):
    label = CLASS_LABELS[class_id]

    def class_iou(y_true, y_pred):
        ious = per_class_iou(y_true, y_pred, num_classes=len(CLASS_IDS))
        return ious[class_id]

    class_iou.__name__ = f"iou_{label}"
    return class_iou


metrics = [
    mean_iou_metric,
    worst_iou_metric,
]

for i in range(len(CLASS_IDS)):
    metrics.append(make_class_iou_metric(i))


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


def batch_mean_iou_skip_absent(y_true, y_pred, num_classes):
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    ious = []

    for c in range(num_classes):
        true_c = tf.equal(y_true, c)
        pred_c = tf.equal(y_pred, c)

        union = tf.reduce_sum(tf.cast(true_c | pred_c, tf.float32))
        intersection = tf.reduce_sum(tf.cast(true_c & pred_c, tf.float32))

        if tf.equal(union, 0):
            continue

        iou = tf.math.divide_no_nan(intersection, union)
        ious.append(iou)

    return tf.reduce_mean(ious) if ious else tf.constant(0.0)


def batch_class_iou(y_true, y_pred, class_index):
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    true_c = tf.equal(y_true, class_index)
    pred_c = tf.equal(y_pred, class_index)

    union = tf.reduce_sum(tf.cast(true_c | pred_c, tf.float32))
    intersection = tf.reduce_sum(tf.cast(true_c & pred_c, tf.float32))

    iou = tf.math.divide_no_nan(intersection, union)
    class_present = tf.greater(union, 0)

    return iou, class_present


metrics = [
    DatasetMeanIoUPresentOnly(len(CLASS_IDS)),
    DatasetMeanIoU(len(CLASS_IDS)),
    DatasetWorstIoU(len(CLASS_IDS)),
]

for i, label in enumerate(CLASS_LABELS):
    metrics.append(DatasetClassIoU(i, label, len(CLASS_IDS)))


class DebugDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.last_batch = None

    def generator(self):
        for batch in self.dataset:
            self.last_batch = batch
            yield batch

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=self.dataset.element_spec,
        )


class BatchDebugIoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, debug_dataset, num_classes, class_labels):
        super().__init__()
        self.debug_dataset = debug_dataset
        self.num_classes = num_classes
        self.class_labels = class_labels

    def on_train_batch_end(self, batch, logs=None):
        x, y_true = self.debug_dataset.last_batch
        y_pred = self.model(x, training=False)

        mean_iou = batch_mean_iou_skip_absent(y_true, y_pred, self.num_classes)

        print(f"\n[Batch Debug] mean IoU (skip absent): {mean_iou.numpy():.4f}")

        for i, label in enumerate(self.class_labels):
            iou, class_present = batch_class_iou(y_true, y_pred, i)
            print(f"  {label:15s}: {iou.numpy(): 4f} | present:{class_present}")


train_dataset = build_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, isTrainingDataset=True, performAugment=True)
val_dataset = build_dataset(VAL_IMG_DIR, VAL_MASK_DIR, isTrainingDataset=False, performAugment=False)

debug_train_wrapper = DebugDatasetWrapper(train_dataset.take(10))
debug_train_dataset = debug_train_wrapper.get_dataset()

model = make_Unet_with_resnet50()

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=total_loss,
    metrics=metrics,
)

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

checkpoint_last = ModelCheckpoint(
    filepath=LAST_MODEL_PATH,
    save_weights_only=False,
    save_best_only=False,
    verbose=1,
)

checkpoint_best = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_weights_only=False,
    save_best_only=True,
    monitor="val_dataset_mean_iou_present_only",
    mode="max",
    verbose=1,
)

csv_logger = CSVLogger(LOG_FILE_PATH, append=True)


class PeriodicLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, every_n_epochs=5):
        super().__init__()
        self.log_file = log_file
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch+1}: {logs}\n")
            print(f"[PeriodicLog] Epoch {epoch+1}: logs written to {self.log_file}")


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_last,
        checkpoint_best,
        csv_logger,
        PeriodicLogCallback(LOG_FILE_PATH, every_n_epochs=5),
    ],
)

model.save("abcd.h5", include_optimizer=True)
model.save("abcd_wo_opt.h5")
model.save("abcd.keras")
