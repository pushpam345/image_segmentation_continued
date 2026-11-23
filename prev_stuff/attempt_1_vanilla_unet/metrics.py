import tensorflow as tf

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_states(self):
        self.miou.reset_states()
