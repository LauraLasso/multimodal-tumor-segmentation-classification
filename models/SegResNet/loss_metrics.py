import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Calculate dice coefficient"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1.0):
    """Calculate dice loss"""
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def combined_loss(y_true, y_pred):
    """Custom Loss Function combining dice and categorical crossentropy"""
    dice = dice_loss(y_true, y_pred)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice + ce

class DiceCoefficient(tf.keras.metrics.Metric):
    """Define custom metrics for multi-class segmentation"""
    def __init__(self, class_id, name="dice_coefficient", **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = tf.cast(y_true[..., self.class_id], tf.float32)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        y_pred_mask = tf.cast(y_pred_class == self.class_id, tf.float32)
        
        intersection = tf.reduce_sum(y_true_mask * y_pred_mask)
        union = tf.reduce_sum(y_true_mask) + tf.reduce_sum(y_pred_mask)
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1)

    def result(self):
        return self.dice_sum / self.count

    def reset_state(self):
        self.dice_sum.assign(0)
        self.count.assign(0)