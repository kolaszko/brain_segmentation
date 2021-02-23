import tensorflow as tf


def dice_metrics(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return numerator / denominator


def dice_loss(y_true, y_pred):
    return 1 - dice_metrics(y_true, y_pred)
