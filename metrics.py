import tensorflow as tf
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    y_pred_classes = tf.reduce_max(y_pred, axis=-1, keepdims=True)
    precision = precision_m(y_true, y_pred_classes)
    recall = recall_m(y_true, y_pred_classes)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class MAUC(tf.keras.metrics.AUC):

  def __init__(self, num_classes, name='auc', **kwargs):
    super(MAUC, self).__init__(name=name, multi_label=True,**kwargs)
    self.num_classes = num_classes

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, (-1, 1))
    y_pred = tf.reshape(tf.math.argmax(y_pred, -1), (-1, 1))
    #y_true = tf.squeeze(tf.one_hot(y_true, self.num_classes), axis=-2)
    super(MAUC, self).update_state(y_true, y_pred, sample_weight)