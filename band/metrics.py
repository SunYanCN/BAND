"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: metrics.py
@time: 2019-12-02 19:21:55
"""

import tensorflow as tf
from tensorflow import keras


class MaskedSparseCategoricalCrossEntropy(keras.metrics.Metric):
    def __init__(self, name='masked_sparse_categorical_crossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossEntropy, self).__init__(name=name, **kwargs)
        self.true_postives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, tf.int32), tf.cast(y_true, tf.int32))
        y_true = tf.cast(y_true, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)

        return self.true_postives.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return tf.identity(self.true_postives)

    def reset_states(self):
        self.true_postives.assign(0.)
