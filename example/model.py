"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: model.py
@time: 2019-11-29 16:32:29
"""

# from keras.models import load_model
# import os
#
# model = load_model(os.path.join('models', 'bert-base-chinese-tf_model.h5'))
# print(model.summary())
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)
