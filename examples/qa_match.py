"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: qa_match.py
@time: 2019-11-22 16:49:32
"""
import time
import tensorflow as tf
from transformers import *

from band.dataset import LCQMC
from band.progress import classification_convert_examples_to_features

dataset = LCQMC(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese')

train_dataset = classification_convert_examples_to_features(data['train'], tokenizer, max_length=20, label_list=label,
                                                            output_mode="classification")
valid_dataset = classification_convert_examples_to_features(data['validation'], tokenizer, max_length=20,
                                                            label_list=label,
                                                            output_mode="classification")

train_dataset = train_dataset.shuffle(100).batch(32, drop_remainder=True).repeat(2)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = valid_dataset.batch(64)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

saved_model_path = "./saved_models/{}".format(int(time.time()))
model.save(saved_model_path, save_format="tf")

