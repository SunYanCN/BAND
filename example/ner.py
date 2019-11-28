"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: ner.py
@time: 2019-11-22 16:49:32
"""

import tensorflow as tf
from transformers import *
from band.progress import ner_convert_examples_to_features
from band.dataset import MSRA_NER

dataset = MSRA_NER(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForTokenClassification.from_pretrained('bert-base-chinese')


train_dataset = ner_convert_examples_to_features(data['train'], tokenizer, max_length=128, label_list=label)
valid_dataset = ner_convert_examples_to_features(data['validation'], tokenizer, max_length=128, label_list=label)


train_dataset = train_dataset.shuffle(100).batch(32,drop_remainder=True).repeat(2)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = valid_dataset.batch(64)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)
#
#
# saved_model_path = "./saved_models/{}".format(int(time.time()))
# model.save(saved_model_path, save_format="tf")
