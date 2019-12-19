"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: classification.py
@time: 2019-11-22 16:49:32
"""
import time
import tensorflow as tf
from transformers import BertConfig, BertTokenizer
from band.model import TFBertForQuestionAnswering
from band.dataset import Squad
from band.progress import squad_convert_examples_to_features, parallel_squad_convert_examples_to_features

USE_XLA = False
USE_AMP = False

EPOCHS = 1
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5
SAVE_MODEL = False
pretrained_dir = "/home/band/models"

dataset = Squad(save_path="/tmp/band")
data, label = dataset.data, dataset.label

train_number, eval_number = dataset.train_examples_num, dataset.eval_examples_num

tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
train_dataset = parallel_squad_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            doc_stride=128, is_training=True, max_query_length=64)
valid_dataset = parallel_squad_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            doc_stride=128, is_training=False, max_query_length=64)

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

config = BertConfig.from_pretrained(pretrained_dir, max_length=MAX_SEQ_LEN, return_unused_kwargs=True)
model = TFBertForQuestionAnswering.from_pretrained(pretrained_dir, config=config, from_pt=True)

print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)

loss = {'start_position': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'end_position': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
metrics = {'start_position': tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
           'end_position': tf.keras.metrics.SparseCategoricalAccuracy('accuracy')}

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number // BATCH_SIZE,
                    validation_data=valid_dataset,
                    validation_steps=eval_number // EVAL_BATCH_SIZE)

if SAVE_MODEL:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
