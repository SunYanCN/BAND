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
from band.model import TFBertForSequenceClassification
from band.dataset import ChnSentiCorp
from band.progress import classification_convert_examples_to_features


USE_XLA = False
USE_AMP = False

EPOCHS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5
SAVE_MODEL = False
pretrained_dir = "/home/band/models"

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

dataset = ChnSentiCorp(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

train_number, eval_number, test_number = dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num

tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
train_dataset = classification_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")
valid_dataset = classification_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")
test_dataset = classification_convert_examples_to_features(data['test'], tokenizer, max_length=MAX_SEQ_LEN,
                                                           label_list=label,
                                                           output_mode="classification")

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

config = BertConfig.from_pretrained(pretrained_dir, num_labels=dataset.num_labels)
model = TFBertForSequenceClassification.from_pretrained(pretrained_dir, config=config, from_pt=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
if USE_AMP:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number // BATCH_SIZE,
                    validation_data=valid_dataset,
                    validation_steps=eval_number // EVAL_BATCH_SIZE)

loss, accuracy = model.evaluate(test_dataset, steps=test_number // TEST_BATCH_SIZE)
print(loss, accuracy)

if SAVE_MODEL:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
