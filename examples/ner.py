"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: ner.py
@time: 2019-11-22 16:49:32
"""
import time
import tensorflow as tf
from transformers import BertTokenizer, BertConfig
from band.progress import ner_convert_examples_to_features
from band.dataset import MSRA_NER
from band.seqeval.callbacks import F1Metrics
from band.model import TFBertForTokenClassification

EPOCHS = 3
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5
SAVE_MODEL = False
pretrained_dir = "/home/band/models"

dataset = MSRA_NER(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

train_number, eval_number, test_number = dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num

config = BertConfig.from_pretrained(pretrained_dir, num_labels=dataset.num_labels)
tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
model = TFBertForTokenClassification.from_pretrained(pretrained_dir, config=config, from_pt=True)

train_dataset = ner_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                 label_list=label,
                                                 pad_token_label_id=0)
valid_dataset = ner_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                 label_list=label,
                                                 pad_token_label_id=0)
test_dataset = ner_convert_examples_to_features(data['test'], tokenizer, max_length=MAX_SEQ_LEN,
                                                label_list=label,
                                                pad_token_label_id=0)

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())

f1 = F1Metrics(dataset.get_labels(), validation_data=valid_dataset,
               steps=eval_number // EVAL_BATCH_SIZE)
history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number // BATCH_SIZE,
                    callbacks=[f1])

loss, accuracy = model.evaluate(test_dataset, steps=test_number//TEST_BATCH_SIZE)
if SAVE_MODEL:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
