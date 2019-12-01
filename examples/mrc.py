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

EPOCHS = 3
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5

dataset = MSRA_NER(save_path="/tmp/band")
data, label = dataset.data, dataset.label

dataset.dataset_information()

train_number, eval_number, test_number = dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num

config = BertConfig.from_pretrained("bert-base-chinese", num_labels=dataset.num_labels)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-chinese', config=config)


train_dataset = ner_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN, label_list=label)
valid_dataset = ner_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN, label_list=label)


train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE,drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())
history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number//BATCH_SIZE,
                    validation_data=valid_dataset,
                    validation_steps=eval_number//EVAL_BATCH_SIZE)

loss, accuracy = model.evaluate(data['train'], batch_size=TEST_BATCH_SIZE)
#
#
# saved_model_path = "./saved_models/{}".format(int(time.time()))
# model.save(saved_model_path, save_format="tf")
