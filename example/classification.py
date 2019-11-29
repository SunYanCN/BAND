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
from transformers import *

from band.dataset import ChnSentiCorp
from band.progress import classification_convert_examples_to_features

USE_XLA = False
USE_AMP = False

EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5


tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

dataset = ChnSentiCorp(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

train_number, eval_number, test_number = dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = classification_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")
valid_dataset = classification_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

config = BertConfig.from_pretrained("bert-base-chinese", num_labels=dataset.num_labels)
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
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

# test_dataset = classification_convert_examples_to_features(data['test'], tokenizer, max_length=MAX_SEQ_LEN,
#                                                            label_list=label,
#                                                            output_mode="classification")

# test_dataset = train_dataset.batch(TEST_BATCH_SIZE).repeat(1)
# test_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
# loss, accuracy = model.evaluate(data['train'], )

# saved_model_path = "./saved_models/{}".format(int(time.time()))
# model.save(saved_model_path, save_format="tf")
"""
USE_XLA 

Epoch 1/5
600/600 [==============================] - 573s 955ms/step - loss: 0.2824 - accuracy: 0.8940 - val_loss: 0.2162 - val_accuracy: 0.9192
Epoch 2/5
600/600 [==============================] - 309s 515ms/step - loss: 0.1577 - accuracy: 0.9444 - val_loss: 0.2361 - val_accuracy: 0.9233
Epoch 3/5
600/600 [==============================] - 309s 514ms/step - loss: 0.0993 - accuracy: 0.9678 - val_loss: 0.2270 - val_accuracy: 0.9333
Epoch 4/5
600/600 [==============================] - 307s 512ms/step - loss: 0.0702 - accuracy: 0.9780 - val_loss: 0.2492 - val_accuracy: 0.9300
Epoch 5/5
600/600 [==============================] - 310s 516ms/step - loss: 0.0572 - accuracy: 0.9815 - val_loss: 0.2675 - val_accuracy: 0.9300

The auto mixed precision graph optimizer is only designed for GPUs of Volta generation (SM 7.0) or later, and if no such GPUs are detected (Titan X is pre-Volta) then it will print the message you see.
"""

"""
NO USE_XLA 

Epoch 1/5
600/600 [==============================] - 355s 592ms/step - loss: 0.2685 - accuracy: 0.8976 - val_loss: 0.2427 - val_accuracy: 0.9142
Epoch 2/5
600/600 [==============================] - 332s 554ms/step - loss: 0.1707 - accuracy: 0.9420 - val_loss: 0.1824 - val_accuracy: 0.9258
Epoch 3/5
600/600 [==============================] - 332s 554ms/step - loss: 0.0934 - accuracy: 0.9686 - val_loss: 0.1995 - val_accuracy: 0.9383

"""