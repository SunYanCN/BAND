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
from band.dataset import MSRA_NER
from band.seqeval.callbacks import F1Metrics
from band.model import TFBertForTokenClassification
from band.utils import TrainConfig
from band.progress import NER_Dataset

pretrained_dir = '/home/band/models'

train_config = TrainConfig(epochs=3, train_batch_size=32, eval_batch_size=32, test_batch_size=1, max_length=128,
                           learning_rate=3e-5, save_model=False)

dataset = MSRA_NER(save_path="/tmp/band")

config = BertConfig.from_pretrained(pretrained_dir, num_labels=dataset.num_labels, return_unused_kwargs=True)
tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
model = TFBertForTokenClassification.from_pretrained(pretrained_dir, config=config, from_pt=True)

ner = NER_Dataset(dataset=dataset, tokenizer=tokenizer, train_config=train_config)
model.compile(optimizer=ner.optimizer, loss=ner.loss, metrics=[ner.metric])

f1 = F1Metrics(dataset.get_labels(), validation_data=ner.valid_dataset, steps=ner.valid_steps)

history = model.fit(ner.train_dataset, epochs=train_config.epochs, steps_per_epoch=ner.test_steps, callbacks=[f1])

loss, accuracy = model.evaluate(ner.test_dataset, steps=ner.test_steps)

if train_config.save_model:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
