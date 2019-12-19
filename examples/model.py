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
# import tensorflow as tf

import os
from transformers import *
import tensorflow as tf
from band.model import TFBertForSequenceClassification


# pretrained_dir = "C:/Users/lenovo/Desktop/chinese_wwm_pytorch"
# pretrained_dir = "C:/Users/lenovo/Desktop/RoBERTa_zh_L12_PyTorch"
# config = BertConfig.from_pretrained(pretrained_dir)
# tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
#
# model = TFBertForSequenceClassification.from_pretrained(pretrained_dir, config, from_pt=True)
# print(model.summary())

escapes = {'t': '\t', 'n': '\n', 'r': '\r', '\\': '\\'}


def parse_field(s):
    o = ''
    if s == '\\N':
        return None
    before, sep, after = s.partition('\\')
    while sep != '':
        o += before
        if after == '':
            raise EnvironmentError
        if after[0] in escapes:
            o += escapes[after[0]]
            before, sep, after = after[1:].partition('\\')
        else:
            before, sep, after = after.partition('\\')
    else:
        o += before
        return o


s = '1\t2\\t\1'

print(parse_field(s))