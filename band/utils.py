"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: utils.py
@time: 2019-11-24 19:13:09
"""
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def text_length_info(texts: list, language='zh', char_level=True, tokenizer=None):
    data_number = len(texts)
    if char_level:
        length = [len(t) for t in texts]
    else:
        if language == 'zh' and tokenizer is None:
            import logging
            import jieba
            jieba.setLogLevel(logging.INFO)
            length = [len(jieba.lcut(t)) for t in tqdm(texts)]

        elif language == 'en' and (tokenizer is None or tokenizer == 'spacy'):
            import spacy
            try:
                nlp = spacy.load('en')
            except OSError:
                nlp = spacy.load('en_core_web_sm')
            length = [len(list(t)) for t in tqdm(nlp.pipe(texts, batch_size=1000, n_threads=5))]
        elif tokenizer == 'nltk':
            import nltk
            length = [len(nltk.word_tokenize(t)) for t in tqdm(texts)]
        else:
            raise Exception("Unsupported tokenizer!")

    max_len, min_len = max(length), min(length)
    mean_len = int(np.mean(length))
    recommended_len = int(np.mean(length) + 2 * np.std(length))
    return [data_number, max_len, min_len, mean_len, recommended_len]


def view_pb_file(pb_file_path: str):
    loaded = tf.saved_model.load(pb_file_path)
    keys = list(loaded.signatures.keys())
    print(keys)
    infer = loaded.signatures[keys[0]]
    print(infer.structured_outputs)


if __name__ == '__main__':
    zh_text = ["我爱中国。"]
    en_text = ["ILove China."]
    print(text_length_info(zh_text, language='zh', char_level=True))
    print(text_length_info(zh_text, language='zh', char_level=False))
    print(text_length_info(en_text, language='en', char_level=True))
    print(text_length_info(en_text, language='en', char_level=False, tokenizer='spacy'))
    print(text_length_info(en_text, language='en', char_level=False, tokenizer='nltk'))
