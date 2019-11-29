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
import six


def parallel_apply(func,
                   iterable,
                   workers,
                   max_queue_size,
                   callback=None,
                   dummy=False):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        callback: 处理单个输出的回调函数；
    copy from: https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue = Queue(max_queue_size), Queue()

    def worker_step(in_queue, out_queue):
        # 单步函数包装成循环执行
        while True:
            d = in_queue.get()
            r = func(d)
            out_queue.put(r)

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append(d)
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for d in iterable:
        in_count += 1
        while True:
            try:
                in_queue.put(d, block=False)
                break
            except six.moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        return results


def cut_and_get_length(tokenizer, texts):

    length = []

    def get_length(result):
        for t in result:
            length.append(len(t))

    parallel_apply(
        func=tokenizer,
        iterable=tqdm(texts),
        workers=10,
        max_queue_size=500,
        callback=get_length,
    )

    return length


def text_length_info(texts: list, language='zh', char_level=True, tokenizer=None):
    data_number = len(texts)
    if char_level:
        length = [len(t) for t in texts]
    else:
        if language == 'zh' and tokenizer is None:
            import logging
            import jieba
            jieba.setLogLevel(logging.INFO)
            length = cut_and_get_length(jieba.lcut, texts)

        elif language == 'en' and (tokenizer is None or tokenizer == 'spacy'):
            import spacy
            try:
                nlp = spacy.load('en')
            except OSError:
                nlp = spacy.load('en_core_web_sm')
            length = [len(list(t)) for t in tqdm(nlp.pipe(texts, batch_size=1000, n_threads=5))]
        elif tokenizer == 'nltk':
            import nltk
            length = cut_and_get_length(nltk.word_tokenize, texts)
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
