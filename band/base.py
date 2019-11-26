"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: base.py
@time: 2019-11-24 18:55:35
"""

import os
import prettytable as pt
import tensorflow as tf
import pandas as pd

from band.utils import text_length_info
from transformers.data.processors import DataProcessor


def download_dataset(save_path: str, dataset_name: str, file_name: str, dataset_url: str, cache_dir: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset_dir = os.path.join(save_path, 'datasets', dataset_name)

    if not os.path.exists(dataset_dir):
        file_path = tf.keras.utils.get_file(
            fname=file_name,
            origin=dataset_url,
            extract=True,
            cache_dir=cache_dir,
        )
        print("Download File to: ", file_path)
    else:
        print("Dataset {} already cached.".format(dataset_dir))

    return dataset_dir


def load_dataset(dataset_dir: str, processor):
    data = {'train': processor.get_train_examples(dataset_dir),
            'validation': processor.get_dev_examples(dataset_dir),
            'test': processor.get_test_examples(dataset_dir)}

    label = processor.get_labels()

    return data, label


def text_information(data, single_text: object = True, language: str = 'zh', char_level: bool = True, tokenizer=None) -> object:
    TableA = pt.PrettyTable()
    TableA.field_names = ["Information", "Data Number", "Text MaxLen", "Text Min_len", "Text Mean_len",
                          "Recommended Len"]

    for data_type, data_value in data.items():
        text_a = [t.text_a for t in data_value]
        TableA.add_row(
            [data_type] + text_length_info(text_a, language=language, char_level=char_level, tokenizer=tokenizer))

    TableA.add_row(['Text A Info'] + [""] * (len(TableA.field_names) - 1))

    print(TableA)

    if not single_text:
        TableB = pt.PrettyTable()
        TableB.field_names = ["Information", "Data Number", "Text MaxLen", "Text Min_len", "Text Mean_len",
                              "Recommended Len"]

        for data_type, data_value in data.items():
            text_b = [t.text_b for t in data_value]
            TableB.add_row(
                [data_type] + text_length_info(text_b, language=language, char_level=char_level, tokenizer=tokenizer))

        TableB.add_row(['Text B Info'] + [""] * (len(TableB.field_names) - 1))

        print(TableB)


def label_information(data):
    Table = pt.PrettyTable()

    from itertools import chain
    from collections import Counter
    Table.field_names = []
    for data_type, data_value in data.items():
        labels = [t.label for t in data_value]
        labels = list(chain.from_iterable(labels))
        label_info = dict(Counter(labels))
        if not Table.field_names:
            Table.field_names = ['Information'] + list(label_info.keys())
        Table.add_row([data_type] + [label_info[k] for k in Table.field_names[1:]])

    Table.add_row(['Label Info'] + [""] * (len(Table.field_names) - 1))
    print(Table)


class Dataset_Base(object):

    def __init__(self, save_path: str = '~/.band'):
        self.save_path = save_path

    def dataset_information(self):
        raise NotImplementedError


class TSV_Processor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        raise NotImplementedError

    def _create_examples(self, lines, set_type):
        raise NotImplementedError


class CSV_Processor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        raise NotImplementedError

    def _create_examples(self, lines, set_type):
        raise NotImplementedError

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        csv_data = pd.read_csv(input_file)
        return csv_data
