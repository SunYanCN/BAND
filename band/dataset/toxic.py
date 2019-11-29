"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: toxic.py
@time: 2019-11-24 21:49:03
"""

from transformers import InputExample
from band.config import DATASET_URL
from band.base import CSV_Processor, Dataset_Base, download_dataset
from band.base import text_information, label_information, load_dataset


class Toxic_Processor(CSV_Processor):

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_labels(self):
        return ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            label = [int(value) for value in line[2:]]
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Toxic(Dataset_Base):
    """
        The kaggle Toxic dataset:
        https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """

    def __init__(self, save_path: str):
        super().__init__(save_path)
        dataset_dir = download_dataset(save_path=self.save_path,
                                       dataset_name="toxic",
                                       file_name="toxic.tar.gz",
                                       dataset_url=DATASET_URL['toxic'],
                                       cache_dir=self.save_path)

        self.data_processor = Toxic_Processor()
        self.data, self.label = load_dataset(processor=self.data_processor, dataset_dir=dataset_dir)

    def dataset_information(self):
        text_information(data=self.data,
                         single_text=True, language='en', char_level=False, tokenizer='nltk')
        # label_information(data=self.data)

    def get_labels(self):
        return self.data_processor.get_labels()

    @property
    def num_labels(self):
        return len(self.get_labels())

    @property
    def train_examples_num(self):
        return len(self.data['train'])

    @property
    def test_examples_num(self):
        return len(self.data['test'])

    @property
    def eval_examples_num(self):
        return len(self.data['validation'])


if __name__ == '__main__':
    dataset = Toxic(save_path='/tmp/band')
    data1, label1 = dataset.data, dataset.label
    dataset.dataset_information()
    print(dataset.get_labels())
    print(dataset.num_labels, dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num)
