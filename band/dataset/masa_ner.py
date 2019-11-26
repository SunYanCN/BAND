"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: masa_ner.py
@time: 2019-11-24 21:22:47
"""
from transformers import InputExample
from band.config import DATASET_URL
from band.base import TSV_Processor, Dataset_Base, download_dataset
from band.base import text_information, label_information, load_dataset


class MSRA_NER_Processor(TSV_Processor):

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_labels(self):
        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            label = line[1].split('\x02')
            text_a = line[0].split('\x02')
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MSRA_NER(Dataset_Base):

    def __init__(self, save_path: str):
        super().__init__(save_path)
        dataset_dir = download_dataset(save_path=self.save_path,
                                       dataset_name="msra_ner",
                                       file_name="msra_ner.tar.gz",
                                       dataset_url=DATASET_URL['msra_ner'],
                                       cache_dir=self.save_path)

        data_processor = MSRA_NER_Processor()
        self.data, self.label = load_dataset(processor=data_processor, dataset_dir=dataset_dir)

    def dataset_information(self):
        text_information(data=self.data,
                         single_text=True, language='zh', char_level=True, tokenizer=None)
        label_information(data=self.data)


if __name__ == '__main__':
    dataset = MSRA_NER(save_path='/tmp/band')
    data1, label1 = dataset.data, dataset.label
    dataset.dataset_information()