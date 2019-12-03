#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import InputExample
from band.config import DATASET_URL
from band.base import TSV_Processor, Dataset_Base, download_dataset
from band.base import text_information, label_information, load_dataset


class ChnSentiCorp_Processor(TSV_Processor):

    def get_example_from_tensor_dict(self, tensor_dict):
        pass

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ChnSentiCorp(Dataset_Base):

    def __init__(self, save_path: str):
        super().__init__(save_path)
        dataset_dir = download_dataset(save_path=self.save_path,
                                       dataset_name="chnsenticorp",
                                       file_name="chnsenticorp.tar.gz",
                                       dataset_url=DATASET_URL['chnsenticorp'],
                                       cache_dir=self.save_path)

        self.data_processor = ChnSentiCorp_Processor()
        self.data, self.label = load_dataset(processor=self.data_processor, dataset_dir=dataset_dir)

    def dataset_information(self):
        text_information(data=self.data,
                         single_text=True, language='zh', char_level=True, tokenizer=None)
        label_information(data=self.data)

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
    dataset = ChnSentiCorp(save_path='/tmp/band')
    data1, label1 = dataset.data, dataset.label
    dataset.dataset_information()
    print(dataset.get_labels())
    print(dataset.num_labels, dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num)
